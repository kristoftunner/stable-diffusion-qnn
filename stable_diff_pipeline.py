# ==============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright 2023 Qualcomm Technologies, Inc. All rights reserved.
#  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
#
#  The party receiving this software directly from QTI (the "Recipient")
#  may use this software as reasonably necessary solely for the purposes
#  set forth in the agreement between the Recipient and QTI (the
#  "Agreement"). The software may be used in source code form solely by
#  the Recipient's employees (if any) authorized by the Agreement. Unless
#  expressly authorized in the Agreement, the Recipient may not sublicense,
#  assign, transfer or otherwise provide the source code to any third
#  party. Qualcomm Technologies, Inc. retains all ownership rights in and
#  to the software
#
#  This notice supersedes any other QTI notices contained within the software
#  except copyright notices indicating different years of publication for
#  different portions of the software. This notice does not supersede the
#  application of any third party copyright notice to that third party's
#  code.
#
#  @@-COPYRIGHT-END-@@
# ==============================================================================

import torch
import numpy
from PIL import Image
from diffusers import DPMSolverMultistepScheduler
from transformers import CLIPTokenizer

from redefined_modules.diffusers.models.attention import CrossAttention

def _np(tensor):
    return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor

def run_tokenizer(tokenizer, prompt):
    with torch.no_grad():
        text_input = tokenizer( prompt, padding="max_length",
                    max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt",)
        bsz = text_input.input_ids.shape[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer( [""] * bsz, padding="max_length", max_length=max_length, return_tensors="pt",)
    return text_input.input_ids, uncond_input.input_ids


def run_text_encoder(model, prompt):
    with torch.no_grad():
        embeddings = model(prompt.to(model.device))[0]
    return embeddings


def run_diffusion_steps(model, text_embedding, seed=3596, diffusion_steps=20, dump=None):
    guidance_scale = 7.5
    h, w = 512, 512
    bsz = 1
    generator = torch.manual_seed(seed)
    latents = torch.randn( (bsz, model.in_channels, h // 8, w // 8), generator=generator,).to(model.device)

    scheduler = DPMSolverMultistepScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    scheduler.set_timesteps(diffusion_steps)

    latents = latents * scheduler.init_noise_sigma
    with torch.no_grad():
        # do it sequentially; on-device will use in this way, and to reduce memory burden
        for t in scheduler.timesteps:
            uncond, cond = text_embedding.chunk(2)
            latent_model_input = scheduler.scale_model_input(latents, t)
            time_emb = model.get_time_embedding(t, bsz)

            if dump is not None:
                dump.append((_np(latent_model_input), _np(time_emb), _np(cond)))
                dump.append((_np(latent_model_input), _np(time_emb), _np(uncond)))

            noise_pred_cond = model(latent_model_input, time_emb, encoder_hidden_states=cond)[0]
            noise_pred_uncond = model(latent_model_input, time_emb, encoder_hidden_states=uncond)[0]

            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents


def run_vae_decoder(model, latent):
    with torch.no_grad():
        img = model(latent)[0]
    return img


def save_image(img, name):
    img = _np(img)
    img = (img * 255).round().astype("uint8")
    img = Image.fromarray(img)
    img.save(name)
    print(f"Saved {name}")

def run_the_pipeline(prompts, unet, text_encoder, vae, tokenizer, test_name, seed=1.36477711e+14):

    prompts = [prompts] if isinstance(prompts, str) else prompts
    with torch.no_grad():
        dump, unet_inputs = [], []
        for i, prompt in enumerate(prompts):
            text_input, uncond_input = run_tokenizer(tokenizer, prompt)
            cond_embeddings= run_text_encoder(text_encoder, text_input)
            uncond_embeddings= run_text_encoder(text_encoder, uncond_input)
            text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
            latent = run_diffusion_steps(unet, text_embeddings, seed=seed, dump=unet_inputs)
            img = run_vae_decoder(vae, latent)
            dump.append(
                {'prompt': prompt,
                 'token_ids': _np(torch.cat([text_input, uncond_input])),
                 'text_embeddings': _np(text_embeddings),
                 'latent': _np(latent),
                 'img': _np(img),
                 'unet_inputs': unet_inputs,
                 }
            )
        numpy.save(f'_exports_/{test_name}', dump, allow_pickle=True)


    return img

def replace_mha_with_sha_blocks(unet_model):
    print("linear layers in CrossAttention will be replaced with convs with single head")
    for name, module in unet_model.named_modules():
        if isinstance(module, CrossAttention):
            module.replace_linear_to_convs()
