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

from tqdm.auto import tqdm
import torch
from diffusers import DPMSolverMultistepScheduler


def run_diffusion_steps(model, condition_emb, seed, diffusion_steps):
    guidance_scale = 7.5
    h, w = 512, 512
    bsz = 1
    generator = torch.manual_seed(seed)
    latents = torch.randn((bsz, model.in_channels, h // 8, w // 8), generator=generator).to(model.device)

    scheduler = DPMSolverMultistepScheduler(beta_start=0.00085,
                                            beta_end=0.012,
                                            beta_schedule="scaled_linear",
                                            num_train_timesteps=1000)
    scheduler.set_timesteps(diffusion_steps)
    latents = latents * scheduler.init_noise_sigma
    
    for t in scheduler.timesteps:
        uncond, cond = condition_emb.chunk(2)
        latent_model_input = scheduler.scale_model_input(latents, t)
        time_emb = model.get_time_embedding(t, bsz)
        noise_pred_cond = model(latent_model_input, time_emb, encoder_hidden_states=cond)[0]
        noise_pred_uncond = model(latent_model_input, time_emb, encoder_hidden_states=uncond)[0]
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    return latents


def eval_unet(model, condition_emb, seed, diffusion_steps):
    output_latents = []
    for emb in tqdm(condition_emb):
        with torch.no_grad():
            output_latents.append(run_diffusion_steps(model, emb, seed, diffusion_steps))
    return output_latents

