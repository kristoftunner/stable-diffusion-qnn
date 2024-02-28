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

from aimet_quantsim import export_all_models, apply_adaround_te, export_vae, export_unet, export_text_encoder
from aimet_quantsim import calibrate_vae, calibrate_unet, calibrate_te
from stable_diff_pipeline import run_the_pipeline, save_image, replace_mha_with_sha_blocks
from stable_diff_pipeline import run_tokenizer, run_text_encoder, run_diffusion_steps, run_vae_decoder
from redefined_modules.diffusers.models.vae import AutoencoderKLDecoder
from redefined_modules.diffusers.models.unet_2d_condition import UNet2DConditionModel
from redefined_modules.transformers.models.clip.modeling_clip import CLIPTextModel
from transformers import CLIPTokenizer
from tqdm import tqdm
import logging
import torch
from argparse import Namespace
import json
import os
import sys
sys.path.insert(0, '.')
sys.setrecursionlimit(10000)


def get_sqnr(org_out, quant_out, in_db=True, eps=1e-10):
    quant_error = org_out - quant_out
    exp_noise = quant_error.pow(2).view(quant_error.shape[0], -1).mean(1) + eps
    exp_signal = org_out.pow(2).view(org_out.shape[0], -1).mean(1)
    sqnr = (exp_signal / exp_noise).mean()
    sqnr_db = 10 * torch.log10(sqnr)
    return sqnr_db if in_db else sqnr


def load_text_encoder(cache_dir=None):
    device = 'cuda'
    dtype = torch.float
    print("Loading pre-trained TextEncoder model")
    text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14',
                                                 torch_dtype=dtype, cache_dir=cache_dir).to(device)
    text_encoder.config.return_dict = False
    return text_encoder


def load_unet(cache_dir=None):
    device = 'cuda'
    dtype = torch.float
    print("Loading pre-trained UNET model")
    unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5',
                                                subfolder="unet", revision='main', torch_dtype=dtype,
                                                cache_dir=cache_dir).to(device)
    unet.config.return_dict = False
    return unet


def load_vae(cache_dir=None):
    device = 'cuda'
    dtype = torch.float
    print("Loading pre-trained VAE model")
    vae = AutoencoderKLDecoder.from_pretrained('runwayml/stable-diffusion-v1-5',
                                               revision='main', subfolder="vae", torch_dtype=dtype,
                                               cache_dir=cache_dir).to(device)
    vae.config.return_dict = False
    return vae


def load_pretrained_models(cache_dir=None):

    text_encoder = load_text_encoder(cache_dir)
    unet = load_unet(cache_dir)
    vae = load_vae(cache_dir)

    return unet, text_encoder, vae


def free_cuda_resources(resources):
    for r in resources:
        del r
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    script_config = {
        'runFloatPipeline': True,
        'textEncoder': {
            'quantize': False,
            'embeddingsOutputFolder': '_exports_/embeddings',
        },
        'unet': {
            'quantize': False,
            'embeddingsInputFolder': '_exports_/embeddings',
            'latentsOutputFolder': '_exports_/latents',
        },
        'vae': {
            'quantize': False,
            'latentsInputFolder': '_exports_/latents',
        },
    }

    with open('config.json', 'rt') as f:
        model_config = Namespace(**json.load(f))

    cache_dir = "./_data_/cache/huggingface/diffusers"

    if script_config['runFloatPipeline']:
        logging.info("Running the float pipeline")
        unet, text_encoder, vae = load_pretrained_models(cache_dir=cache_dir)

        replace_mha_with_sha_blocks(unet)

        tokenizer = CLIPTokenizer.from_pretrained(
            'openai/clip-vit-large-patch14', cache_dir=cache_dir)

        prompt = "San Diego downtown and people on the beach"
        image = run_the_pipeline(prompt, unet, text_encoder,
                                 vae, tokenizer, test_name='fp32')
        save_image(image.squeeze(0), 'generated.png')
        free_cuda_resources([unet, text_encoder, vae])

    with open(model_config.calibration_prompts, "rt") as f:
        logging.info(
            f"Loading prompts from {model_config.calibration_prompts}")
        prompts = f.readlines()

    if script_config['textEncoder']['quantize']:
        text_encoder = load_text_encoder(cache_dir)
        tokenizer = CLIPTokenizer.from_pretrained(
            'openai/clip-vit-large-patch14', cache_dir=cache_dir)

        tokens = [run_tokenizer(tokenizer, prompt) for prompt in prompts]

        logging.info("Applying adaround on text encoder")
        text_encoder_sim = apply_adaround_te(
            text_encoder, tokens, model_config)

        logging.info("Calibrating text encoder")
        text_encoder_sim = calibrate_te(text_encoder_sim, tokens, model_config)

        logging.info("Exporting text encoder")
        export_text_encoder(text_encoder_sim, tokens)
        embeddings = [torch.cat([run_text_encoder(text_encoder_sim.model, uncond),
                                 run_text_encoder(text_encoder_sim.model, cond)]) for cond, uncond in tokens]

        #############################################################################################
        # ADAROUND PERFORMANCE EVALUATION (can be removed later)
        # test with in_place: false, apply_adaround_text_encoder=true or false
        # with adaround: 17.59, w.o. adaround: 15.119
        org_emb = [torch.cat([run_text_encoder(text_encoder, uncond),
                              run_text_encoder(text_encoder, cond)]) for cond, uncond in tokens]
        sqnr = get_sqnr(torch.stack(org_emb), torch.stack(embeddings))
        logging.info(
            f"Adaround applied: {model_config.apply_adaround_text_encoder} | text encoder SQNR: {sqnr}")
        #############################################################################################

        # save the embeddings
        logging.info('Saving embeddings')
        for i, emb in enumerate(embeddings):
            torch.save(emb, os.path.join(
                script_config['textEncoder']['embeddingsOutputFolder'], f'embedding_{i}.pt'))

        free_cuda_resources(
            [text_encoder, text_encoder_sim, embeddings, org_emb])

    if script_config['unet']['quantize']:

        unet = load_unet(cache_dir)
        replace_mha_with_sha_blocks(unet)

        # load embeddings
        embedding_files = os.listdir(
            script_config['textEncoder']['embeddingsOutputFolder'])
        embeddings = [torch.load(os.path.join(
            script_config['textEncoder']['embeddingsOutputFolder'], f)) for f in embedding_files]

        logging.info("Calibrating unet")
        unet_sim = calibrate_unet(unet, embeddings, model_config)

        # save latents
        logging.info('Creating latents')
        latents = []
        for emb in tqdm(embeddings):
            latents.append(run_diffusion_steps(unet_sim.model, emb))

        logging.info('Saving latents')
        for i, latent in enumerate(latents):
            logging.info(f"Saving latent {i}")
            torch.save(latent, os.path.join(
                script_config['unet']['latentsOutputFolder'], f'latent_{i}.pt'))
        export_unet(unet_sim, embeddings)

        free_cuda_resources([unet, unet_sim, latents, embeddings])

    if script_config['vae']['quantize']:
        vae = load_vae(cache_dir)

        # load latents
        latent_files = os.listdir(
            script_config['unet']['latentsOutputFolder'])
        latents = [torch.load(os.path.join(
            script_config['unet']['latentsOutputFolder'], f)) for f in latent_files]

        logging.info("Calibrating vae")
        vae_sim = calibrate_vae(vae, latents, model_config)
        export_vae(vae_sim, latents)

        logging.info('Executing vae decoder with images')
        images = []
        for latent in tqdm(latents):
            images.append(run_vae_decoder(vae_sim.model, latent))

        for i, image in enumerate(images):
            save_image(image.squeeze(0), f'generated_after_quant_{i}.png')

        # logging.info('Running the quantized pipeline')
        # image = run_the_pipeline(prompt, unet_sim.model, text_encoder_sim.model,
        #                         vae_sim.model, tokenizer, test_name='int8')
        # save_image(image.squeeze(0), 'generated_after_quant.png')

        logging.info('Exporting quantized vae decoder')
        export_vae(vae_sim, latents)
        free_cuda_resources([vae, vae_sim, latents])
