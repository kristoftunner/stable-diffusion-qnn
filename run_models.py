import torch
from diffusers import UNet2DConditionModel
from diffusers.models.embeddings import get_timestep_embedding
import numpy as np
from model_run_utils.model_utils import run_tokenizer, run_vae, \
    run_text_encoder, run_unet, run_scheduler, create_scheduler, create_tokenizer
from model_run_utils.env_setup import setup_env, check_user_inputs
import cv2


def get_time_embedding(timestep):
    timestep = torch.tensor([timestep])
    t_emb = get_timestep_embedding(timestep, 320, True, 0)

    emb = time_embeddings(t_emb).detach().numpy()
    return emb


def get_timestep(step, scheduler):
    return np.int32(scheduler.timesteps.numpy()[step])


if __name__ == '__main__':

    net_run_binary = setup_env()
    user_prompt = "decorated modern country house interior, 8 k, light reflections"

    # User defined seed value
    user_seed = np.int64(1.36477711e+14)

    # User defined step value, any integer value in {20, 50}
    user_step = 20

    # User define text guidance, any float value in [5.0, 15.0]
    user_text_guidance = 7.5

    check_user_inputs(user_seed, user_step, user_text_guidance)

    time_embeddings = UNet2DConditionModel.from_pretrained(
        'runwayml/stable-diffusion-v1-5', subfolder='unet', cache_dir='./cache/diffusers').time_embedding

    # Define Tokenizer output max length (must be 77)
    tokenizer = create_tokenizer(77)
    scheduler = create_scheduler(user_step)

    uncond_tokens = run_tokenizer("")
    cond_tokens = run_tokenizer(user_prompt)

    # Run Text Encoder on Tokens
    uncond_text_embedding = run_text_encoder(uncond_tokens)
    user_text_embedding = run_text_encoder(cond_tokens)

    # Initialize the latent input with random initial latent
    random_init_latent = torch.randn(
        (1, 4, 64, 64), generator=torch.manual_seed(user_seed)).numpy()
    latent_in = random_init_latent.transpose((0, 2, 3, 1)).copy()

    # Run the loop for user_step times
    for step in range(user_step):
        print(f'Step {step} Running...')

        # Get timestep from step
        timestep = get_timestep(step, scheduler)

        # Run U-net for const embeddings
        unconditional_noise_pred = run_unet(
            latent_in, get_time_embedding(timestep), uncond_text_embedding)

        # Run U-net for user text embeddings
        conditional_noise_pred = run_unet(
            latent_in, get_time_embedding(timestep), user_text_embedding)

        # Run Scheduler
        latent_in = run_scheduler(
            unconditional_noise_pred, conditional_noise_pred, latent_in, timestep)

    # Run VAE
    output_image = run_vae(latent_in)
    cv2.imwrite('output_image.png', output_image)