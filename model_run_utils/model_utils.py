import shutil
import os
import torch
import numpy as np
from tokenizers import Tokenizer
from diffusers import DPMSolverMultistepScheduler
import subprocess


def create_scheduler(user_step):
    scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_start=0.00085,
                                            beta_end=0.012, beta_schedule="scaled_linear")
    scheduler.set_timesteps(user_step)
    return scheduler


def create_tokenizer(tokenizer_max_length):
    tokenizer = Tokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Setting max length to tokenizer_max_length
    tokenizer.enable_truncation(tokenizer_max_length)
    tokenizer.enable_padding(pad_id=49407, length=tokenizer_max_length)
    return tokenizer


def run_tokenizer(tokenizer, prompt):
    token_ids = tokenizer.encode(prompt).ids
    token_ids = np.array(token_ids, dtype=np.float32)

    return token_ids


def run_scheduler(scheduler, noise_pred_uncond, noise_pred_text, latent_in, timestep, user_text_guidance):
    # Convert all inputs from NHWC to NCHW
    noise_pred_uncond = np.transpose(noise_pred_uncond, (0, 3, 1, 2)).copy()
    noise_pred_text = np.transpose(noise_pred_text, (0, 3, 1, 2)).copy()
    latent_in = np.transpose(latent_in, (0, 3, 1, 2)).copy()

    # Convert all inputs to torch tensors
    noise_pred_uncond = torch.from_numpy(noise_pred_uncond)
    noise_pred_text = torch.from_numpy(noise_pred_text)
    latent_in = torch.from_numpy(latent_in)

    # Merge noise_pred_uncond and noise_pred_text based on user_text_guidance
    noise_pred = noise_pred_uncond + user_text_guidance * \
        (noise_pred_text - noise_pred_uncond)
    latent_out = scheduler.step(
        noise_pred, timestep, latent_in).prev_sample.numpy()

    # Convert latent_out from NCHW to NHWC
    latent_out = np.transpose(latent_out, (0, 2, 3, 1)).copy()

    return latent_out


def run_qnn_net_run(model_context, input_data_list):
    # Define tmp directory path for intermediate artifacts
    tmp_dirpath = os.path.abspath('tmp')
    os.makedirs(tmp_dirpath, exist_ok=True)

    # Dump each input data from input_data_list as raw file
    # and prepare input_list_filepath for qnn-net-run
    input_list_text = ''
    for index, input_data in enumerate(input_data_list):
        raw_file_path = f'{tmp_dirpath}/input_{index}.raw'
        input_data.tofile(raw_file_path)
        input_list_text += raw_file_path + ' '

    # Create input_list_filepath and add prepared input_list_text into this file
    input_list_filepath = f'{tmp_dirpath}/input_list.txt'
    with open(input_list_filepath, 'w') as f:
        f.write(input_list_text)

    SDK_dir = os.getenv('QNN_SDK_ROOT')
    net_run_binary = os.path.join(
        SDK_dir, 'bin/aarch64-windows-msvc/qnn-net-run.exe')

    cmd = [f'{net_run_binary}', '--retreive_context', 'f{model_context}', '--backend',
           'bin/QnnHtp.dll', '--input_list', f'{input_list_filepath}', '--output_dir', f'{tmp_dirpath}']

    ret = subprocess.run(cmd, capture_output=False, text=True)

    if ret != 0:
        raise ValueError(f'qnn_net_run failed for {model_context}')

    output_data = np.fromfile(
        f'{tmp_dirpath}/Result_0/output_1.raw', dtype=np.float32)

    # cleanup
    shutil.rmtree(tmp_dirpath)
    return output_data


def run_text_encoder(context_path, input_data):
    output_data = run_qnn_net_run(
        context_path, [input_data])
    # Output of Text encoder should be of shape (1, 77, 768)
    output_data = output_data.reshape((1, 77, 768))
    return output_data


def run_unet(context_path, input_data_1, input_data_2, input_data_3):
    output_data = run_qnn_net_run(context_path, [
                                  input_data_1, input_data_2, input_data_3])
    # Output of UNet should be of shape (1, 64, 64, 4)
    output_data = output_data.reshape((1, 64, 64, 4))
    return output_data


def run_vae(context_path, input_data):
    output_data = run_qnn_net_run(
        context_path, [input_data])

    output_data = np.clip(output_data*255.0, 0.0, 255.0).astype(np.uint8)
    output_data = output_data.reshape((512, 512, 3))
    return output_data
