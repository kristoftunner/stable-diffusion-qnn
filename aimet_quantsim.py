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

import os
import exceptions
from tqdm.auto import tqdm

import torch

from aimet_torch import onnx_utils
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.onnx_utils import OnnxExportApiArgs
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters

from stable_diff_pipeline import run_text_encoder, run_diffusion_steps, run_vae_decoder


class ExtendedOnnxExportApiArgs(OnnxExportApiArgs):
    def __init__(self, opset_version=None, input_names=None, output_names=None,
                 verbose=False, use_external_data_format=False):
        OnnxExportApiArgs.__init__(self, opset_version=opset_version, input_names=input_names,
                                   output_names=output_names)
        self.use_external_data_format = use_external_data_format
        self.verbose = verbose

    @property
    def kwargs(self):
        return dict({'use_external_data_format': self.use_external_data_format, 'verbose': self.verbose},
                    **super(ExtendedOnnxExportApiArgs, self).kwargs)


def export_quantsim_model(qsim, output_path, dummy_input, filename_prefix, verbose=False, opset_version=11,
                          use_external_data_format=False, input_names=None, output_names=None):

    onnx_utils.update_all_onnx_nodes_name = False
    onnx_utils.simplify_onnx_model = False

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    onnx_api_args = ExtendedOnnxExportApiArgs(verbose=verbose, opset_version=opset_version,
                                              use_external_data_format=use_external_data_format,
                                              input_names=input_names, output_names=output_names)

    # cpu
    if isinstance(dummy_input, tuple):
        dummy_input = tuple([d.cpu() for d in dummy_input])
    else:
        dummy_input = dummy_input.cpu()
    device = qsim.model.device
    qsim.model.cpu()
    qsim.export(path=output_path, filename_prefix=filename_prefix, dummy_input=dummy_input,
                onnx_export_args=onnx_api_args)
    print(f"ONNX saved at {output_path}")
    qsim.model.to(device)


def export_text_encoder(model, tokens):
    print("Exporting Text Encoder ----")
    dummy_input = tokens[0][0].to(model.model.device)
    export_quantsim_model(model, '_exports_/text_encoder/onnx', dummy_input, 'text_encoder',
                          input_names=['input_1'], output_names=['output_1'])


def export_unet(model, embeddings):
    print("Exporting UNET ----")
    dummy_input = unet_dummy_input(model.model, embeddings)
    export_quantsim_model(model, '_exports_/unet/onnx', dummy_input, 'unet',
                          use_external_data_format=True,
                          input_names=['input_1', 'input_2', 'input_3'], output_names=['output_1'])


def export_vae(model, latents):
    print("Exporting VAE ----")
    dummy_input = latents[0]
    export_quantsim_model(model, '_exports_/vae_decoder/onnx', dummy_input, 'vae_decoder',
                          input_names=['input_1'], output_names=['output_1'])


def export_all_models(te_sim, unet_sim, vae_sim, tokens, embeddings, latents):

    export_text_encoder(te_sim, tokens)
    export_unet(unet_sim, embeddings)
    export_vae(vae_sim, latents)


def apply_adaround_te(fp_model, adaround_data, config):
    OUTPUT_PATH = '_exports_/adaround_model'

    num_iter = config.adaround_iter_text_encoder
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    def _dummy_fw(model, train_data):
        # only adapt adaround for conditional inputs
        return model(train_data[0])

    dummy_input = adaround_data[0][0].to(fp_model.device)
    params = AdaroundParameters(data_loader=adaround_data,
                                num_batches=len(adaround_data),
                                default_num_iterations=num_iter,
                                forward_fn=_dummy_fw)

    ada_model = Adaround.apply_adaround(fp_model,
                                        dummy_input,
                                        params,
                                        path=OUTPUT_PATH,
                                        filename_prefix="parameter",
                                        default_param_bw=config.parameter_bit_width,
                                        default_quant_scheme=config.quant_scheme)

    # save
    torch.save(ada_model.state_dict(), os.path.join(
        OUTPUT_PATH, "state_dict.pt"))
    quant_sim = QuantizationSimModel(model=ada_model, quant_scheme=config.quant_scheme, dummy_input=dummy_input,
                                     default_output_bw=config.activation_bit_width,
                                     default_param_bw=config.parameter_bit_width,
                                     in_place=config.in_place,  # this will change model if True
                                     config_file=config.config_file)

    quant_sim.set_and_freeze_param_encodings(
        encoding_path=os.path.join(OUTPUT_PATH, "parameter.encodings"))
    return quant_sim


def calibrate_te(te_sim, tokens, config):

    def _set_exceptions(sim):
        exceptions.qnn_exceptions(sim, config)
        exceptions.qnn_input_exceptions(sim.model,
                                        config.text_encoder_exception_type.replace("text_encoder_attn_", ""), config)

    def _forward_pass_calibration_samples(model, eval_data):
        for text_input, uncond_input in tqdm(eval_data):
            with torch.no_grad():
                _ = run_text_encoder(model, text_input)
                _ = run_text_encoder(model, uncond_input)

    _set_exceptions(te_sim)
    te_sim.compute_encodings(_forward_pass_calibration_samples, tokens)

    return te_sim


def unet_dummy_input(unet_model, embeddings):
    # latent[1,4,64,64], time_emb[1,1280], embed[1,77,768]
    batch_size, h, w = 1, 512, 512
    dummy_latents = torch.randn(
        (batch_size, unet_model.in_channels, h // 8, w // 8)).to(unet_model.device)
    time_emb_dim = unet_model.config["block_out_channels"][0] * 4
    dummy_time_emb = torch.randn(
        (batch_size, time_emb_dim)).to(unet_model.device)
    dummy_input = embeddings[0].to(unet_model.device)[1:, :, :]
    return dummy_latents, dummy_time_emb, dummy_input


def calibrate_unet(model, embeddings, config):

    def _set_exceptions(sim):
        exceptions.qnn_exceptions(sim, config)
        if config.unet_exception_type.startswith("UNET_attn_"):
            exceptions.qnn_input_exceptions(sim.model,
                                            config.unet_exception_type.replace("UNET_attn_", ""), config)

    def _forward_pass_calibration_samples(model, embeddings):
        for embedding in tqdm(embeddings):
            with torch.no_grad():
                latent = run_diffusion_steps(model, embedding)

    dummy_input = unet_dummy_input(model, embeddings)

    quant_sim = QuantizationSimModel(model=model, quant_scheme=config.quant_scheme, dummy_input=dummy_input,
                                     default_output_bw=config.activation_bit_width,
                                     default_param_bw=config.parameter_bit_width,
                                     in_place=config.in_place,  # this will change model if True
                                     config_file=config.config_file)
    _set_exceptions(quant_sim)

    quant_sim.compute_encodings(_forward_pass_calibration_samples, embeddings)
    return quant_sim


def calibrate_vae(model, latents, config):

    def _set_exceptions(sim):
        exceptions.qnn_exceptions(sim, config)
        if config.vae_exception_type.startswith("VAE_attn_"):
            exceptions.qnn_input_exceptions(sim.model,
                                            config.vae_exception_type.replace("VAE_attn_", ""), config)

    def _forward_pass_calibration_samples(model, latents):
        for latent in tqdm(latents):
            with torch.no_grad():
                img = run_vae_decoder(model, latent)

    dummy_input = latents[0]
    quant_sim = QuantizationSimModel(model=model, quant_scheme=config.quant_scheme, dummy_input=dummy_input,
                                     default_output_bw=config.activation_bit_width,
                                     default_param_bw=config.parameter_bit_width,
                                     in_place=config.in_place,  # this will change model if True
                                     config_file=config.config_file)
    _set_exceptions(quant_sim)
    quant_sim.compute_encodings(_forward_pass_calibration_samples, latents)

    return quant_sim
