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

# write custom exception/overrides functions here

import torch

from aimet_torch.qc_quantize_op import QcQuantizeWrapper
from aimet_torch import elementwise_ops

from redefined_modules.diffusers.models.attention import CrossAttention, AttentionBlock
from redefined_modules.transformers.models.clip.modeling_clip import CLIPAttention, CLIPTextEmbeddings

def cross_attn_exceptions(module, attn_part, bits):
    if attn_part == "q":
        module.baddbmm_1.input_quantizers[0].enabled = True
        module.baddbmm_1.input_quantizers[0].bitwidth = int(bits)
    elif attn_part == "k":
        module.baddbmm_1.input_quantizers[1].enabled = True
        module.baddbmm_1.input_quantizers[1].bitwidth = int(bits)
    elif attn_part == "sm":
        pass 
        # module.matmul_1.input_quantizers[0].enabled = True
        # module.matmul_1.input_quantizers[0].bitwidth = int(bits)
    elif attn_part == "v":
        module.matmul_1.input_quantizers[1].enabled = True
        module.matmul_1.input_quantizers[1].bitwidth = int(bits)
    elif attn_part == "as":
        module.softmax_1.input_quantizers[0].enabled = True
        module.softmax_1.input_quantizers[0].bitwidth = int(bits)
    else:
        raise ValueError(f'No exception implemented for {cur_exept}')

def cross_attn_exceptions_newmha(module, attn_part, bits):
    num_heads = 8
    if attn_part == "q":
        pass
        """
        output of conv/Q-proj is directly connected ot matmul
        """
        # for i in range(num_heads):
            # module.matmul_1[i].input_quantizers[0].enabled = True
            # module.matmul_1[i].input_quantizers[0].bitwidth = int(bits)
    elif attn_part == "k":
        for i in range(num_heads):
            module.matmul_1[i].input_quantizers[1].enabled = True
            module.matmul_1[i].input_quantizers[1].bitwidth = int(bits)
    elif attn_part == "sm":
        pass
        # for i in range(num_heads):
        #     module.matmul_2[i].input_quantizers[0].enabled = True
        #     module.matmul_2[i].input_quantizers[0].bitwidth = int(bits)
    elif attn_part == "v":
        for i in range(num_heads):
            module.matmul_2[i].input_quantizers[1].enabled = True
            module.matmul_2[i].input_quantizers[1].bitwidth = int(bits)
    elif attn_part == "as":
        for i in range(num_heads):
            module.softmax_1[i].input_quantizers[0].enabled = True
            module.softmax_1[i].input_quantizers[0].bitwidth = int(bits)
    else:
        raise ValueError(f'No exception implemented for {cur_exept}')


def clip_attn_exceptions(module, attn_part, bits):
    if attn_part == "q":
        module.bmm_1.input_quantizers[0].enabled = True
        module.bmm_1.input_quantizers[0].bitwidth = int(bits)
    elif attn_part == "k":
        module.bmm_1.input_quantizers[1].enabled = True
        module.bmm_1.input_quantizers[1].bitwidth = int(bits)
    elif attn_part == "sm":
        module.bmm_2.input_quantizers[1].enabled = True
        module.bmm_2.input_quantizers[1].bitwidth = int(bits)
    elif attn_part == "v":
        module.bmm_2.input_quantizers[0].enabled = True
        module.bmm_2.input_quantizers[0].bitwidth = int(bits)
    elif attn_part == "as":
        module.softmax.input_quantizers[0].enabled = True
        module.softmax.input_quantizers[0].bitwidth = int(bits)
        module.softmax.output_quantizers[0].enabled = True
        module.softmax.output_quantizers[0].bitwidth = int(bits)
    else:
        raise ValueError(f'No exception implemented for {cur_exept}')

def clip_embedding_exceptions(module):
    # 'Embedding or Gather' need to have the same settings for input[0] and output[0]
    output_quantizer = module.token_embedding.output_quantizers[0]
    for _, param_quantizer in module.token_embedding.param_quantizers.items():
        param_quantizer.enabled = output_quantizer.enabled
        param_quantizer.use_symmetric_encodings = output_quantizer.use_symmetric_encodings
        param_quantizer.bitwidth = output_quantizer.bitwidth

def attn_exceptions(module, attn_part, bits):
    # identical to cross attn
    """
    exceptions for AttentionBlock
    """
    try:
        cross_attn_exceptions(module, attn_part, bits)
    except:
        print(f"This layer ({module}) probabily will not be used: the number of input quantizers of Add, Matmul is not initialized properly because the layer is not included in the forward pass (probably encoder of VAE)")
    pass

def qnn_input_exceptions(qsim_model, exceptions, args):
    """
    cross attention exceptions
    """
    count = 0
    for name, module in qsim_model.named_modules():
        if isinstance(module, (AttentionBlock, CrossAttention, CLIPAttention)):
            for exception in exceptions.split("_"):
                # check for layer
                cur_exept = exception.split(':')
                if len(cur_exept) == 2:
                    layer = int(cur_exept[0])
                    if layer == count:
                        exception = cur_exept[1]
                        print(f"Apply {exception} to layer {name}")
                    else:
                        continue

                cur_exept = exception.split('=')
                if len(cur_exept) == 2:
                    attn_part, bits = cur_exept
                    if isinstance(module, AttentionBlock):
                        attn_exceptions(module, attn_part, bits)
                    elif isinstance(module, CrossAttention):
                        if args.replace_attn_singlehead_conv:
                            cross_attn_exceptions_newmha(module, attn_part, bits)
                        else:
                            cross_attn_exceptions(module, attn_part, bits)
                    elif isinstance(module, CLIPAttention):
                        clip_attn_exceptions(module, attn_part, bits)
        elif isinstance(module, CLIPTextEmbeddings):
            clip_embedding_exceptions(module)
            count += 1
    print(f"Set to exception {exceptions} for {count} attention modules.")

def disable_dropouts(quant_sim, config):
    for name, module in quant_sim.model.named_modules():
        if isinstance(module, QcQuantizeWrapper) and isinstance(module._module_to_wrap, torch.nn.Dropout):
            module.output_quantizers[0].enabled = False

def matmul_exceptions(quant_sim, config):
    if getattr(config, "use_symmetric_matmul", False):
        print("EXCEPTIONS:: Use symmetric quantizer for the second input of matmul")
        for name, module in quant_sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper) and isinstance(module._module_to_wrap, elementwise_ops.MatMul):
                if len(module.input_quantizers) >= 2:
                    module.input_quantizers[1].use_symmetric_encodings = True

def layernorm_exceptions(quant_sim, config):
    if getattr(config, "use_asymmetric_layernorm_weights", False):
        print("EXCEPTIONS:: Use asymmetric weight quantizer for layernorm")
        for name, module in quant_sim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper) and isinstance(module._module_to_wrap, torch.nn.LayerNorm):
                module.param_quantizers["weight"].use_symmetric_encodings = False

def qnn_exceptions(quant_sim, config):
    group_norm_exceptions(quant_sim, config)
    disable_dropouts(quant_sim, config)
    matmul_exceptions(quant_sim, config)
    layernorm_exceptions(quant_sim, config)
    # maybe matmul comes here

def qnn_overrides(quant_sim, config):
    softmax_fixed_encodings(quant_sim, config)
    sigmoid_fixed_encodings(quant_sim, config)


def group_norm_exceptions(quant_sim, config):
    if getattr(config, "gn_exceptions", False):
        print("EXCEPTIONS:: group norms' weight and biases will be asymmetric 16 bit")
        for name, module in quant_sim.model.named_modules():
            if isinstance(module, (QcQuantizeWrapper)):
                if isinstance(module._module_to_wrap, torch.nn.GroupNorm):
                    for _, param_quantizer in module.param_quantizers.items():
                        param_quantizer.enabled = True
                        param_quantizer.use_symmetric_encodings = False
                        param_quantizer.bitwidth = 16


def _fix_encodings_to_0_1(quant_sim, target):
    for name, module in quant_sim.model.named_modules():
        if isinstance(module, QcQuantizeWrapper) and isinstance(module._module_to_wrap, target):
            enc = module.output_quantizers[0].encoding
            if not enc:
                print(f"WARNING:: {name} does not have encodings, min/max ranges of this layer will not fixed to 0~1")
            else:
                enc.delta = 1 / (2**enc.bw - 1)
                enc.offset = 0.
                enc.min = 0.
                enc.max = 1.
                module.output_quantizers[0].freeze_encoding() # not required but just to ensure


def softmax_fixed_encodings(quant_sim, config):
    if getattr(config, "softmax_encoding_override", False):
        print("EXCEPTIONS:: softmax output quantizers will have 0~1 fixed range")
        _fix_encodings_to_0_1(quant_sim, torch.nn.Softmax)


def sigmoid_fixed_encodings(quant_sim, config):
    if getattr(config, "silu_sigmoid_encoding_override", False):
        print("EXCEPTIONS:: sigmoid output quantizers will have 0~1 fixed range")
        _fix_encodings_to_0_1(quant_sim, torch.nn.Sigmoid)




