# /usr/bin/env python3.5
# -*- mode: python -*-

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
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
# =============================================================================
# Not a contribution

"""PyTorch Huggingface model redefined with elementwise/functionals replaced with class definitions"""

# pylint: skip-file

import math

import torch
from packaging import version
from torch import nn, Tensor

from transformers.utils import logging
from aimet_torch import elementwise_ops

logger = logging.get_logger(__name__)

if version.parse(torch.__version__) >= version.parse("1.4"):
    gelu = torch.nn.GELU

if version.parse(torch.__version__) >= version.parse("1.7"):
    silu = torch.nn.SiLU

if version.parse(torch.__version__) >= version.parse("1.9"):
    mish = torch.nn.Mish


def linear_act(x):
    return x


class Identity(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """
    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)


class Mish(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))


class QcSiLU(nn.Module):
    def __init__(self):
        super(QcSiLU, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        #self.mult = elementwise_ops.Multiply()
        self.mult = Identity()

    def forward(self, input: Tensor):
        #return self.mult(input, self.sigmoid(input))
        return self.mult(input * self.sigmoid(input))


ACT2FN = {
    "relu": torch.nn.ReLU,
    # "silu": silu,
    "silu": QcSiLU,
    "swish": torch.nn.Hardswish,
    "gelu": gelu,
    "gelu_new": gelu,
    "tanh": torch.nn.Tanh,
    "mish": mish,
    "linear": linear_act,
    "sigmoid": torch.nn.Sigmoid,
    "quick_gelu": QuickGELUActivation,
    "mish": Mish,
}


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]()
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")
