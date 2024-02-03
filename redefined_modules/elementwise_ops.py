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

from typing import Optional
import torch

class Interpolate(torch.nn.Module):
    """ Interpolate module for a functional interpolate"""

    def __init__(self, 
        mode: str = "nearest", 
        align_corners: Optional[bool] = None, 
        scale_factor: Optional[float] = None
    ):
        super(Interpolate, self).__init__()
        self.mode = mode
        self.align_corners = align_corners
        self.scale_factor = scale_factor

    def forward(self, *inputs) -> torch.Tensor:
        """
        Forward-pass routine for interpolate op
        """
        x = inputs[0]
        if inputs[1]:
            size = inputs[1].tolist()
        else:
            size = None
        out = torch.nn.functional.interpolate(
            input=x, size=size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )
        return out

class BMM(torch.nn.Module):
    """
    Batch Matmul module for functional bmm (torch.bmm)
    """
    @staticmethod
    def forward(input, mat2):
        return torch.bmm(input, mat2)

class BADDBMM(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, batch1, batch2, beta=1, alpha=1, out=None):
        return torch.baddbmm(input=input, batch1=batch1, batch2=batch2, beta=beta, alpha=alpha)

class Concat(torch.nn.Module):
    """ Concat module for a functional concat"""
    @staticmethod
    def forward(*args, **kwargs) -> torch.Tensor:
        """
        Forward-pass routine for concat op
        """
        return torch.cat(*args, **kwargs)