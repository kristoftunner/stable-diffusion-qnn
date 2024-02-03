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
from tqdm.auto import tqdm
import numpy as np

def run_vae_encoder(model, latent):
    img = model(latent)[0]
    img = img.cpu().numpy()
    img = (img * 255).round().astype("uint8")
    return img


def eval_vae_encoder(model, latents):
    output_imgs = []
    for latent in tqdm(latents):
        with torch.no_grad():
            output_imgs.append(run_vae_encoder(model, latent))
    output_imgs = np.concatenate(output_imgs, axis=0)
    output_imgs = [output_imgs[i] for i in range(output_imgs.shape[0])]
    return output_imgs

