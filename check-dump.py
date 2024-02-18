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

import numpy as np


def sqnr(signal, noisy, eps=1e-10):
    noise = signal - noisy
    exp_noise = (noise ** 2).mean() + eps
    exp_signal = (signal ** 2).mean()
    sqnr = (exp_signal / exp_noise)
    sqnr_db = 10 * np.log10(sqnr)
    return sqnr_db


# before = np.load('_exports_/fp32.npy', allow_pickle=True)
# after = np.load('_exports_/int8.npy', allow_pickle=True)
# keys = ['text_embeddings', 'latent', 'img']
#
# print(','.join(keys))
# for i, (b, a) in enumerate(zip(before, after)):
#    print(','.join([str(sqnr(a[key], b[key])) for key in keys]))

src = np.load('_exports_/fp32.npy', allow_pickle=True)
outdir = '_exports_'

with open('te_input_list.txt', 'wt') as input_list:
    te_inputs = src[0]['token_ids'].astype(np.float32)
    for i in range(len(te_inputs)):
        te_inputs[i].tofile(f'{outdir}/te_input_{i+1}.bin')
        print(f'{outdir}/te_input_{i+1}.bin', file=input_list)

with open('unet_input_list.txt', 'wt') as input_list:
    for i, (latent, time_emb, hidden) in enumerate(src[0]['unet_inputs']):
        latent.transpose(0, 2, 3, 1).tofile(f'{outdir}/unet_input_{i+1}_1.bin')
        time_emb.tofile(f'{outdir}/unet_input_{i+1}_2.bin')
        hidden.tofile(f'{outdir}/unet_input_{i+1}_3.bin')
        print(f'{outdir}/unet_input_{i+1}_1.bin {outdir}/unet_input_{i+1}_2.bin {outdir}/unet_input_{i+1}_3.bin', file=input_list)

with open('vae_input_list.txt', 'wt') as input_list:
    vae_iputs = src[0]['latent']
    vae_iputs.transpose(0, 2, 3, 1).tofile(f'{outdir}/vae_input_1.bin')
    print(f'{outdir}/vae_input_1.bin', file=input_list)
