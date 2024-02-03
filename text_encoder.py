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


def read_file(path):
    with open(path, "rt") as f:
        txt = f.readlines()
    return txt


def prepare_and_tokenize_prompt(tokenizer, prompt):
    text_input = tokenizer(prompt,
                           padding="max_length",
                           max_length=tokenizer.model_max_length,
                           truncation=True,
                           return_tensors="pt")
    bsz = text_input.input_ids.shape[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * bsz,
                             padding="max_length",
                             max_length=max_length,
                             return_tensors="pt")

    return text_input.input_ids, uncond_input.input_ids


def run_text_encoder(model, tokenizer, prompt):
    text_input, uncond_input = prepare_and_tokenize_prompt(tokenizer, prompt)
    text_embeddings = model(text_input.to(model.device))[0]
    uncond_embeddings = model(uncond_input.to(model.device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings


def eval_text_encoder(model, tokenizer, txt_data):
    text_embeddings = []
    for prompt in tqdm(txt_data):
        with torch.no_grad():
            text_embeddings.append(run_text_encoder(model, tokenizer, prompt))
    return text_embeddings

