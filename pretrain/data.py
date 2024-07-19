#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

import torch
import torch.nn as nn
import torchaudio
from PIL import Image
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo

from models.multimodal_preprocessors import SimpleTokenizer

DEFAULT_AUDIO_FRAME_SHIFT_MS = 10  # in milliseconds

BPE_PATH = "bpe/bpe_simple_vocab_16e6.txt.gz"

def load_and_transform_text(text, device):
    if text is None:
        return None
    tokenizer = SimpleTokenizer(bpe_path=BPE_PATH)
    tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens

def load_and_transform_structure_data(pdb_paths, device, to_tensor=True):
    if pdb_paths is None:
        return None
    pdb_transform = transforms.Compose([])
    pdb_outputs = []
    for pdb_path in pdb_paths:
        if to_tensor:
            pdb = pdb_transform(torch.load(pdb_path)).to(device)
            pdb_outputs.append(pdb)
        else:
            pdb = pdb_transform(torch.load(pdb_path))
            pdb_outputs.append(pdb)
    return pdb_outputs if not to_tensor else torch.stack(pdb_outputs, dim=0)


def load_and_transform_sequence_data(fasta_paths, device, to_tensor=True):
    if fasta_paths is None:
        return None
    fasta_transform = transforms.Compose([])
    fasta_outputs = []
    for fasta_path in fasta_paths:
        if to_tensor:
            fasta = fasta_transform(torch.load(fasta_path)).to(device)
            fasta_outputs.append(fasta)
        else:
            fasta = fasta_transform(torch.load(fasta_path))
            fasta_outputs.append(fasta)
    return fasta_outputs if not to_tensor else torch.stack(fasta_outputs, dim=0)