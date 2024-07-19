#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
from functools import partial
from types import SimpleNamespace
from typing import Dict

import torch
import torch.nn as nn

from models.helpers import (EinOpsRearrange, LearnableLogitScaling, Normalize,
                            SelectElement, SelectEOSAndProject)
from models.multimodal_preprocessors import (SequPreprocessor,
                                             StruPreprocessor,
                                             TextPreprocessor, )
from models.transformer import MultiheadAttention, SimpleTransformer, ViTAttention, SeqEncoderTrunk, StrEncoderTrunk

import esm
from esm import inverse_folding

ModalityType = SimpleNamespace(
    TEXT="text",
    SEQUENCE='sequence',
    STRUCTURE='structure',
)


class ImageBindModel(nn.Module):
    def __init__(
            self,
            out_embed_dim=768,
            text_embed_dim=768,
            text_num_blocks=12,
            text_num_heads=12,
            seq_encoder=None,
            str_encoder=None,
    ):
        super().__init__()

        ##### Definition of protein encoders #####
        self.seq_encoder = seq_encoder

        self.str_encoder = str_encoder

        self.modality_preprocessors = self._create_modality_preprocessors(
            text_embed_dim,
        )

        self.modality_trunks = self._create_modality_trunks(
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
        )

        self.modality_heads = self._create_modality_heads(
            out_embed_dim,
            text_embed_dim,
            # TODO modify the embedding dim later, just for test
            sequence_embed_dim=1280,
            structure_embed_dim=512,
        )

        self.modality_postprocessors = self._create_modality_postprocessors(
            out_embed_dim
        )

        for name, param in self.str_encoder.named_parameters():
            param.requires_grad = True

        for name, param in self.seq_encoder.named_parameters():
            param.requires_grad = True
        

    def _create_modality_preprocessors(
            self,
            text_embed_dim=768,
    ):
        sequ_preprocessor = SequPreprocessor()

        stru_preprocessor = StruPreprocessor()

        text_preprocessor = TextPreprocessor(
            context_length=77,
            vocab_size=49408,
            embed_dim=text_embed_dim,
            causal_masking=True,
        )

        modality_preprocessors = {
            # TODO: define the preprocess part for sequence and structure
            ModalityType.SEQUENCE: sequ_preprocessor,
            ModalityType.STRUCTURE: stru_preprocessor,
            ModalityType.TEXT: text_preprocessor,
        }

        return nn.ModuleDict(modality_preprocessors)

    def seq_trunk(self):
        return SeqEncoderTrunk(seq_encoder=self.seq_encoder)

    def str_trunk(self):
        return StrEncoderTrunk(str_encoder=self.str_encoder)
    
    def _create_modality_trunks(
            self,
            text_embed_dim=768,
            text_num_blocks=8,
            text_num_heads=8,
    ):
        def instantiate_trunk(
                embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path
        ):
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6)
                    if pre_transformer_ln
                    else nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
            )

        modality_trunks = {}

        # TODO: define the trunks part for sequence and structure
        modality_trunks[ModalityType.SEQUENCE] = self.seq_trunk()

        modality_trunks[ModalityType.STRUCTURE] = self.str_trunk()

        modality_trunks[ModalityType.TEXT] = instantiate_trunk(
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=False,
            drop_path=0.0,
        )

        return nn.ModuleDict(modality_trunks)

    def _create_modality_heads(
            self,
            out_embed_dim,
            text_embed_dim,
            sequence_embed_dim,
            structure_embed_dim,
    ):
        modality_heads = {}
        # TODO: define the modality head part for sequence and structure
        modality_heads[ModalityType.SEQUENCE] = nn.Sequential(
            nn.LayerNorm(normalized_shape=sequence_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(sequence_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.STRUCTURE] = nn.Sequential(
            nn.LayerNorm(normalized_shape=structure_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Linear(structure_embed_dim, out_embed_dim, bias=False),
        )

        modality_heads[ModalityType.TEXT] = SelectEOSAndProject(
            proj=nn.Sequential(
                nn.LayerNorm(normalized_shape=text_embed_dim, eps=1e-6),
                nn.Linear(text_embed_dim, out_embed_dim, bias=False),
            )
        )

        return nn.ModuleDict(modality_heads)

    def _create_modality_postprocessors(self, out_embed_dim):
        modality_postprocessors = {}
        # TODO: define the postprocess part for sequence and structure
        modality_postprocessors[ModalityType.SEQUENCE] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        )

        modality_postprocessors[ModalityType.STRUCTURE] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        )

        modality_postprocessors[ModalityType.TEXT] = nn.Sequential(
            Normalize(dim=-1), LearnableLogitScaling(learnable=True)
        )

        return nn.ModuleDict(modality_postprocessors)

    def forward(self, inputs):
        outputs = {}

        for modality_key, modality_value in inputs.items():
            if modality_value is not None:
                modality_value = self.modality_preprocessors[modality_key](
                    **{modality_key: modality_value}
                )
                trunk_inputs = modality_value["trunk"]
                head_inputs = modality_value["head"]
                modality_value = self.modality_trunks[modality_key](**trunk_inputs)
                modality_value = self.modality_heads[modality_key](
                    modality_value, **head_inputs
                )
                modality_value = self.modality_postprocessors[modality_key](
                    modality_value
                )
                outputs[modality_key] = modality_value

        return outputs

def proteinbind_huge(seq_encoder=None, str_encoder=None):
    model = ImageBindModel(
        text_embed_dim=1024,
        text_num_blocks=8,
        text_num_heads=8,
        out_embed_dim=1024,
        seq_encoder=seq_encoder,
        str_encoder=str_encoder
    )

    return model


def save_module(module_dict: nn.ModuleDict, module_name: str = "",
                checkpoint_dir: str = "./.checkpoints/full", postfix: str = "_last",
                extension: str = "pth"):
    try:
        torch.save(module_dict.state_dict(),
                   os.path.join(checkpoint_dir, f"imagebind-{module_name}{postfix}.{extension}"))
        logging.info(f"Saved parameters for module {module_name} to {checkpoint_dir}.")
    except FileNotFoundError:
        logging.warning(f"Could not save module parameters for {module_name} to {checkpoint_dir}.")


def load_module(module_dict: nn.ModuleDict, module_name: str = "",
                checkpoint_dir: str = "./.checkpoints/full", postfix: str = "_last",
                extension: str = "pth"):
    try:
        module_dict.load_state_dict(torch.load(
            os.path.join(checkpoint_dir, f"imagebind-{module_name}{postfix}.{extension}")), strict=False)
        logging.info(f"Loaded parameters for module {module_name} from {checkpoint_dir}.")
    except FileNotFoundError:
        logging.warning(f"Could not load module parameters for {module_name} from {checkpoint_dir}.")
