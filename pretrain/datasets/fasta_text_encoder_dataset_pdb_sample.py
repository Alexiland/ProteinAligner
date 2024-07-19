import os
from typing import Optional, Callable

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from models.imagebind_model import ModalityType
import data
import torch
from torch.utils.data.dataloader import default_collate

import json
import sys
import esm
from esm.inverse_folding.util import CoordBatchConverter
from Bio import SeqIO


class FastaTextDataset(Dataset):
    def __init__(self, seq_alphabet, mapping: str, fasta_root: str, transform: Optional[Callable] = None,
                 split: str = 'train', train_size: float = 0.99, random_seed: int = 42, device: str = 'cpu'):
        self.fasta_root = fasta_root
        self.transform = transform
        self.device = device

        self.mapping = mapping
        self.seq_alphabet = seq_alphabet

        self.paths = []
        self.ids = []
        self.seq_batch_converter = seq_alphabet.get_batch_converter()

        with open(self.mapping) as f:
            self.fasta_text_mappings = json.load(f)

        for pair in self.fasta_text_mappings:
            fasta_path = os.path.join(self.fasta_root, pair["uniprot_id"]+".fasta")
            text = pair["function"]
            self.paths.append((fasta_path, text))
            self.ids.append(pair["uniprot_id"])

        # Split dataset
        train_set, test_set = train_test_split(self.paths, train_size=train_size, random_state=random_seed)

        if split == 'train':
            self.paths = train_set
        elif split == 'test':
            self.paths = test_set
        else:
            raise ValueError(f"Invalid split argument. Expected 'train' or 'test', got {split}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        fasta_path, text = self.paths[index][0], self.paths[index][1]

        fasta_sequences = SeqIO.parse(open(fasta_path), 'fasta')
        fasta = next(fasta_sequences)
        seq_data = (fasta.id, str(fasta.seq))

        function_text = text

        texts = data.load_and_transform_text([function_text], self.device)  # torch.Size([1, 77])

        return {
            "fasta_seq": fasta,
            "fasta_id": self.ids[index],
            "fasta_data": seq_data,
            "text": texts
        }

    def collate_fn(self, samples):
        # Processing FASTA entries
        fasta_batch = []
        text_batch = []
        for fasta_data in samples:
            fasta_batch.append(fasta_data["fasta_data"])
            text_batch.append(fasta_data["text"])
        fasta_batch_labels, fasta_batch_strs, fasta_batch_tokens = self.seq_batch_converter(fasta_batch)
        fasta_batch_lens = (fasta_batch_tokens != self.seq_alphabet.padding_idx).sum(1)


        return (fasta_batch_tokens, fasta_batch_lens), ModalityType.SEQUENCE, text_batch, ModalityType.TEXT
