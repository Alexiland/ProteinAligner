import os
from typing import Optional, Callable

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

from models.imagebind_model import ModalityType
import data
from torch.utils.data.dataloader import default_collate
import numpy as np

import json
import sys
import esm
from esm.inverse_folding.util import CoordBatchConverter
from Bio import SeqIO


class FastaPdbDataset(Dataset):
    def __init__(self, pdb_alphabet, seq_alphabet, mapping: str, pdb_root: str, fasta_root: str, chain="A", transform: Optional[Callable] = None,
                 split: str = 'train', train_size: float = 0.99, random_seed: int = 42, device: str = 'cpu'):
        self.pdb_root = pdb_root
        self.fasta_root = fasta_root
        self.transform = transform
        self.device = device
        self.mapping = mapping
        self.paths = []
        self.ids = []
        self.chain = chain

        self.pdb_alphabet = pdb_alphabet
        self.seq_alphabet = seq_alphabet

        self.pdb_batch_converter = CoordBatchConverter(pdb_alphabet)
        self.seq_batch_converter = seq_alphabet.get_batch_converter()

        with open(self.mapping) as f:
            self.fasta_pdb_mappings = json.load(f)

        for pair in self.fasta_pdb_mappings:
            fasta_path = os.path.join(self.fasta_root, pair["uniprot_id"]+".fasta")
            pdb_path = os.path.join(self.pdb_root, pair["pdb_id"].lower() + ".pdb")
            self.paths.append((fasta_path, pdb_path))
            self.ids.append((pair["uniprot_id"], pair["pdb_id"]))

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
        fasta_path, pdb_path = self.paths[index][0], self.paths[index][1]

        coords, native_seq = esm.inverse_folding.util.load_coords(pdb_path, self.chain)

        fasta_sequences = SeqIO.parse(open(fasta_path), 'fasta')
        fasta = next(fasta_sequences)
        seq_data = (fasta.id, str(fasta.seq))

        # return fasta_path, ModalityType.SEQUENCE, pdb_path, ModalityType.STRUCTURE
        return {
            "pdb_coords": coords,
            "chain": self.chain,
            "native_seq": native_seq,
            "pdb_id": self.ids[index][1],
            "fasta_seq": fasta,
            "fasta_id": self.ids[index][0],
            "fasta_data": seq_data
        }

    def collate_fn(self, samples):
        # Processing FASTA entries
        fasta_batch = []
        for fasta_data in samples:
            fasta_batch.append(fasta_data["fasta_data"])
        fasta_batch_labels, fasta_batch_strs, fasta_batch_tokens = self.seq_batch_converter(fasta_batch)
        fasta_batch_lens = (fasta_batch_tokens != self.seq_alphabet.padding_idx).sum(1)

        # # processing PDB entries
        # pdb_batch = []
        # for pdb_data in samples:
        #     pdb_batch.append((pdb_data["pdb_coords"], None, pdb_data["native_seq"]))
        # coords, confidence, strs, tokens, padding_mask = self.pdb_batch_converter(
        #     pdb_batch, device=None)

        coords = [pdb_data["pdb_coords"] for pdb_data in samples]
        native_seq = [pdb_data["native_seq"] for pdb_data in samples]
        return (fasta_batch_tokens, fasta_batch_lens), ModalityType.SEQUENCE, (coords, native_seq), ModalityType.STRUCTURE
