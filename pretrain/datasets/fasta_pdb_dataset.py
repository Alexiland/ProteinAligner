import os
from typing import Optional, Callable

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

from models.imagebind_model import ModalityType
import data
from torch.utils.data.dataloader import default_collate
import numpy as np


class FastaPdbDataset(Dataset):
    def __init__(self, pdb_root: str, fasta_root: str, transform: Optional[Callable] = None,
                 split: str = 'train', train_size: float = 0.99, random_seed: int = 42, device: str = 'cpu'):
        self.pdb_root = pdb_root
        self.fasta_root = fasta_root
        self.transform = transform
        self.device = device

        self.paths = []
        # sequence is of O15318.fasta, corresponding text is O15318.txt, pdb is O15318_001.pdb...O15318_xxx.pdb
        # since pdb is a subset of sequence, we use pdb as the key to match sequence file
        for filename in os.listdir(self.fasta_root):
            if filename.endswith('.pt'):
                # maybe change to filename[:X] is better? since the sequence name is fix-length
                self.paths.append(filename)
                
        # Split dataset
        train_set, test_set = train_test_split(self.paths, train_size=train_size, random_state=random_seed)

        if split == 'train':
            self.paths = train_set
        elif split == 'test':
            self.paths = test_set
        else:
            raise ValueError(f"Invalid split argument. Expected 'train' or 'test', got {split}")

        self.pdb_paths = [os.path.join(self.pdb_root, path) for path in self.paths]
        self.fasta_paths = [os.path.join(self.fasta_root, path) for path in self.paths]
        
        self.processors = torch.nn.AdaptiveAvgPool2d((256, 1024))

    def __len__(self):
        return len(self.paths)
    
    def _data_prepro(self, data):
        data_tensor = data[0].transpose(1, 0)
        return [self.processors(data_tensor)]

    def __getitem__(self, index):
        fasta_path, pdb_path = self.fasta_paths[index], self.pdb_paths[index]
        fastas = data.load_and_transform_sequence_data([fasta_path], self.device, to_tensor=False)
        pdbs = data.load_and_transform_structure_data([pdb_path], self.device)
        
        if self.transform is not None:
            fasta = fastas[0]
            fastas = self.transform(fasta)
            
        # fastas, pdbs = self._data_prepro(fastas), self._data_prepro(pdbs)

        return fastas, ModalityType.SEQUENCE, pdbs, ModalityType.STRUCTURE
    
    def collater(self, samples):
        new_samples = []
        max_len_protein_dim0 = -1
        max_len_seq_dim0 = -1
        max_len_protein_dim2 = -1
        max_len_seq_dim2 = -1
        for sample in samples:
            pdb_embeddings = sample[2][0]
            shape_dim0 = pdb_embeddings.shape[0]
            max_len_protein_dim0 = max(max_len_protein_dim0, shape_dim0)
            shape_dim2 = pdb_embeddings.shape[2]
            max_len_protein_dim2 = max(max_len_protein_dim2, shape_dim2)

            seq_embeddings = sample[0][0]
            seq_shape_dim0 = seq_embeddings.shape[0]
            max_len_seq_dim0 = max(max_len_seq_dim0, seq_shape_dim0)
            seq_shape_dim2 = seq_embeddings.shape[2]
            max_len_seq_dim2 = max(max_len_seq_dim2, seq_shape_dim2)

        for sample in samples:
            pdb_embeddings = sample[2][0]
            shape_dim0 = pdb_embeddings.shape[0]
            shape_dim2 = pdb_embeddings.shape[2]
            pad1 = (0, max_len_protein_dim2 - shape_dim2, 0, 0, 0, max_len_protein_dim0 - shape_dim0)
            arr1_padded = torch.nn.functional.pad(pdb_embeddings, pad1, mode='constant', )
            
            seq_embeddings = sample[0][0]
            seq_shape_dim0 = seq_embeddings.shape[0]
            seq_shape_dim2 = seq_embeddings.shape[2]
            pad_seq = (0, max_len_seq_dim2 - seq_shape_dim2, 0, 0, 0, max_len_seq_dim0 - seq_shape_dim0)
            arr_seq_padded = torch.nn.functional.pad(seq_embeddings, pad_seq, mode='constant', )
            
            new_samples.append(([arr_seq_padded], sample[1], [arr1_padded], sample[3]))

        return default_collate(new_samples)
