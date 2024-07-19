import os
from typing import Optional, Callable

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from models.imagebind_model import ModalityType
import data
import torch
from torch.utils.data.dataloader import default_collate


class FastaTextDataset(Dataset):
    def __init__(self, text_root: str, fasta_root: str, transform: Optional[Callable] = None,
                 split: str = 'train', train_size: float = 0.99, random_seed: int = 42, device: str = 'cpu'):
        self.text_root = text_root
        self.fasta_root = fasta_root
        self.transform = transform
        self.device = device

        self.paths = []

        # sequence is of O15318.fasta, corresponding text is O15318.txt, pdb is O15318_001.pdb...O15318_xxx.pdb
        # since pdb is a subset of sequence, we use pdb as the key to match sequence file
        for filename in os.listdir(self.text_root):
            if filename.endswith('.txt'):
                # maybe change to filename[:X] is better? since the sequence name is fix-length
                self.paths.append(filename.split('.')[0])

        # Split dataset
        train_set, test_set = train_test_split(self.paths, train_size=train_size, random_state=random_seed)

        if split == 'train':
            self.paths = train_set
        elif split == 'test':
            self.paths = test_set
        else:
            raise ValueError(f"Invalid split argument. Expected 'train' or 'test', got {split}")

        self.text_paths = [os.path.join(self.text_root, path+'.txt') for path in self.paths]
        self.fasta_paths = [os.path.join(self.fasta_root, path+'.pt') for path in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        fasta_path, text_path = self.fasta_paths[index], self.text_paths[index]
        
        fastas = data.load_and_transform_sequence_data([fasta_path], self.device, to_tensor=False)
        texts = data.load_and_transform_text([text_path], self.device)  # torch.Size([1, 77])

        return fastas, ModalityType.SEQUENCE, texts, ModalityType.TEXT

    def collater(self, samples):
        new_samples = []
        max_len_seq_dim0 = -1
        max_len_seq_dim2 = -1
        for sample in samples:
            seq_embeddings = sample[0][0]
            seq_shape_dim0 = seq_embeddings.shape[0]
            max_len_seq_dim0 = max(max_len_seq_dim0, seq_shape_dim0)
            seq_shape_dim2 = seq_embeddings.shape[2]
            max_len_seq_dim2 = max(max_len_seq_dim2, seq_shape_dim2)

        for sample in samples:
            seq_embeddings = sample[0][0]
            seq_shape_dim0 = seq_embeddings.shape[0]
            seq_shape_dim2 = seq_embeddings.shape[2]
            pad_seq = (0, max_len_seq_dim2 - seq_shape_dim2, 0, 0, 0, max_len_seq_dim0 - seq_shape_dim0)
            arr_seq_padded = torch.nn.functional.pad(seq_embeddings, pad_seq, mode='constant', )

            # text_embedding = sample[2].repeat(1024,1).transpose(1,0).float()

            new_samples.append(([arr_seq_padded], sample[1], sample[2], sample[3]))

        return default_collate(new_samples)
