import torch
from torch.utils.data import DataLoader

class JointDataLoader:
    def __init__(self, dataloader1: DataLoader, dataloader2: DataLoader):
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
        self.iter1 = iter(dataloader1)
        self.iter2 = iter(dataloader2)
    
    def __iter__(self):
        return self

    def __next__(self):
        if len(self.dataloader1) < len(self.dataloader2):
            try:
                batch1 = next(self.iter1)
            except StopIteration:
                self.iter1 = iter(self.dataloader1)
                batch1 = next(self.iter1)
        else:
            batch1 = next(self.iter1)

        if len(self.dataloader2) < len(self.dataloader1):
            try:
                batch2 = next(self.iter2)
            except StopIteration:
                self.iter2 = iter(self.dataloader2)
                batch2 = next(self.iter2)
        else:
            batch2 = next(self.iter2)
            
        return batch1, batch2

    def __len__(self):
        return max(len(self.dataloader1), len(self.dataloader2))