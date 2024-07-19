import torch
from torch.utils.data import DataLoader

class JointAlternateDataLoader:
    def __init__(self, dataloader1: DataLoader, dataloader2: DataLoader):
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
        self.dataloader_iter1 = iter(dataloader1)
        self.dataloader_iter2 = iter(dataloader2)
        self.current_dataloader = 1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_dataloader == 1:
            try:
                batch = next(self.dataloader_iter1)
                self.current_dataloader = 2
                return batch, "1"
            except StopIteration:
                self.dataloader_iter1 = iter(self.dataloader1)
                return next(self)
        else:
            try:
                batch = next(self.dataloader_iter2)
                self.current_dataloader = 1
                return batch, "2"
            except StopIteration:
                self.dataloader_iter2 = iter(self.dataloader2)
                return next(self)

    def __len__(self):
        return len(self.dataloader1) + len(self.dataloader2)