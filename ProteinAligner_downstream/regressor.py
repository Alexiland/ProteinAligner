import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
from utils import chopping, padding, onehot_encoding, onehot_decoding
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MyRegressor(nn.Module):
    def __init__(self, hidden_size=1280):
        super(MyRegressor, self).__init__()

        self.fc0 = nn.Linear(hidden_size, 100)
        self.fc1 = nn.Linear(100, 1)

    def forward(self, x): # x is the output embedding of esm encoder
        x = self.fc0(x)
        x = nn.functional.relu(x)
        x = self.fc1(x)
        return x