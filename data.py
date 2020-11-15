import torch
from torch.utils.data import Dataset
import pandas as  pd
class Data(Dataset):
    def __init__(self, file):
        self.file = pd.read_csv(file)
        self.x = self.file.iloc[0 : len(self.file.iloc[0]), 0:10].values
        self.y = self.file.iloc[0 : len(self.file.iloc[0]), 10].values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]).float(), torch.tensor([self.y[idx]]).long()
