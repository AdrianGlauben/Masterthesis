import torch
import torch.nn as nn
import numpy as np



class SPMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = dataset[idx]
        X = np.array(data[0:4], dtype=np.int32)
        y_1 = data[0] / (1 + data[2])
        y_2 = np.sqrt(data[3]) * data[1] / (1 + data[2])
        return X, np.array([y_1, y_2], dtype=np.int32)



class SimplePM(nn.Module):
    def __init__(self, input_size = 4, hidden_size = 64, output_size = 2):
        super(SimplePM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
