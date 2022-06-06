import torch
import torch.nn as nn
import numpy as np
from connect_board import board
from encoder_decoder_c4 import encode_board



class SPMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset['data']
        self.c = dataset['cpuct']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        X = np.array(data[0:4], dtype=np.float32)
        y_1 = data[0] / (1 + data[2])
        y_2 = np.sqrt(data[3]) * data[1] / (1 + data[2])
        return X, np.array([y_1 + self.c * y_2], dtype=np.float32)



class SimplePM(nn.Module):
    def __init__(self, input_size = 4, hidden_size = 64, output_size = 1):
        super(SimplePM, self).__init__()
        self.id = 'SPM'
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



class ConvPMDataset(torch.utils.data.Datset):
    def __init__(self, dataset, mh_size=4):
        self.dataset = dataset['data']
        self.c = dataset['cpuct']
        self.expansions = dataset['expansions']
        self.mh_size = mh_size


    def _get_move_history_representation(self, move_history):
        cboard = board()
        positions = []
        for m in move_history:
            cboard.drop_piece(m)
            pos = encode_board(cboard).transpose(2,0,1)
            positions.extend(pos)

        if len(positions) > 3 * self.mh_size:
            return positions[-3 * self.mh_size:]
        else:
            return positions


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        data = self.dataset[idx]
        X = np.zeros([4 + 3 * self.mh_size,6,7], dtype=np.float32)

        # 4 Planes for standard PUCT inputs
        X[0] += data[0] / self.expansions   # Normalized Total Child Value
        X[1] += data[1]                     # Child Prior
        X[2] += data[2] / self.expansions   # Normalized Child Visits
        X[3] += data[3] / self.expansions   # Normalized Parent Visists

        # mh_size * 3 Planes for Move History
        positions = self._get_move_history_representation(data[4])
        for i, pos in enumerate(positions):
            X[4+i] += pos

        y_1 = data[0] / (1 + data[2])
        y_2 = np.sqrt(data[3]) * data[1] / (1 + data[2])
        return X, np.array([y_1 + self.c * y_2], dtype=np.float32)



class ConvPM(nn.Module):
