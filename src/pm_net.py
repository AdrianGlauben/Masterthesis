import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from connect_board import board
from encoder_decoder_c4 import encode_board



class SPMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, c):
        self.dataset = dataset
        self.c = c

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        X = np.concatenate(data[0:4], dtype=np.float32)
        y_1 = data[0] / (1 + data[2])
        y_2 = np.sqrt(data[3]) * data[1] / (1 + data[2])
        puct = y_1 + self.c * y_2
        return X, np.argmax(puct)



class SPMDataset_QVar(torch.utils.data.Dataset):
    def __init__(self, dataset, c):
        self.dataset = dataset
        self.c = c

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        X = np.concatenate(data[0:4], dtype=np.float32)
        X = np.concatenate(X, data[5], dtype=np.float32)
        y_1 = data[0] / (1 + data[2])
        y_2 = np.sqrt(data[3]) * data[1] / (1 + data[2])
        puct = y_1 + self.c * y_2
        return X, np.argmax(puct)



class SimplePM(nn.Module):
    def __init__(self, input_size = 7*3+1, hidden_size = 64, output_size = 7):
        super(SimplePM, self).__init__()
        self.id = 'SPM_base'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



class SimplePM_QVar(nn.Module):
    def __init__(self, input_size = 7*4+1, hidden_size = 64, output_size = 7):
        super(SimplePM, self).__init__()
        self.id = 'SPM_QVar'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



class ConvPMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, c, expansions):
        self.dataset = dataset
        self.c = c
        self.expansions = expansions

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self):
        data = self.dataset[idx]
        X = np.zeros([22,6,7], dtype=np.float32)

        # 22 Planes for standard PUCT inputs
        plane_idx = 0
        for d in data[0]:
            X[plane_idx] += d / self.expansions   # Normalized Total Child Value
            plane_idx += 1

        for d in data[1]:
            X[plane_idx] += d                    # Child Prior
            plane_idx += 1

        for d in data[2]:
            X[plane_idx] += d / self.expansions
            plane_idx += 1

        X[plane_idx] += data[3][0] / self.expansions   # Normalized Parent Visists

        y_1 = data[0] / (1 + data[2])
        y_2 = np.sqrt(data[3]) * data[1] / (1 + data[2])
        puct = y_1 + self.c * y_2
        return X, np.argmax(puct)



class ConvPMDataset_MH(torch.utils.data.Dataset):
    def __init__(self, dataset, c, expansions, mh_size=4):
        self.dataset = dataset
        self.c = c
        self.expansions = expansions
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
        X = np.zeros([22 + 3 * self.mh_size,6,7], dtype=np.float32)

        # 22 Planes for standard PUCT inputs
        plane_idx = 0
        for d in data[0]:
            X[plane_idx] += d / self.expansions   # Normalized Total Child Value
            plane_idx += 1

        for d in data[1]:
            X[plane_idx] += d                    # Child Prior
            plane_idx += 1

        for d in data[2]:
            X[plane_idx] += d / self.expansions
            plane_idx += 1

        X[plane_idx] += data[3][0] / self.expansions   # Normalized Parent Visists
        plane_idx += 1

        # mh_size * 3 Planes for Move History
        positions = self._get_move_history_representation(data[4])
        for pos in positions:
            X[plane_idx] += pos
            plane_idx += 1

        y_1 = data[0] / (1 + data[2])
        y_2 = np.sqrt(data[3]) * data[1] / (1 + data[2])
        puct = y_1 + self.c * y_2
        return X, np.argmax(puct)



class ConvPMDataset_QVar(torch.utils.data.Dataset):
    def __init__(self, dataset, c, expansions):
        self.dataset = dataset
        self.c = c
        self.expansions = expansions

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self):
        data = self.dataset[idx]
        X = np.zeros([29,6,7], dtype=np.float32)

        # 22 Planes for standard PUCT inputs
        plane_idx = 0
        for d in data[0]:
            X[plane_idx] += d / self.expansions   # Normalized Total Child Value
            plane_idx += 1

        for d in data[1]:
            X[plane_idx] += d                    # Child Prior
            plane_idx += 1

        for d in data[2]:
            X[plane_idx] += d / self.expansions
            plane_idx += 1

        X[plane_idx] += data[3][0] / self.expansions   # Normalized Parent Visists
        plane_idx += 1

        # 7 Planes for Q-Variance
        for d in data[5]:
            X[plane_idx] += d
            plane_idx += 1

        y_1 = data[0] / (1 + data[2])
        y_2 = np.sqrt(data[3]) * data[1] / (1 + data[2])
        puct = y_1 + self.c * y_2
        return X, np.argmax(puct)



class ConvPMDataset_All(torch.utils.data.Dataset):
    def __init__(self, dataset, c, expansions, mh_size=4):
        self.dataset = dataset
        self.c = c
        self.expansions = expansions
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
        X = np.zeros([29 + 3 * self.mh_size,6,7], dtype=np.float32)

        # 22 Planes for standard PUCT inputs
        plane_idx = 0
        for d in data[0]:
            X[plane_idx] += d / self.expansions   # Normalized Total Child Value
            plane_idx += 1

        for d in data[1]:
            X[plane_idx] += d                    # Child Prior
            plane_idx += 1

        for d in data[2]:
            X[plane_idx] += d / self.expansions
            plane_idx += 1

        X[plane_idx] += data[3][0] / self.expansions   # Normalized Parent Visists
        plane_idx += 1

        # mh_size * 3 Planes for Move History
        positions = self._get_move_history_representation(data[4])
        for pos in positions:
            X[plane_idx] += pos
            plane_idx += 1

        # 7 Planes for Q-Variance
        for d in data[5]:
            X[plane_idx] += d
            plane_idx += 1

        y_1 = data[0] / (1 + data[2])
        y_2 = np.sqrt(data[3]) * data[1] / (1 + data[2])
        puct = y_1 + self.c * y_2
        return X, np.argmax(puct)



class ConvBlock(nn.Module):
    def __init__(self, input_size):
        super(ConvBlock, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(self.input_size, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s):
        s = s.view(-1, self.input_size, 6, 7)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s



class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out



class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(128, 3, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3*6*7, 32)
        self.fc2 = nn.Linear(32, 7)

    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, 3*6*7)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = self.fc2(v)
        return v



class ConvPM(nn.Module):
    def __init__(self, num_res_blocks=2):
        super(ConvPM, self).__init__()
        self.id = 'ConvPM_base'
        self.conv = ConvBlock(22)
        for block in range(num_res_blocks):
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock()
        self.num_res_blocks = num_res_blocks

    def forward(self,s):
        s = self.conv(s)
        for block in range(self.num_res_blocks):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s



class ConvPM_MH(nn.Module):
    def __init__(self, num_res_blocks=2):
        super(ConvPM, self).__init__()
        self.id = 'ConvPM_base'
        self.conv = ConvBlock(34)
        for block in range(num_res_blocks):
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock()
        self.num_res_blocks = num_res_blocks

    def forward(self,s):
        s = self.conv(s)
        for block in range(self.num_res_blocks):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s



class ConvPM_QVar(nn.Module):
    def __init__(self, num_res_blocks=2):
        super(ConvPM, self).__init__()
        self.id = 'ConvPM_QVar'
        self.conv = ConvBlock(29)
        for block in range(num_res_blocks):
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock()
        self.num_res_blocks = num_res_blocks

    def forward(self,s):
        s = self.conv(s)
        for block in range(self.num_res_blocks):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s



class ConvPM_All(nn.Module):
    def __init__(self, num_res_blocks=2):
        super(ConvPM, self).__init__()
        self.id = 'ConvPM_base'
        self.conv = ConvBlock(41)
        for block in range(num_res_blocks):
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock()
        self.num_res_blocks = num_res_blocks

    def forward(self,s):
        s = self.conv(s)
        for block in range(self.num_res_blocks):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s
