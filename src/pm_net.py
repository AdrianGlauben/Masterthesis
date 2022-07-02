import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from connect_board import board
from encoder_decoder_c4 import encode_board


def get_dataset(pm_id, dataset, expansions, ppo=False):
    if pm_id == 'SPM_base':
        ds = SPMDataset(dataset, ppo)
    elif pm_id == 'SPM_QVar':
        ds = SPMDataset_QVar(dataset, ppo)
    elif pm_id == 'ConvPM_base':
        ds = ConvPMDataset(dataset, expansions, ppo)
    elif pm_id == 'ConvPM_QVar':
        ds = ConvPMDataset_QVar(dataset, expansions, ppo)
    elif pm_id == 'ConvPM_MH':
        ds = ConvPMDataset_MH(dataset, expansions, ppo=ppo)
    elif pm_id == 'ConvPM_All':
        ds = ConvPMDataset_All(dataset, expansions, ppo=ppo)
    return ds


def get_pm(pm_id):
    if pm_id == 'SPM_base':
        pm = SimplePM()
    elif pm_id == 'SPM_QVar':
        pm = SimplePM_QVar()
    elif pm_id == 'ConvPM_base':
        pm = ConvPM()
    elif pm_id == 'ConvPM_QVar':
        pm = ConvPM_QVar()
    elif pm_id == 'ConvPM_MH':
        pm = ConvPM_MH()
    elif pm_id == 'ConvPM_All':
        pm = ConvPM_All()
    return pm


class PPO_Loss(torch.nn.Module):
    def __init__(self):
        super(PPO_Loss, self).__init__()

    def forward(self, outputs, outputs_old, labels, mask, clip_epsilon):
        r_theta = outputs / outputs_old * labels
        r_theta_clipped = torch.clamp(r_theta, min=1-clip_epsilon, max=1+clip_epsilon) * labels
        l = torch.minimum(r_theta, r_theta_clipped)
        l_masked = l * mask
        return l.mean()



class SPMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, ppo=False):
        self.dataset = dataset
        self.ppo = ppo

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        X = np.concatenate(data[0:4], dtype=np.float32)
        X = np.concatenate([X, data[6]], dtype=np.float32)
        if self.ppo:
            label = 1 if data[7] == data[9] else -1
            return torch.from_numpy(X), label, torch.from_numpy(data[6])
        return X, data[8]



class SPMDataset_QVar(torch.utils.data.Dataset):
    def __init__(self, dataset, ppo=False):
        self.dataset = dataset
        self.ppo = ppo

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        X = np.concatenate([data[0], data[1], data[2], data[3], data[5], data[6]], axis=None, dtype=np.float32)
        if self.ppo:
            label = 1 if data[7] == data[9] else -1
            return torch.from_numpy(X), label, torch.from_numpy(data[6])
        return X, data[8]



class SimplePM(nn.Module):
    def __init__(self, input_size = 7*4+1, hidden_size = 512, output_size = 7):
        super(SimplePM, self).__init__()
        self.id = 'SPM_base'
        self.in_fc = nn.Linear(input_size, hidden_size)
        self.hidden_fc1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.hidden_fc2 = nn.Linear(int(hidden_size/2), int(hidden_size/4))
        self.out_fc = nn.Linear(int(hidden_size/4), output_size)


    def forward(self, x):
        x = self.in_fc(x)
        x = F.relu(x)
        x = self.hidden_fc1(x)
        x = F.relu(x)
        x = self.hidden_fc2(x)
        x = F.relu(x)
        x = self.out_fc(x)
        return x



class SimplePM_QVar(nn.Module):
    def __init__(self, input_size = 7*5+1, hidden_size = 512, output_size = 7):
        super(SimplePM_QVar, self).__init__()
        self.id = 'SPM_QVar'
        self.in_fc = nn.Linear(input_size, hidden_size)
        self.hidden_fc1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.hidden_fc2 = nn.Linear(int(hidden_size/2), int(hidden_size/4))
        self.out_fc = nn.Linear(int(hidden_size/4), output_size)


    def forward(self, x):
        x = self.in_fc(x)
        x = F.relu(x)
        x = self.hidden_fc1(x)
        x = F.relu(x)
        x = self.hidden_fc2(x)
        x = F.relu(x)
        x = self.out_fc(x)
        return x



class ConvPMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, expansions, ppo=False):
        self.dataset = dataset
        self.expansions = expansions
        self.ppo = ppo

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
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

        for d in data[6]:
            X[plane_idx] += d                   # Action mask
            plane_idx += 1

        if self.ppo:
            label = 1 if data[7] == data[9] else -1
            return torch.from_numpy(X), label, torch.from_numpy(data[6])
        return X, data[8]



class ConvPMDataset_MH(torch.utils.data.Dataset):
    def __init__(self, dataset, expansions, mh_size=4, ppo=False):
        self.dataset = dataset
        self.expansions = expansions
        self.mh_size = mh_size
        self.ppo = ppo


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

        for i, d in enumerate(data[6]):
            X[34+i] += d                   # Action mask

        if self.ppo:
            label = 1 if data[7] == data[9] else -1
            return torch.from_numpy(X), label, torch.from_numpy(data[6])
        return X, data[8]



class ConvPMDataset_QVar(torch.utils.data.Dataset):
    def __init__(self, dataset, expansions, ppo=False):
        self.dataset = dataset
        self.expansions = expansions
        self.ppo = ppo

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        X = np.zeros([36,6,7], dtype=np.float32)

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

        for d in data[6]:
            X[plane_idx] += d                   # Action mask
            plane_idx += 1

        if self.ppo:
            label = 1 if data[7] == data[9] else -1
            return torch.from_numpy(X), label, torch.from_numpy(data[6])
        return X, data[8]



class ConvPMDataset_All(torch.utils.data.Dataset):
    def __init__(self, dataset, expansions, mh_size=4, ppo=False):
        self.dataset = dataset
        self.expansions = expansions
        self.mh_size = mh_size
        self.ppo = ppo


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
        X = np.zeros([36 + 3 * self.mh_size,6,7], dtype=np.float32)

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
        for i, d in enumerate(data[5]):
            X[34 + i] += d

        for i, d in enumerate(data[6]):
            X[41 + i] += d

        if self.ppo:
            label = 1 if data[7] == data[9] else -1
            return torch.from_numpy(X), label, torch.from_numpy(data[6])
        return X, data[8]



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



class ConvPM_MH(nn.Module):
    def __init__(self, num_res_blocks=2):
        super(ConvPM_MH, self).__init__()
        self.id = 'ConvPM_MH'
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



class ConvPM_QVar(nn.Module):
    def __init__(self, num_res_blocks=2):
        super(ConvPM_QVar, self).__init__()
        self.id = 'ConvPM_QVar'
        self.conv = ConvBlock(36)
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
        super(ConvPM_All, self).__init__()
        self.id = 'ConvPM_All'
        self.conv = ConvBlock(48)
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
