import pickle
from evaluator_c4 import load_pickle
from connect_board import board
from encoder_decoder_c4 import encode_board
import numpy as np
import os
import itertools
from copy import deepcopy
from pm_net import ConvPM, ConvPMDataset, ConvPM_All, SimplePM
import torch
from ppo_data_generation import play_game

arr1 = np.array([[1, 2], [3, 4], [0, 0]])
arr2 = np.array([[5, 6], [7,8]])
x = np.concatenate(arr1[0:2])
print(np.concatenate([arr1[0:2], arr2], axis=None, dtype=np.float32))
exit()

tensor_1 = torch.tensor([[2, 2], [3, 2]], dtype=torch.float32)
tensor_2 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
action_mask = torch.tensor([[1, 1], [0, 1]])
labels = [1, -1]
labels = torch.tensor([[l]*2 for l in labels])
r = tensor_1 / tensor_2 * labels
r_clipped = torch.clamp(r, min=0.8, max=1.2) * labels
min = torch.minimum(r, r_clipped)
masked_min = min * action_mask
result = masked_min.mean()

print(r)
print(r_clipped)
print(min)
print(masked_min)
print(result)
exit()

dataset = {'data': []}
data_path = "./data/pm_data/game_data/"

for idx,file in enumerate(os.listdir(data_path)):
    filename = os.path.join(data_path,file)
    with open(filename, 'rb') as fo:
        game_data = pickle.load(fo, encoding='bytes')
        dataset['expansions'] = game_data['expansions']
        dataset['cpuct'] = game_data['cpuct']
        dataset['data'].extend(game_data['data'])
    break

convpm_dataset = ConvPMDataset(dataset['data'], dataset['cpuct'], dataset['expansions'])
net = ConvPM()
X, y = convpm_dataset[-1]
input_tensor = torch.tensor(X)
print(net(input_tensor))
