import pickle
from evaluator_c4 import load_pickle
from connect_board import board
from encoder_decoder_c4 import encode_board
import numpy as np
import os
import itertools
from copy import deepcopy
from pm_net import ConvPM, ConvPMDataset
import torch

values = [-0.085, 0.07, 0.056, -0.005, -0.023, 0.012, 0.036, -0.056, -0.8, -0.9, -0.85]
v_mean = 0
v_m2 = 0
n = 0
for v in values:
    n += 1
    v_mean_old = v_mean
    v_mean += (v - v_mean)/n
    v_m2 += (v - v_mean_old) * (v - v_mean)

print(v_m2/n)
print(np.var([]))
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
