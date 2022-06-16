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

dataset = {'data': []}
data_path = "./data/pm_data/game_data/"

for idx,file in enumerate(os.listdir(data_path)):
    filename = os.path.join(data_path,file)
    with open(filename, 'rb') as fo:
        game_data = pickle.load(fo, encoding='bytes')
        dataset['expansions'] = game_data['expansions']
        dataset['cpuct'] = game_data['cpuct']
        dataset['data'].extend(game_data['data'])

print(len(dataset['data']))
exit()

convpm_dataset = ConvPMDataset(dataset)
net = ConvPM()
X, y = convpm_dataset[-1]
print(X)
print(X.shape)
input_tensor = torch.tensor(X)
print(net(input_tensor))
