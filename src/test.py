import pickle
from evaluator_c4 import load_pickle
from connect_board import board
from encoder_decoder_c4 import encode_board
import numpy as np
import os
import itertools
from copy import deepcopy

mh = [1, 2, 3, 4, 5, 6, 5, 1, 2, 2]
cboard = board()
positions = []
for m in mh:
    cboard.drop_piece(m)
    pos = encode_board(cboard).transpose(2,0,1)
    print(pos)
    positions.extend(pos)

if len(positions) > 12:
    input = positions[-12:]
else:
    input = positions

print(input)

X = np.zeros([12,6,7])

for i, pos in enumerate(input):
    X[i] = pos

print(X)

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

print(dataset['expansions'])
print(dataset['cpuct'])
print(len(dataset['data']))
