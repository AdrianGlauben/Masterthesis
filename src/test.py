import pickle
from evaluator_c4 import load_pickle
from connect_board import board
from encoder_decoder_c4 import encode_board
import numpy as np
import os
import itertools
from copy import deepcopy
import pm_net
import torch
from ppo_data_generation import play_game
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from MCTS_c4 import UCT_search
from alpha_net_c4 import ConnectNet

base_path = './data/round_robin/models/'
model_path = 'base_model/a0_model_iter42.pth.tar'

path = os.path.join(base_path, model_path)

model = ConnectNet(6)
model.load_state_dict(torch.load(path)['state_dict'])

pm_id = 'ConvPM_All'
pm_path = f'planning_model/{pm_id}/{pm_id}_iter_30.pth.tar'
pmp = os.path.join(base_path, pm_path)

pm = pm_net.get_pm(pm_id)
pm.load_state_dict(torch.load(pmp)['state_dict'])

model.cuda()
pm.cuda()

cboard = board()

root = UCT_search(cboard, 200, model, 1.1, c=3, planning_model=pm, move_history=[])

dist = root.child_number_visits

plt.bar(np.arange(7), dist, width=0.9)

plt.xlabel('Action Index')
plt.ylabel('Visit Count')
plt.title('PPO: CRN QV/MH')

plt.savefig(f'./data/round_robin/results/graphs/PPO_{pm_id}_dist.png')
plt.show()


exit()

def erInf(x):
    pi = 3.1415926535897
    a = 8 * (pi - 3) / (3 * pi * (4 - pi))
    y = np.log(1 - x * x)
    z = 2 / (pi * a) + y / 2

    ret = np.sqrt(np.sqrt(z * z - y / a) - z)

    if x < 0:
        return -ret
    return ret

def phiInv(p):
    return np.sqrt(2) * erInf(2 * p - 1)

def diff(p):
    return -400 * np.log10(1 / p - 1)

def get_error(wins, losses, draws):
    n = wins + losses + draws
    w = wins / n
    l = losses / n
    d = draws / n
    m_mu = w + d / 2

    devW = w * (1 - m_mu)**2
    devL = l * (0 - m_mu)**2
    devD = d * (0.5 - m_mu)**2

    m_stdev = np.sqrt(devW + devL + devD) / np.sqrt(n)

    muMin = m_mu + phiInv(0.025) * m_stdev
    muMax = m_mu + phiInv(0.975) * m_stdev
    return (diff(muMax) - diff(muMin)) / 2

file = 'im_models'
target_path = f'./data/round_robin/results/graphs/{file}_error_bar.png'

with open(f'./data/round_robin/results/{file}', 'rb') as pkl_file:
    results = pickle.load(pkl_file)

for result in results:
    print(result['name'])
    print(result['wins'], result['losses'], result['draws'])
    print('')
exit()

elos = [int(d['elo']) for d in results]
print(results[-1])
x = [0, 10, 20, 30, 40]
x_ticks = ['0', '10', '20', '30', 'A0 Base Model']

error = [get_error(d['wins'], d['losses'], d['draws']) for d in results]
print(error)

plt.ylabel('Elo Rating')
plt.xlabel('Training Iterations')

plt.title('PPO Training: CRN QV/MH')

plt.xticks(x, x_ticks)
barlist = plt.bar(x, elos, yerr=error, width=5)
barlist[-1].set_color('tab:orange')
plt.bar_label(barlist)
#plt.tick_params(axis='x', which='major', labelsize=6)

# plt.savefig(target_path)
#
# plt.show()


exit()

tensor_1 = torch.tensor([[2, 2], [3, 2]], dtype=torch.float32)

tensor_2 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
action_mask = torch.tensor([[1, 1], [0, 1]])
labels = [-1, 1]
labels = torch.tensor([[l]*2 for l in labels])

tensor_1 = torch.nn.functional.log_softmax(tensor_1, dim=1).exp()
tensor_2 = torch.nn.functional.log_softmax(tensor_2, dim=1).exp()

r = tensor_1 / tensor_2
r_clipped = torch.clamp(r, min=0.8, max=1.2)

min = torch.min(r * labels, r_clipped * labels)
minimum = torch.minimum(r * labels, r_clipped * labels)
print(min)
print(minimum)
exit()

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
