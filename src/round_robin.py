from benchmark import play_game
from itertools import permutations
from alpha_net_c4 import ConnectNet
from argparse import ArgumentParser
import pickle
from rtpt import RTPT
import numpy as np
import os
import torch
import pm_net

MODEL_DIR = './data/round_robin/models/'
BASE_MODELS = [{'name': 'a0_model_iter_0', 'path': 'base_model/a0_model_iter0.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                {'name': 'a0_model_iter_10', 'path': 'base_model/a0_model_iter10.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                {'name': 'a0_model_iter_20', 'path': 'base_model/a0_model_iter20.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                {'name': 'a0_model_iter_30', 'path': 'base_model/a0_model_iter30.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                {'name': 'a0_model_iter_42', 'path': 'base_model/a0_model_iter42.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}]

IM_MODELS = [{'name': 'SPM_base_iter_0', 'pm_id': 'SPM_base', 'path': 'planning_model/SPM_base/SPM_base_iter_0.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                {'name': 'SPM_QVar_iter_0', 'pm_id': 'SPM_QVar', 'path': 'planning_model/SPM_QVar/SPM_QVar_iter_0.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                {'name': 'ConvPM_base_iter_0', 'pm_id': 'ConvPM_base', 'path': 'planning_model/ConvPM_base/ConvPM_base_iter_0.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                {'name': 'ConvPM_QVar_iter_0', 'pm_id': 'ConvPM_QVar', 'path': 'planning_model/ConvPM_QVar/ConvPM_QVar_iter_0.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                {'name': 'ConvPM_MH_iter_0', 'pm_id': 'ConvPM_MH', 'path': 'planning_model/ConvPM_MH/ConvPM_MH_iter_0.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                {'name': 'ConvPM_All_iter_0', 'pm_id': 'ConvPM_All', 'path': 'planning_model/ConvPM_All/ConvPM_All_iter_0.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                {'name': 'a0_model_iter_42', 'pm_id': 'SPM_base', 'path': 'base_model/a0_model_iter42.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}]

PPO_MODELS = [{'name': 'SPM_base_iter_30', 'pm_id': 'SPM_base', 'path': 'planning_model/SPM_base/SPM_base_iter_30.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                {'name': 'SPM_QVar_iter_30', 'pm_id': 'SPM_QVar', 'path': 'planning_model/SPM_QVar/SPM_QVar_iter_30.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                {'name': 'ConvPM_base_iter_30', 'pm_id': 'ConvPM_base', 'path': 'planning_model/ConvPM_base/ConvPM_base_iter_30.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                {'name': 'ConvPM_QVar_iter_30', 'pm_id': 'ConvPM_QVar', 'path': 'planning_model/ConvPM_QVar/ConvPM_QVar_iter_30.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                {'name': 'ConvPM_MH_iter_30', 'pm_id': 'ConvPM_MH', 'path': 'planning_model/ConvPM_MH/ConvPM_MH_iter_30.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                {'name': 'ConvPM_All_iter_30', 'pm_id': 'ConvPM_All', 'path': 'planning_model/ConvPM_All/ConvPM_All_iter_30.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                {'name': 'a0_model_iter_42', 'path': 'base_model/a0_model_iter42.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}]

SPM_BASE_MODELS = [{'name': 'SPM_base_iter_0', 'pm_id': 'SPM_base', 'path': 'planning_model/SPM_base/SPM_base_iter_0.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                    {'name': 'SPM_base_iter_10', 'pm_id': 'SPM_base', 'path': 'planning_model/SPM_base/SPM_base_iter_10.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                    {'name': 'SPM_base_iter_20', 'pm_id': 'SPM_base', 'path': 'planning_model/SPM_base/SPM_base_iter_20.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                    {'name': 'SPM_base_iter_30', 'pm_id': 'SPM_base', 'path': 'planning_model/SPM_base/SPM_base_iter_30.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                    {'name': 'a0_model_iter_42', 'path': 'base_model/a0_model_iter42.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}]

SPM_QVAR_MODELS = [{'name': 'SPM_QVar_iter_0', 'pm_id': 'SPM_QVar', 'path': 'planning_model/SPM_QVar/SPM_QVar_iter_0.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                    {'name': 'SPM_QVar_iter_10', 'pm_id': 'SPM_QVar', 'path': 'planning_model/SPM_QVar/SPM_QVar_iter_10.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                    {'name': 'SPM_QVar_iter_20', 'pm_id': 'SPM_QVar', 'path': 'planning_model/SPM_QVar/SPM_QVar_iter_20.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                    {'name': 'SPM_QVar_iter_30', 'pm_id': 'SPM_QVar', 'path': 'planning_model/SPM_QVar/SPM_QVar_iter_30.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                    {'name': 'a0_model_iter_42', 'path': 'base_model/a0_model_iter42.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}]

CONVPM_BASE_MODELS = [{'name': 'ConvPM_base_iter_0', 'pm_id': 'ConvPM_base', 'path': 'planning_model/ConvPM_base/ConvPM_base_iter_0.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                        {'name': 'ConvPM_base_iter_10', 'pm_id': 'ConvPM_base', 'path': 'planning_model/ConvPM_base/ConvPM_base_iter_10.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                        {'name': 'ConvPM_base_iter_20', 'pm_id': 'ConvPM_base', 'path': 'planning_model/ConvPM_base/ConvPM_base_iter_20.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                        {'name': 'ConvPM_base_iter_30', 'pm_id': 'ConvPM_base', 'path': 'planning_model/ConvPM_base/ConvPM_base_iter_30.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                        {'name': 'a0_model_iter_42', 'path': 'base_model/a0_model_iter42.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}]

CONVPM_QVAR_MODELS = [{'name': 'ConvPM_QVar_iter_0', 'pm_id': 'ConvPM_QVar', 'path': 'planning_model/ConvPM_QVar/ConvPM_QVar_iter_0.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                        {'name': 'ConvPM_QVar_iter_10', 'pm_id': 'ConvPM_QVar', 'path': 'planning_model/ConvPM_QVar/ConvPM_QVar_iter_10.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                        {'name': 'ConvPM_QVar_iter_20', 'pm_id': 'ConvPM_QVar', 'path': 'planning_model/ConvPM_QVar/ConvPM_QVar_iter_20.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                        {'name': 'ConvPM_QVar_iter_30', 'pm_id': 'ConvPM_QVar', 'path': 'planning_model/ConvPM_QVar/ConvPM_QVar_iter_30.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                        {'name': 'a0_model_iter_42', 'path': 'base_model/a0_model_iter42.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}]

CONVPM_MH_MODELS = [{'name': 'ConvPM_MH_iter_0', 'pm_id': 'ConvPM_MH', 'path': 'planning_model/ConvPM_MH/ConvPM_MH_iter_0.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                    {'name': 'ConvPM_MH_iter_10', 'pm_id': 'ConvPM_MH', 'path': 'planning_model/ConvPM_MH/ConvPM_MH_iter_10.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                    {'name': 'ConvPM_MH_iter_20', 'pm_id': 'ConvPM_MH', 'path': 'planning_model/ConvPM_MH/ConvPM_MH_iter_20.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                    {'name': 'ConvPM_MH_iter_30', 'pm_id': 'ConvPM_MH', 'path': 'planning_model/ConvPM_MH/ConvPM_MH_iter_30.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                    {'name': 'a0_model_iter_42', 'path': 'base_model/a0_model_iter42.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}]

CONVPM_ALL_MODELS = [{'name': 'ConvPM_All_iter_0', 'pm_id': 'ConvPM_All', 'path': 'planning_model/ConvPM_All/ConvPM_All_iter_0.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                        {'name': 'ConvPM_All_iter_10', 'pm_id': 'ConvPM_All', 'path': 'planning_model/ConvPM_All/ConvPM_All_iter_10.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                        {'name': 'ConvPM_All_iter_20', 'pm_id': 'ConvPM_All', 'path': 'planning_model/ConvPM_All/ConvPM_All_iter_20.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                        {'name': 'ConvPM_All_iter_30', 'pm_id': 'ConvPM_All', 'path': 'planning_model/ConvPM_All/ConvPM_All_iter_30.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}, \
                        {'name': 'a0_model_iter_42', 'path': 'base_model/a0_model_iter42.pth.tar', 'wins': 0, 'losses': 0, 'draws': 0, 'elo': 400}]


def get_pairings(idxs):
    array = []
    for i in idxs:
        for j in idxs:
            if i != j and [j, i] not in array:
                array.append([i, j])
    return array


def update_stats(models, i, j, winner):
    r_i = 10**(models[i]['elo']/400)
    r_j = 10**(models[j]['elo']/400)
    E_i = r_i / (r_i + r_j)
    E_j = r_j / (r_i + r_j)
    if winner is None:
        models[i]['draws'] += 1
        models[j]['draws'] += 1
        models[i]['elo'] += 32 * (0.5 - E_i)
        models[j]['elo'] += 32 * (0.5 - E_j)
    elif winner == 0:
        models[i]['wins'] += 1
        models[j]['losses'] += 1
        models[i]['elo'] += 32 * (1 - E_i)
        models[j]['elo'] += 32 * (0 - E_j)
    elif winner == 1:
        models[j]['wins'] += 1
        models[i]['losses'] += 1
        models[i]['elo'] += 32 * (0 - E_i)
        models[j]['elo'] += 32 * (1 - E_j)

    return models


def round_robin(models, pre_moves, args, expansions=200, c=3):
    pairings = get_pairings(np.arange(len(models)))

    for i, j in pairings:
        model_1 = models[i]['model']
        model_2 = models[j]['model']

        pm_1 = models[i]['pm']
        pm_2 = models[j]['pm']

        name_1 = models[i]['name']
        name_2 = models[j]['name']

        elo_1 = int(models[i]['elo'])
        elo_2 = int(models[j]['elo'])

        with torch.no_grad():
            print(f'### {name_1} | {elo_1} vs {name_2} | {elo_2} ###')
            winner = play_game(model_1, model_2, expansions, c, pre_moves, pm_1=pm_1, pm_2=pm_2)
            models = update_stats(models, i, j, winner)

            elo_1 = int(models[i]['elo'])
            elo_2 = int(models[j]['elo'])

            print(f'### {name_2} | {elo_2} vs {name_1} | {elo_1} ###')
            winner = play_game(model_2, model_1, expansions, c, pre_moves, pm_1=pm_2, pm_2=pm_1)
            models = update_stats(models, j, i, winner)

    return models


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_id", type=str, default='base_models', help="Chose the experiment to conduct.")
    parser.add_argument("--num_games", type=int, default=25, help="Chose the number of games. In [1, 50]. Note: The total number of games depends on how many models play the tournament.")

    args = parser.parse_args()

    with open('./data/round_robin/eval_positions', 'rb') as pkl_file:
        pre_moves = pickle.load(pkl_file)

    pre_moves = pre_moves[:args.num_games]

    if args.experiment_id == 'base_models':
        models = BASE_MODELS
    if args.experiment_id == 'im_models':
        models = IM_MODELS
    if args.experiment_id == 'ppo_models':
        models = PPO_MODELS
    if args.experiment_id == 'SPM_base':
        models = SPM_BASE_MODELS
    if args.experiment_id == 'SPM_QVar':
        models = SPM_QVAR_MODELS
    if args.experiment_id == 'ConvPM_base':
        models = CONVPM_BASE_MODELS
    if args.experiment_id == 'ConvPM_QVar':
        models = CONVPM_QVAR_MODELS
    if args.experiment_id == 'ConvPM_MH':
        models = CONVPM_MH_MODELS
    if args.experiment_id == 'ConvPM_All':
        models = CONVPM_ALL_MODELS


    if args.experiment_id == 'base_models':
        for model in models:
            path = os.path.join(MODEL_DIR, model['path'])
            model['model'] = ConnectNet(6)
            model['model'].load_state_dict(torch.load(path)['state_dict'])
            model['model'].eval()
            if torch.cuda.is_available():
                model['model'].cuda()
            model['pm'] = None
    else:
        path = os.path.join(MODEL_DIR, 'base_model/a0_model_iter42.pth.tar')
        base = ConnectNet(6)
        base.load_state_dict(torch.load(path)['state_dict'])
        base.eval()
        if torch.cuda.is_available():
            base.cuda()

        for model in models:
            model['model'] = base
            if model['name'] == 'a0_model_iter_42':
                model['pm'] = None
            else:
                path = os.path.join(MODEL_DIR, model['path'])
                model['pm'] = pm_net.get_pm(model['pm_id'])
                model['pm'].load_state_dict(torch.load(path)['state_dict'])
                model['pm'].eval()
                if torch.cuda.is_available():
                    model['pm'].cuda()

    rtpt = RTPT(name_initials="AG", experiment_name='C4_PPO', max_iterations=len(pre_moves))

    rtpt.start()

    for i, moves in enumerate(pre_moves):
        print(f'######### Round {i+1}/{len(pre_moves)} #########')
        models = round_robin(models, moves, args, expansions=200, c=3)
        rtpt.step()

    if not os.path.exists('./data/round_robin/results/'):
        os.makedirs('./data/round_robin/results/')

    for model in models:
        model.pop('model', None)
        model.pop('pm', None)

    with open(f'./data/round_robin/results/{args.experiment_id}', 'wb') as output:
        pickle.dump(models, output)

    for model in models:
        name = model['name']
        wins = model['wins']
        losses = model['losses']
        elo = model['elo']
        print(f'### {name} ### \n Wins: {wins} \n Losses: {losses} \n Elo: {elo}')
