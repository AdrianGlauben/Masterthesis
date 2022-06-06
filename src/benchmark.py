import numpy as np
from MCTS_c4 import UCT_search, do_decode_n_move_pieces, get_policy
from alpha_net_c4 import ConnectNet
from pm_net import SimplePM
from connect_board import board
import torch
import pickle

def play_game(model_1, model_2, expansions_per_move=200, c=1.5, pre_moves=None, pm_1=None, pm_2=None):
    current_board = board()
    t = 0.1
    if pre_moves != None:
        for move in pre_moves:
            current_board.drop_piece(move)
    checkmate = False
    winner = None

    while checkmate == False and current_board.actions() != []:
        if current_board.player == 0:
            root = UCT_search(current_board,expansions_per_move,model_1,t,c=c,planning_model=pm_1)
            policy = get_policy(root)
        elif current_board.player == 1:
            root = UCT_search(current_board,expansions_per_move,model_2,t,c=c,planning_model=pm_2)
            policy = get_policy(root)
        current_board = do_decode_n_move_pieces(current_board,np.argmax(policy)) # decode move and move piece(s)
        if current_board.check_winner() == True: # someone wins
            if current_board.player == 0: # Model 2 wins
                winner = 1
            elif current_board.player == 1: # Model 1 wins
                winner = 0
            checkmate = True
    print(current_board.current_board)
    return winner


def evaluate(model_1, model_2, expansions_per_move=200, c=1.5, use_pre_moves=True, pm_1=None, pm_2=None):
    stats_1 = [0, 0, 0]
    stats_2 = [0, 0, 0]

    if use_pre_moves:
        with open('./data/eval_positions', 'rb') as pkl_file:
            pre_moves = pickle.load(pkl_file)

        for i, moves in enumerate(pre_moves):
            winner = play_game(model_1, model_2, expansions_per_move, c, moves, pm_1=pm_1, pm_2=pm_2)
            if winner is not None:
                stats_1[winner] += 1
            else:
                stats_1[2] += 1
            print(f'M1 / M2 ### Games played: {i+1} ### Current stats: {stats_1}')

        for i, moves in enumerate(pre_moves):
            winner = play_game(model_2, model_1, expansions_per_move, moves, c, pm_1=pm_2, pm_2=pm_1)
            if winner is not None:
                stats_2[abs(winner-1)] += 1
            else:
                stats_2[2] += 1
            print(f'M2 / M1 ### Games played: {i+1} ### Current Stats: {stats_2}')

    else:
        for i in range(50):
            winner = play_game(model_1, model_2, expansions_per_move, c, pm_1=pm_1, pm_2=pm_2)
            if winner is not None:
                stats_1[winner] += 1
            else:
                stats_1[2] += 1
            print(f'M1 / M2 ### Games played: {i+1} ### Current stats: {stats_1}')

        for i in range(50):
            winner = play_game(model_2, model_1, expansions_per_move, c, pm_1=pm_2, pm_2=pm_1)
            if winner is not None:
                stats_2[abs(winner-1)] += 1
            else:
                stats_2[2] += 1
            print(f'M2 / M1 ### Games played: {i+1} ### Current Stats: {stats_2}')

    return stats_1, stats_2


MODEL_1_PATH = './training_history/run5/cc4_current_net__iter15.pth.tar'
MODEL_2_PATH = './training_history/run2/cc4_current_net__iter74.pth.tar'

PM_MODEL_1_PATH = './data/pm_data/SimplePM_run2_59'

model_1 = ConnectNet(12)
model_2 = ConnectNet(12)

pm_model_1 = SimplePM()

checkpoint = torch.load(MODEL_1_PATH)
model_1.load_state_dict(checkpoint['state_dict'])

checkpoint = torch.load(MODEL_2_PATH)
model_2.load_state_dict(checkpoint['state_dict'])

pm_model_1.load_state_dict(torch.load(PM_MODEL_1_PATH))

if torch.cuda.is_available():
    model_1.cuda()
    model_2.cuda()
    pm_model_1.cuda()

print(evaluate(model_1, model_2, 200, 1.5, True, pm_1=None, pm_2=None))
