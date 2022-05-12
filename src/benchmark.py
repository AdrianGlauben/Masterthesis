import numpy as np
from MCTS_c4 import UCT_search, do_decode_n_move_pieces, get_policy
from alpha_net_c4 import ConnectNet
from connect_board import board
import torch

def play_game(model_1, model_2, expansions_per_move=200):
    current_board = board()
    checkmate = False
    winner = None
    t = 0.1
    while checkmate == False and current_board.actions() != []:
        if current_board.player == 0:
            root = UCT_search(current_board,expansions_per_move,model_1,t)
            policy = get_policy(root, t)
        elif current_board.player == 1:
            root = UCT_search(current_board,expansions_per_move,model_2,t)
            policy = get_policy(root, t)
        current_board = do_decode_n_move_pieces(current_board,\
                                                np.random.choice(np.array([0,1,2,3,4,5,6]), \
                                                                 p = policy)) # decode move and move piece(s)
        if current_board.check_winner() == True: # someone wins
            if current_board.player == 0: # Model 2 wins
                winner = 1
            elif current_board.player == 1: # Model 1 wins
                winner = 0
            checkmate = True
    print(current_board.current_board)
    return winner


def evaluate(model_1, model_2, expansions_per_move=200):
    stats_1 = [0, 0, 0]
    stats_2 = [0, 0, 0]

    for i in range(50):
        winner = play_game(model_1, model_2, expansions_per_move)
        if winner is not None:
            stats_1[winner] += 1
        else:
            stats_1[2] += 1
        print(f'M1 / M2 ### Games played: {i+1} ### Current stats: {stats_1}')

    for i in range(50):
        winner = play_game(model_2, model_1, expansions_per_move)
        if winner is not None:
            stats_2[abs(winner-1)] += 1
        else:
            stats_2[2] += 1
        print(f'M2 / M1 ### Games played: {i+1} ### Current Stats: {stats_2}')

    return stats_1, stats_2


MODEL_1_PATH = './training_history/5i_100g_200e_8r/model_data/cc4_current_net__iter4.pth.tar'
MODEL_2_PATH = './training_history/5i_100g_200e_8r/model_data/cc4_current_net__iter5.pth.tar'

model_1 = ConnectNet()
model_2 = ConnectNet()

checkpoint = torch.load(MODEL_1_PATH)
model_1.load_state_dict(checkpoint['state_dict'])

checkpoint = torch.load(MODEL_2_PATH)
model_2.load_state_dict(checkpoint['state_dict'])

if torch.cuda.is_available():
    model_1.cuda()
    model_2.cuda()

print(evaluate(model_1, model_2, 200))