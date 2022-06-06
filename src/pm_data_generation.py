import numpy as np
from MCTS_c4 import UCT_search, do_decode_n_move_pieces, get_policy
from alpha_net_c4 import ConnectNet
from connect_board import board
import torch
import pickle
import os


def play_game(model, expansions_per_move=200, c=1.5):
    current_board = board()
    checkmate = False
    winner = None
    move_count = 0
    dataset = {'expansions': expansions_per_move, 'cpuct': c, 'data': []}
    move_history = []

    while checkmate == False and current_board.actions() != []:
        if move_count < 11:
            t = 1.1
        else:
            t = 0.1
        move_count += 1

        root, data = UCT_search(current_board, expansions_per_move, model, t, c, generate_data=True)
        for i in range(len(data)):
            data[i][4] = move_history + data[i][4]
        dataset['data'].extend(data)

        policy = get_policy(root, t)
        move = np.random.choice(np.array([0,1,2,3,4,5,6]), p = policy)
        move_history.append(move)

        current_board = do_decode_n_move_pieces(current_board, move) # decode move and move piece(s)
        if current_board.check_winner() == True: # someone wins
            if current_board.player == 0: # Model 2 wins
                winner = 1
            elif current_board.player == 1: # Model 1 wins
                winner = 0
            checkmate = True
    print(current_board.current_board)
    return winner, dataset


MODEL_PATH = './training_history/run2/cc4_current_net__iter74.pth.tar'
NUM_GAMES = 3

model = ConnectNet(12)

checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['state_dict'])

if torch.cuda.is_available():
    model.cuda()

if not os.path.isdir("data/pm_data/game_data"):
    os.mkdir("data/pm_data/game_data")

for i in range(NUM_GAMES):
    print(f'Game No. {i+1}')
    winner, dataset = play_game(model, expansions_per_move=200)

    completeName = os.path.join("./data/pm_data/game_data/",\
                                f'game_{i}')
    with open(completeName, 'wb') as output:
        pickle.dump(dataset, output)
