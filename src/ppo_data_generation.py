import numpy as np
from MCTS_c4 import UCT_search, do_decode_n_move_pieces, get_policy
from alpha_net_c4 import ConnectNet
from connect_board import board
from encoder_decoder_c4 import encode_board
import pm_net
import torch
import pickle
import os
import time


def play_game(a0_model, pm, args):
    current_board = board()
    checkmate = False
    winner = None
    move_count = 0
    dataset = {'expansions': args.expansions_per_move, 'data': []}
    move_history = []
    move_history_input = []

    while checkmate == False and current_board.actions() != []:
        if move_count < 11:
            t = args.temperature_MCTS
        else:
            t = 0.1
        move_count += 1
        root, data = UCT_search(current_board, args.expansions_per_move, a0_model, t, generate_ppo_data=True, planning_model=pm, move_history=move_history_input)
        for i in range(len(data)):
            data[i][4] = move_history + data[i][4]
        dataset['data'].extend(data)

        policy = get_policy(root, t)
        move = np.random.choice(np.array([0,1,2,3,4,5,6]), p = policy)
        move_history.append(move)

        current_board = do_decode_n_move_pieces(current_board, move) # decode move and move piece(s)
        move_history_input.append(encode_board(current_board).transpose(2, 0, 1))

        if current_board.check_winner() == True: # someone wins
            if current_board.player == 0: # Player 2 wins
                winner = 1
            elif current_board.player == 1: # Player 1 wins
                winner = 0
            checkmate = True
    print(current_board.current_board)
    for i in range(len(dataset['data'])):
        dataset['data'][i].append(winner)
    return dataset, winner


def generate_data(args, iteration):
    base_path = './data/ppo_data/'
    pm_net_name = f'pm_model_data/{args.pm_id}/{args.pm_id}_iter_{iteration}.pth.tar'
    pm_path = os.path.join(base_path, pm_net_name)
    a0_model_path = os.path.join(base_path, 'a0_model.pth.tar')

    a0_model = ConnectNet(12)
    a0_model.load_state_dict(torch.load(a0_model_path)['state_dict'])

    pm = pm_net.get_pm(args.pm_id)
    pm.load_state_dict(torch.load(pm_path)['state_dict'])

    if torch.cuda.is_available():
        a0_model.cuda()
        pm.cuda()

    pm.eval()
    a0_model.eval()

    data_path = os.path.join(base_path, f'game_data/iter_{iteration}/')

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    for i in range(args.num_games_per_iteration):
        print(f'### Iteration: {iteration+1}/{args.total_iterations} ### Game: {i+1}/{args.num_games_per_iteration} ###')
        t0 = time.time()
        with torch.no_grad():
            dataset, winner = play_game(a0_model, pm, args)
        t1 = time.time()
        print(f'--- Winner: {winner} // Time: {t1-t0:.2f}s ---')

        game_data_path = os.path.join(data_path, f'game_{i}')

        with open(game_data_path, 'wb') as output:
            pickle.dump(dataset, output)
