import numpy as np
from MCTS_c4 import UCT_search, do_decode_n_move_pieces, get_policy
from alpha_net_c4 import ConnectNet
from pm_net import SimplePM, SimplePM_QVar, ConvPM, ConvPM_QVar, ConvPM_MH, ConvPM_All
from connect_board import board
from encoder_decoder_c4 import encode_board
import torch
import pickle
import time

def play_game(model_1, model_2, expansions_per_move=200, c=1.5, pre_moves=None, pm_1=None, pm_2=None):
    current_board = board()
    t = 0.1
    move_history = []
    mh = []
    if pre_moves != None:
        for move in pre_moves:
            current_board.drop_piece(move)
            move_history.append(encode_board(current_board).transpose(2, 0, 1))
    checkmate = False
    winner = None

    times = []

    while checkmate == False and current_board.actions() != []:
        if current_board.player == 0:
            t1 = time.time()
            root = UCT_search(current_board,expansions_per_move,model_1,t,c=c,planning_model=pm_1, move_history=move_history)
            t2 = time.time()
            policy = get_policy(root)
        elif current_board.player == 1:
            t1 = time.time()
            root = UCT_search(current_board,expansions_per_move,model_2,t,c=c,planning_model=pm_2, move_history=move_history)
            t2 = time.time()
            policy = get_policy(root)
        current_board = do_decode_n_move_pieces(current_board,np.argmax(policy)) # decode move and move piece(s)
        mh.append(np.argmax(policy))
        move_history.append(encode_board(current_board).transpose(2, 0, 1))
        times.append(t2 - t1)
        if current_board.check_winner() == True: # someone wins
            if current_board.player == 0: # Model 2 wins
                winner = 1
            elif current_board.player == 1: # Model 1 wins
                winner = 0
            checkmate = True
    print(current_board.current_board)
    print(mh)
    print(winner)
    print(len(mh))
    return winner, times


def evaluate(model_1, model_2, expansions_per_move=200, c=1.5, use_pre_moves=True, pm_1=None, pm_2=None):
    # [Wins Model 1, Wins Model 2, Draws] Both outputs in that order!!!!
    stats_1 = [0, 0, 0]
    stats_2 = [0, 0, 0]
    game_times = []

    if use_pre_moves:
        with open('./data/eval_positions', 'rb') as pkl_file:
            pre_moves = pickle.load(pkl_file)
            pre_moves = [[]]

        for i, moves in enumerate(pre_moves):
            winner, times = play_game(model_1, model_2, expansions_per_move, c, moves, pm_1=pm_1, pm_2=pm_2)
            game_times.append(times)
            if winner is not None:
                stats_1[winner] += 1
            else:
                stats_1[2] += 1
            print(f'M1 / M2 ### Games played: {i+1} ### Current stats: {stats_1}')

        # for i, moves in enumerate(pre_moves):
        #     winner, times = play_game(model_2, model_1, expansions_per_move, c, moves, pm_1=pm_2, pm_2=pm_1)
        #     game_times.append(times)
        #     if winner is not None:
        #         stats_2[abs(winner-1)] += 1
        #     else:
        #         stats_2[2] += 1
        #     print(f'M2 / M1 ### Games played: {i+1} ### Current Stats: {stats_2}')

    else:
        for i in range(50):
            winner, times = play_game(model_1, model_2, expansions_per_move, c, pm_1=pm_1, pm_2=pm_2)
            game_times.append(times)
            if winner is not None:
                stats_1[winner] += 1
            else:
                stats_1[2] += 1
            print(f'M1 / M2 ### Games played: {i+1} ### Current stats: {stats_1}')

        for i in range(50):
            winner, times = play_game(model_2, model_1, expansions_per_move, c, pm_1=pm_2, pm_2=pm_1)
            game_times.append(times)
            if winner is not None:
                stats_2[abs(winner-1)] += 1
            else:
                stats_2[2] += 1
            print(f'M2 / M1 ### Games played: {i+1} ### Current Stats: {stats_2}')

    print(f'Model 1 win rate: {(stats_1[0] + stats_2[0])/100}')
    print(f'Model 2 win rate: {(stats_1[1] + stats_2[1])/100}')

    return stats_1, stats_2, game_times


if __name__ == "__main__":
    MODEL_1_PATH = './training_history/run6/cc4_current_net__iter42.pth.tar'
    MODEL_2_PATH = './training_history/run6/cc4_current_net__iter42.pth.tar'

    PM_MODEL_1_PATH = './data/round_robin/models/planning_model/ConvPM_All/ConvPM_All_iter_30.pth.tar'
    PM_MODEL_2_PATH = './data/round_robin/models/planning_model/SPM_base/SPM_base_iter_30.pth.tar'

    model_1 = ConnectNet(6)
    #model_2 = ConnectNet(6)

    pm_model_1 = ConvPM_All()
    #pm_model_2 = SimplePM()

    checkpoint = torch.load(MODEL_1_PATH)
    model_1.load_state_dict(checkpoint['state_dict'])

    #checkpoint = torch.load(MODEL_2_PATH)
    #model_2.load_state_dict(checkpoint['state_dict'])

    pm_model_1.load_state_dict(torch.load(PM_MODEL_1_PATH)['state_dict'])
    #pm_model_2.load_state_dict(torch.load(PM_MODEL_2_PATH)['state_dict'])

    if torch.cuda.is_available():
        model_1.cuda()
        #model_2.cuda()
        pm_model_1.cuda()
        #pm_model_2.cuda()

    model_1.eval()
    #model_2.eval()
    pm_model_1.eval()
    #pm_model_2.eval()

    stats_1, stats_2, times = evaluate(model_1, model_1, 200, 3, True, pm_1=pm_model_1, pm_2=pm_model_1)
    # print(stats_1)
    # print(stats_2)
    # times = np.concatenate(times)
    # mean = np.mean(times)
    # std = np.std(times)
    # nps = 200 / mean
    # nps_plus_minus = 200 / (mean - std) - nps
    #
    # print(f'NPS: {nps} +/- {nps_plus_minus}')
