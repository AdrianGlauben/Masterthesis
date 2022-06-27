import numpy as np
from MCTS_c4 import UCT_search, do_decode_n_move_pieces, get_policy
from alpha_net_c4 import ConnectNet
from connect_board import board
from argparse import ArgumentParser
from rtpt import RTPT
import torch
import pickle
import os
from ppo_data_generation import generate_data
from ppo_train import train_pm
from pm_net import ConvPM_All


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--total_iterations", type=int, default=3, help="Total Number of iterations.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for PMs.")
    parser.add_argument("--num_games_per_iteration", type=int, default=1, help="The number of games per iteration.")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size used in training.")
    parser.add_argument("--iteration", type=int, default=0, help="The iteration from which to resume training.")
    parser.add_argument("--temperature_MCTS", type=float, default=1.1, help="Temperature for first 10 moves of each MCTS self-play")
    parser.add_argument('--expansions_per_move', type=int, default=200, help='Number of expansions per MCTS move')
    parser.add_argument('--pm_id', type=str, default='SPM_base', help='Planning Model identifier. From: [SPM_base, SPM_QVar, ConvPM_base, ConvPM_QVar, ConvPM_MH, ConvPM_All]')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='Epsilon for Clipped PPO objective.')

    args = parser.parse_args()

    rtpt = RTPT(name_initials="AG", experiment_name='C4_PPO', max_iterations=args.total_iterations)

    rtpt.start()

    for i in range(args.iteration, args.total_iterations):
        generate_data(args, i)
        train_pm(args, i)
        rtpt.step()
