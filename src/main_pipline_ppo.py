import numpy as np
from MCTS_c4 import UCT_search, do_decode_n_move_pieces, get_policy
from alpha_net_c4 import ConnectNet
from connect_board import board
import torch
import pickle
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--total_iterations", type=int, default=10, help="Total Number of iterations.")
    parser.add_argument("--lr", type=int, default=0.0001, help="Learning rate for PMs.")
    parser.add_argument("--num_games", type=int, default=10, help="Total Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=32, help="Total Number of iterations.")
    parser.add_argument("--total_iterations", type=int, default=0, help="Total Number of iterations.")
