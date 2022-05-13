import pickle
from evaluator_c4 import load_pickle
from connect_board import board
import numpy as np
import os

positions = []
while len(positions) < 50:
    current_board = board()
    length = np.random.choice([1, 2, 3, 4])
    moves = []
    for i in range(length):
        legal_moves = current_board.actions()
        move = np.random.choice(legal_moves)
        current_board.drop_piece(move)
        moves.append(move)
    if moves not in positions:
        positions.append(moves)

print(len(positions))

# completeName = os.path.join("./data/",\
#                             'eval_positions')
# with open(completeName, 'wb') as output:
#     pickle.dump(positions, output)
