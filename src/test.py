import pickle
from evaluator_c4 import load_pickle
from connect_board import board
import numpy as np


array = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
array = array[-int(len(array)*0.3):]
print(array)
exit()

positions = []
for i in range(5):
    current_board = board()
    for i in range(3):
        moves = current_board.actions()
        move = np.random.choice(moves)
        current_board.drop_piece(move)
        print(current_board.current_board)
        positions.append(current_board.current_board)
    print(positions)
    exit()

print(positions)
print(len(positions))
