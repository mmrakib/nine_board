#!/usr/bin/python3

# File:   agent.py
# Course: comp3411/9814 - Artificial Intelligence
# Task:   assignment 3 - nine-board tic-tac-toe
# Author: mohammad mayaz rakib
# zID:    z5361151

# Usage:
# ./agent.py -p <PORT>

import inspect
import copy

import numpy as np # type: ignore

#
# Debug
#

# Enable/disable debugging mode
DEBUG = True

# Log message if debugging mode enabled
def log(message, end = '\n'):
    current_frame = inspect.currentframe()
    caller_frame = inspect.getouterframes(current_frame, 2)

    func = caller_frame[1].function
    lineno = caller_frame[1].lineno

    if DEBUG:
        print(f'[line {lineno}] [{func}()] ' + message, end = end)

#
# Constants
#

# Cell values
EMPTY = 0
PLAYER = 1
OPPONENT = 2

# Game state values
ONGOING = 0
PLAYER_WIN = 1
OPPONENT_WIN = 2
DRAW = 3

# Score values
PLAYER_WIN_SCORE = 500
OPPONENT_WIN_SCORE = -500
PLAYER_TWO_ROW_SCORE = 5
OPPONENT_TWO_ROW_SCORE = -5
PLAYER_EDGE_PAIR_SCORE = 3
OPPONENT_EDGE_PAIR_SCORE = -3
PLAYER_ONE_CELL_SCORE = 2
OPPONENT_ONE_CELL_SCORE = -2
DRAW_SCORE = 0

# +- infinity definitions
POS_INF = float('inf')
NEG_INF = float('-inf')

# Maximum search tree depth
MAX_DEPTH = 1000

#
# Pre-computed values
#
triples = [(0, 1, 2), (3, 4, 5), (6, 7, 8), \
           (0, 3, 6), (1, 4, 7), (2, 5, 8), \
           (0, 4, 8), (2, 4, 6)]

pairs = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 3), \
         (1, 4), (1, 5), (2, 4), (2, 5), (3, 4), \
         (3, 6), (3, 7), (4, 5), (4, 6), (4, 7), \
         (4, 8), (5, 7), (5, 8), (6, 7), (7, 8)]

edge_pairs = [(0, 2), (3, 5), (6, 8), \
              (0, 6), (1, 7), (2, 8), \
              (0, 8), (2, 6)]

positional_score = [1, 0, 1, 0, 2, 0, 1, 0, 1]

#
# Game state
#
board = np.zeros((9, 9), dtype='int32')
board_index = 0
move_index = 0

#
# Game static evaluation
#
def is_cell_empty(n, pos):
    global board

    if board[n][pos] == EMPTY:
        return True
    else:
        return False

def compute_subboard_full(subboard):
    if np.all(subboard):
        return True
    else:
        return False
    
def compute_subboard_state(subboard):
    for tri in triples:
        i, j, k = tri

        if subboard[i] == subboard[j] and \
            subboard[j] == subboard[k] and \
            subboard[i] != EMPTY:

            if subboard[i] == PLAYER:
                return PLAYER_WIN
            elif subboard[i] == OPPONENT:
                return OPPONENT_WIN
    
    if compute_subboard_full(subboard):
        return DRAW
    
    return ONGOING

def compute_subboard_value(subboard):
    global positional_score
    values = positional_score

    for pair in pairs:
        i, j = pair

        if subboard[i] == subboard[j] and \
           subboard[i] != EMPTY:
            if subboard[i] == PLAYER:
                values[i] += PLAYER_TWO_ROW_SCORE
                values[j] += PLAYER_TWO_ROW_SCORE
            if subboard[j] == OPPONENT:
                values[i] += OPPONENT_TWO_ROW_SCORE
                values[i] += OPPONENT_TWO_ROW_SCORE

    for pair in edge_pairs:
        i, j = pair

        if subboard[i] == subboard[j] and \
           subboard[i] != EMPTY:
            if subboard[i] == PLAYER:
                values[i] += PLAYER_EDGE_PAIR_SCORE
                values[j] += PLAYER_EDGE_PAIR_SCORE
            if subboard[j] == OPPONENT:
                values[i] += OPPONENT_EDGE_PAIR_SCORE
                values[j] == OPPONENT_EDGE_PAIR_SCORE

    for pos, cell in enumerate(subboard):
        if cell != EMPTY:
            if cell == PLAYER:
                values[pos] += PLAYER_ONE_CELL_SCORE
            if cell == OPPONENT:
                values[pos] += OPPONENT_ONE_CELL_SCORE

    return sum(values)

def compute_subboard_score(subboard):
    state = compute_subboard_state(subboard)

    if state > 0:
        if state == PLAYER_WIN:
            return PLAYER_WIN_SCORE
        elif state == OPPONENT_WIN:
            return OPPONENT_WIN_SCORE
        else:
            return DRAW_SCORE
    else:
        return compute_subboard_value(subboard)
    
def compute_board_scores(board):
    values = np.zeros(9, dtype='int32')

    for pos, subboard in enumerate(board):
        value = compute_subboard_score(subboard)
        values[pos] = value

    return values

def compute_preference(board):
    values = compute_board_scores(board)

    max_value = NEG_INF
    max_index = 0

    for pos, value in enumerate(values):
        if value > max_value:
            max_value = value
            max_index = pos

    return max_index
    
#
# Game strategy
#
def generate_moves(subboard, turn):
    moves = []

    for pos, cell in enumerate(subboard):
        if cell == EMPTY:
            new_subboard = copy.copy(subboard)
            new_subboard[pos] = turn
            moves.append((pos, new_subboard))

    return moves

def minimax(subboard, depth, alpha, beta, turn):
    state = compute_subboard_state(subboard)

    if state > 0 or depth == 0:
        return compute_subboard_score(subboard)
    
    if turn == PLAYER:
        max_eval = NEG_INF
        moves = generate_moves(subboard, turn)

        for move in moves:
            eval = minimax(move[1], depth - 1, alpha, beta, OPPONENT)
            max_eval = max(eval, max_eval)

            alpha = max(eval, alpha)
            if beta <= alpha:
                break

        return max_eval
    
    if turn == OPPONENT:
        min_eval = POS_INF
        moves = generate_moves(subboard, turn)

        for move in moves:
            eval = minimax(move[1], depth - 1, alpha, beta, PLAYER)
            min_eval = min(eval, min_eval)

            beta = min(eval, beta)
            if beta <= alpha:
                break

        return min_eval

def run_minimax(n, turn):
    global board
    subboard = board[n]

    moves = generate_moves(subboard, turn)
    if (len(moves) == 0):
        n = np.random.randint(0, 9)
        pos = np.random.randint(0, 9)

        while board[n, pos] != EMPTY:
            n = np.random.randint(0, 9)
            pos = np.random.randint(0, 9)

        return pos

    best_eval = NEG_INF if turn == PLAYER else POS_INF
    best_move_index = 0

    for move_index, move in enumerate(moves):
        eval = minimax(move[1], MAX_DEPTH, NEG_INF, POS_INF, OPPONENT if turn == PLAYER else PLAYER)

        if turn == PLAYER:
            if eval > best_eval:
                best_eval = eval
                best_move_index = move_index

        if turn == OPPONENT:
            if eval < best_eval:
                best_eval = eval
                best_move_index = move_index

    return moves[best_move_index][0]

def get_opponent_optimal_moves(board):
    moves = []

    for pos, subboard in enumerate(board):
        move = run_minimax(pos, OPPONENT)
        moves.append(move)

    return moves

#
# Actions
#

def make_move(n, pos, turn):
    global board
    global board_index
    global move_index

    board[n][pos] = turn
    board_index - pos
    move_index += 1

def decide_move(n, turn):
    global board

    preference = compute_preference(board)
    opponent_moves = get_opponent_optimal_moves(board)

    for pos, move in enumerate(opponent_moves):
        if preference == move and is_cell_empty(n, pos):
            return pos
        
    return run_minimax(n, turn)

def main():
    global board
    global board_index
    global move_index
    
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            val = np.random.randint(0, 3)
            if val > 0:
                move_index += 1
            board[i, j] = val

    board_index = np.random.randint(0, 9)

    print(f'board:\n{board}')
    print(f'board_index: {board_index}')
    print(f'move_index: {move_index}\n')

    pos = decide_move(board_index, PLAYER)
    print(f'pos: {pos}\n')

    make_move(board_index, pos, PLAYER)

    print(f'board after move: \n{board}')

if __name__ == '__main__':
    main()
