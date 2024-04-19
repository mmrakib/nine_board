#!/usr/bin/python3

# File:   agent.py
# Course: comp3411/9814 - Artificial Intelligence
# Task:   assignment 3 - nine-board tic-tac-toe
# Author: mohammad mayaz rakib
# zID:    z5361151

# Usage:
# ./agent.py -p <PORT>

# Summary:
#
# I employed a heuristic greedy search algorithm to complete this assignment. Essentially, this program works by utilising a minimax algorithm with alpha-beta pruning on each subboard of the entire 9x9 board. Thus, it is a minimax algorithm designed to solve each regular 3x3 tic-tac-toe problem as optimally as possible. Since the objective is to finish one of the subboards first, it makes sense to make optimal moves per subboard. However, this doesn't consider the world model, which is the entire 9x9 board and its constraints, when it performs any move planning. This makes a minimax-only algorithm a fundamentally greedy algorithm.
# For this reason, I also implemented a heuristic to work with the greedy algorithm. This heuristic essentially performs a static evaluation on each subboard of the overall board, considering nothing but the positions of the both types of marker. It then returns a score between -500 and 500, where 500 is a winning board, and -500 is a losing board. Incomplete boards are evaluated based off of their position, such as whether or not they are close to the centre, using the matrix,
# 1 0 1
# 0 2 0
# 1 0 1
# giving preference to the centre spot and corner spots. It is also uses the frequency of two in a rows, that are 'near completion', as well as two in a rows with a gap in between, both of which are represented in a pre-computed manner using the 'pairs' and 'edge pairs' matrices. Utilising this heuristic gives the greedy algorithm a notion of 'closeness', and thus allows it to play optimally (i.e. run minimax) on the 'most preferred' 3x3 subboard, giving it a slight edge over a purely greedy approach.

import socket
import argparse
import inspect
import copy

import numpy as np # type: ignore

#
# Debug
#

# Enable/disable debugging mode
DEBUG = False

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
def generate_random_move(n):
    pos = np.random.randint(0, 9)
    while not is_cell_empty(n, pos):
        pos = np.random.randint(0, 9)

    return pos

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

    if (len(moves) != 0):        
        if compute_subboard_full(subboard):
            log('ERROR: Subboard is full and cannot place!')

        return generate_random_move(n)

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

    for pos, _ in enumerate(board):
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

    if is_cell_empty(n, pos):
        board[n][pos] = turn
        board_index = pos
        move_index += 1
    else:
        log('ERROR: Placed at non-empty cell')
        log(f'n = {n}, pos = {pos}')
        return generate_random_move(n)

def decide_move(n, turn):
    return run_minimax(n, turn)

#
# Server
#
def parse(string):
    global board_index

    if '(' in string:
        command, args = string.split('(')
        args = args.split(')')[0]
        args = args.split(',')
    else:
        command, args = string, []

    if command == "second_move":
        n = int(args[0])
        pos = int(args[1])

        n -= 1
        pos -= 1

        make_move(n, pos, OPPONENT)
        move_pos = decide_move(n, pos)
        log(f'move_pos: {move_pos}')
        return move_pos + 1
    
    if command == "third_move":
        n = int(args[0])
        pos = int(args[1])
        pos2 = int(args[2])

        n -= 1
        pos -= 1
        pos2 -= 1

        make_move(n, pos, PLAYER)
        make_move(board_index, pos2, OPPONENT)
        move_pos = decide_move(board_index, PLAYER)
        log(f'move_pos: {move_pos}')
        return move_pos + 1
    
    if command == "next_move":
        pos = int(args[0])

        pos -= 1

        make_move(board_index, pos, OPPONENT)
        move_pos = decide_move(board_index, PLAYER)
        log(f'move_pos: {move_pos}')
        return move_pos + 1
    
    if command == "win":
        print("We won the game.")
        return -1
    
    if command == "loss":
        print("We lost the game")
        return -1
    
    return 0

def listen(args):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = int(args.port)

    s.connect(('localhost', port))

    while True:
        text = s.recv(1024).decode()
        if not text:
            continue

        for line in text.split('\n'):
            response = parse(line)

            if response == -1:
                s.close()
                return
            elif response > 0:
                s.sendall((str(response) + '\n').encode())

def start():
    parser = argparse.ArgumentParser(prog = 'agent.py', \
                                     description = 'comp3411 - assignment 3 - nine-board tic-tac-toe', \
                                     epilog = 'author: mohammad mayaz rakib (z5361151)')
    parser.add_argument('-p', '--port', required = True)
    args = parser.parse_args()

    listen(args)

def main():
    start()

if __name__ == '__main__':
    main()
