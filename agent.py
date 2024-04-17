#!/usr/bin/python3

# File:   agent.py
# Course: comp3411/9814 - Artificial Intelligence
# Task:   assignment 3 - nine-board tic-tac-toe
# Author: mohammad mayaz rakib
# zID:    z5361151

# Usage:
# ./agent.py -p <PORT>

import socket
import sys
import copy
import random

import numpy as np # type: ignore

#
# Constants
#
POS_INF = float('inf')
NEG_INF = float('-inf')

EMPTY = 0
PLAYER = 1
OPPONENT = 2

ONGOING = 0
PLAYER_WIN = 1
OPPONENT_WIN = 2
DRAW = 3

PLAYER_WIN_SCORE = 10
OPPONENT_WIN_SCORE = -10
DRAW_SCORE = 0

#
# Pre-computed values
#

# 0 1 2
# 3 4 5
# 6 7 8

markers = ['.', 'X', 'O'] # Corresponds to cell values
triplets = [(0, 1, 2), (3, 4, 5), (6, 7, 8), \
            (0, 3, 6), (1, 4, 7), (2, 5, 8), \
            (0, 4, 8), (2, 4, 6)]

#
# Game state
#
board = np.zeroes((9, 9), dtype='int8')
board_index = 0

def get_subboard(n):
    return board[n]

#
# Game static evaluation
#
def compute_subboard_full(subboard):
    for cell in subboard:
        if cell == EMPTY:
            return False
    return True

def compute_subboard_winner(subboard):
    for tri in triplets:
        i, j, k = tri

        if subboard[i] == subboard[j] and \
            subboard[j] == subboard[k] and \
            subboard[k] == subboard[i] and \
            subboard[i] != EMPTY:
            
            if subboard[i] == PLAYER:
                return PLAYER_WIN
            elif subboard[i] == OPPONENT:
                return OPPONENT_WIN
        
    if compute_subboard_full(subboard):
        return DRAW
    
    return ONGOING

def compute_subboard_score(subboard, winner = -1):
    # Skip winner computation if already computed
    if (winner != -1):
        winner = compute_subboard_winner(subboard)

    if winner > 0:
        if winner == PLAYER_WIN:
            return PLAYER_WIN_SCORE
        elif winner == OPPONENT_WIN:
            return OPPONENT_WIN_SCORE
        else:
            return DRAW_SCORE
    else:
        # TODO: Add static evaluation for incomplete game states; random for now
        return random.randint(OPPONENT_WIN_SCORE + 1, PLAYER_WIN_SCORE - 1)
    
#
# Game strategy
#
def generate_moves(board, n, turn):
    subboard = get_subboard(n)
    moves = []

    for pos, cell in enumerate(subboard):
        if cell == EMPTY:
            newboard = copy.deepcopy(board)
            newboard[n][pos] = turn
            moves.append( (pos, newboard) ) # move = (position played, copy of board w/ position played)
    
    return moves

def minimax(board, n, depth, alpha, beta, turn):
    subboard = get_subboard(board)
    winner = compute_subboard_winner(subboard)

    if winner > 0 or depth == 0:
        return compute_subboard_score(subboard, winner)
    
    if turn == PLAYER:
        max_eval = NEG_INF
        moves = generate_moves(board, n, turn)

        for move in moves:
            eval = minimax(move[1], move[0], depth - 1, alpha, beta, OPPONENT)
            max_eval = max(eval, max_eval)

            alpha = max(eval, alpha)
            if beta <= alpha:
                break

        return max_eval
    
    if turn == OPPONENT:
        min_eval = POS_INF
        moves = generate_moves(board, n, turn)

        for move in moves:
            eval = minimax(move[1], move[0], depth - 1, alpha, beta, PLAYER)
            min_eval = min(eval, min_eval)

            beta = min(eval, beta)
            if beta <= alpha:
                break

        return min_eval

#
# Actions
#
def decide_move(board, n, turn):
    moves = generate_moves(board, n, turn)
    best_eval = NEG_INF
    best_move_index = 0

    for move_index, move in enumerate(moves):
        eval = minimax(move[1], move[0], 5, NEG_INF, POS_INF, OPPONENT if turn == PLAYER else PLAYER)
        
        if eval > best_eval:
            best_eval = eval
            best_move_index = move_index

    return moves[best_move_index][0]

def make_move(n, pos, turn):
    global board
    global board_index

    board[n][pos] = turn
    board_index = pos

def player_move(n, pos = -1):
    # Skip if position already decided
    if (pos == -1):
        pos = decide_move(board, n, PLAYER)
    make_move(n, pos, PLAYER)
    return pos
    
def opponent_move(n, pos):
    make_move(n, pos, OPPONENT)

#
# Server
#
def parse(string):
    global board_index
    
    if "(" in string:
        command, args = string.split("(")
        args = args.split(")")[0]
        args = args.split(",")
    else:
        command, args = string, []

    if command == "second_move":
        n = int(args[0])
        pos = int(args[1])

        opponent_move(n, pos)
        return player_move(n)
    
    elif command == "third_move":
        n = int(args[0])
        pos = int(args[1])
        pos2 = int(args[2])

        player_move(n, pos)
        opponent_move(board_index, pos2)
        return player_move(n)
    
    elif command == "next_move":
        pos = int(args[0])

        opponent_move(board_index, pos)
        return player_move(board_index)
    
    elif command == "win":
        print("We won.")
        return -1
    
    elif command == "loss":
        print("We lost.")
        return -1

    return 0

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = int(sys.argv[2])

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

if __name__ == '__main__':
    pass
