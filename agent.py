#!/usr/bin/python3

# File: agent.py 
# Course: COMP3411/9814 Artificial Intelligence
# Assessment: Assignment 3 - Nine-Board Tic-Tac-Toe Agent
# Author: Mohammad Mayaz Rakib
# zID: z5361151

# USAGE:
# ./agent.py -p <PORT>

import socket
import sys
import numpy as np # type: ignore

import copy

#
# Game state/model
#

EMPTY = 0
PLAYER = 1
OPPONENT = 2

boards = np.zeros((10, 10), dtype="int8") # 0-th index ignored
s = [".","X","O"]
curr = 0 

#
# Static evaluation of game state
#

# 1 2 3  
# 4 5 6
# 7 8 9

triplets = [(1, 2, 3), (4, 5, 6), (7, 8, 9), \
            (1, 4, 7), (2, 5, 8), (3, 6, 9), \
            (1, 5, 9), (3, 5, 7)]

def check_board_winner(board):
    for tri in triplets:
        i, j, k = tri
        
        if (board[i] == board[j] and \
            board[j] == board[k] and \
            board[k] == board[i]):
            return board[i]
    
    return 0

def check_game_over(boards):
    for board in boards[1:]:
        winner = check_board_winner(board)
        
        if winner:
            return winner
    
    return 0

def eval_winner(winner):
    if winner == PLAYER:
        return 1
    else:
        return -1

#
# Move generation
#

def generate_moves(boards, n, turn):
    board = boards[n]
    empty = []
    moves = []

    for pos, cell in enumerate(board):
        if cell == EMPTY:
            empty.append(pos)

    for pos in empty:
        new_boards = copy.deepcopy(boards)
        new_boards[n][pos] = turn
        moves.append(new_boards)

    return moves

#
# Minimax algorithm w/ alpha-beta pruning
#

def minimax(boards, n, depth, alpha, beta, turn):
    winner = check_game_over(boards)
    if winner != 0 or depth == 0:
        return eval_winner(winner)
    
    if turn == PLAYER:
        max_eval = float('-inf')
        moves = generate_moves(boards, n, turn)

        for move in moves:
            eval = minimax(move, depth - 1, alpha, beta, OPPONENT)
            max_eval = max(max_eval, eval)

            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        
        return max_eval
    
    if turn == OPPONENT:
        min_eval = float('inf')
        moves = generate_moves(boards, n, turn)

        for move in moves:
            eval = minimax(move, depth - 1, alpha, beta, PLAYER)
            min_eval = min(min_eval, eval)

            beta = min(beta, eval)
            if beta <= alpha:
                break

        return min_eval

#
# Actions
#

def play():
    n = np.random.randint(1,9)
    while boards[curr][n] != 0:
        n = np.random.randint(1,9)

    place(curr, n, 1)
    return n

def place( board, num, player ):
    global curr
    curr = num
    boards[board][num] = player

#
# Debug
#

def print_board_row(bd, a, b, c, i, j, k):
    print(" "+s[bd[a][i]]+" "+s[bd[a][j]]+" "+s[bd[a][k]]+" | " \
             +s[bd[b][i]]+" "+s[bd[b][j]]+" "+s[bd[b][k]]+" | " \
             +s[bd[c][i]]+" "+s[bd[c][j]]+" "+s[bd[c][k]])

def print_board(board):
    print_board_row(board, 1,2,3,1,2,3)
    print_board_row(board, 1,2,3,4,5,6)
    print_board_row(board, 1,2,3,7,8,9)
    print(" ------+-------+------")
    print_board_row(board, 4,5,6,1,2,3)
    print_board_row(board, 4,5,6,4,5,6)
    print_board_row(board, 4,5,6,7,8,9)
    print(" ------+-------+------")
    print_board_row(board, 7,8,9,1,2,3)
    print_board_row(board, 7,8,9,4,5,6)
    print_board_row(board, 7,8,9,7,8,9)
    print()

#
# Server
#

def parse(string):
    if "(" in string:
        command, args = string.split("(")
        args = args.split(")")[0]
        args = args.split(",")
    else:
        command, args = string, []

    if command == "second_move":
        place(int(args[0]), int(args[1]), 2)
        return play()
    elif command == "third_move":
        place(int(args[0]), int(args[1]), 1)
        place(curr, int(args[2]), 2)
        return play()
    elif command == "next_move":
        place(curr, int(args[0]), 2)
        return play()

    elif command == "win":
        print("Yay!! We win!! :)")
        return -1

    elif command == "loss":
        print("We lost :(")
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
        for line in text.split("\n"):
            response = parse(line)
            if response == -1:
                s.close()
                return
            elif response > 0:
                s.sendall((str(response) + "\n").encode())

if __name__ == '__main__':
    main()
