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

#
# Game state/model
#

boards = np.zeros((10, 10), dtype="int8")
s = [".","X","O"]
curr = 0 

#
# Static evaluation of game state
#



#
# Minimax algorithm w/ alpha-beta pruning
#



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
