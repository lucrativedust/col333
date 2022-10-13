import random
from tracemalloc import start
import numpy as np
from typing import List, Tuple, Dict
from connect4.utils import get_pts, get_valid_actions, Integer
import time

# 0 based indexing always

class AIPlayer:
    def __init__(self, player_number: int, time: int):
        """
        :param player_number: Current player number
        :param time: Time per move (seconds)
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.time = time
        # Do the rest of your implementation here

    def state_update(self, state, move, isPop, player_number):
        board = state[0]
        new_board = board.copy()
        rows = len(board)
        if not isPop and board[0][move]:
            return -1
        elif isPop and (board[-1][move]==0 or state[1][player_number].get_int()==0 or 1+move%2!=player_number):
            return -1
        elif isPop:
            first_filled = 0
            for i in range(rows):
                if board[i][move]==0:
                    new_board[i][move] = 0
                else :
                    new_board[i][move] = 0
                    first_filled = i
                    break
            for i in range(first_filled, rows-1):
                new_board[i+1][move] = board[i][move]
            return (new_board, {player_number: Integer(state[1][player_number].get_int()-1), 3-player_number: state[1][3-player_number]})
        else :
            first_filled = 0
            for i in range(rows):
                if board[i][move]==0:
                    continue
                else :
                    first_filled = i
                    break
            new_board[first_filled-1][move] = player_number
            return (new_board, state[1])

    def get_expectimax_score(self, state, player_number : int, max_depth : int) -> int :
        move_scores = {}
        columns = len(state[0])
        if max_depth == 0 :
            return self.evaluate(state)
        for col in range(columns) :
            new_state_push = self.state_update(state, col, 0, player_number)
            new_state_pop = self.state_update(state, col, 1, player_number)
            if new_state_push != -1:
                move_scores["push{}".format(col)] = self.get_expectimax_score(new_state_push,3-player_number, max_depth-1)
            if new_state_pop != -1:
                move_scores["pop{}".format(col)] = self.get_expectimax_score(new_state_pop,3-player_number, max_depth-1)
        retval = -1e9
        if player_number == self.player_number:
            for move in move_scores:
                retval = max(retval, move_scores[move])
            return retval
        else :
            retval = 0
            cnt = 0
            for move in move_scores:
                retval += move_scores[move]
                cnt += 1
            if cnt==0:
                return 0
            retval /= cnt
            return retval

    def get_intelligent_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        """
        Given the current state of the board, return the next move
        This will play against either itself or a human player
        :param state: Contains:
                        1. board
                            - a numpy array containing the state of the board using the following encoding:
                            - the board maintains its same two dimensions
                                - row 0 is the top of the board and so is the last row filled
                            - spaces that are unoccupied are marked as 0
                            - spaces that are occupied by player 1 have a 1 in them
                            - spaces that are occupied by player 2 have a 2 in them
                        2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
        :return: action (0 based index of the column and if it is a popout move)
        """
        # Do the rest of your implementation here
        return self.get_expectimax_move(state)
        # raise NotImplementedError('Whoops I don\'t know what to do')

    def get_expectimax_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        """
        Given the current state of the board, return the next move based on
        the Expecti max algorithm.
        This will play against the random player, who chooses any valid move
        with equal probability
        :param state: Contains:
                        1. board
                            - a numpy array containing the state of the board using the following encoding:
                            - the board maintains its same two dimensions
                                - row 0 is the top of the board and so is the last row filled
                            - spaces that are unoccupied are marked as 0
                            - spaces that are occupied by player 1 have a 1 in them
                            - spaces that are occupied by player 2 have a 2 in them
                        2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
        :return: action (0 based index of the column and if it is a popout move)
        """
        start_time = time.time()
        columns = len(state[0][0])
        move = -1
        isPop = 0
        maax = -1e9
        # for depth in range(10):
        #     for col in range(columns) :
        #         new_state_push = self.state_update(state, col, 0, self.player_number)
        #         new_state_pop = self.state_update(state, col, 1, self.player_number)
        #         if new_state_push != -1:
        #             val = self.get_expectimax_score(new_state_push, 3-self.player_number, depth)
        #             if val > maax :
        #                 maax = val
        #                 move = col
        #                 isPop = 0 
        #         if new_state_pop != -1:
        #             val = self.get_expectimax_score(new_state_pop, 3-self.player_number, depth)
        #             if val > maax :
        #                 maax = val
        #                 move = col
        #                 isPop = 1
        #     move = -1
        #     isPop = 0
        #     maax = -1e9
        for col in range(columns) :
            new_state_push = self.state_update(state, col, 0, self.player_number)
            new_state_pop = self.state_update(state, col, 1, self.player_number)
            if new_state_push != -1:
                val = self.get_expectimax_score(new_state_push, 3-self.player_number, 3)
                if val > maax :
                    maax = val
                    move = col
                    isPop = 0 
            if new_state_pop != -1:
                val = self.get_expectimax_score(new_state_pop, 3-self.player_number, 3)
                if val > maax :
                    maax = val
                    move = col
                    isPop = 1
        print(move, isPop)
        return (move, isPop)
        raise NotImplementedError('Whoops I don\'t know what to do')

    def evaluate(self, state: Tuple[np.array, Dict[int, Integer]]) -> int :
        return get_pts(self.player_number, state[0])-get_pts(3-self.player_number, state[0]) 