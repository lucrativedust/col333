# from copy import deepcopy
# import random
# from typing import Dict, List, Tuple
# import time
# import numpy as np
# from connect4.utils import Integer, get_pts, get_valid_actions


# class AIPlayer:
#     def evaluation(self,state):
#         my_player_number = self.player_number
#         v1 = get_pts(my_player_number, state[0])
#         v2 = get_pts(3-my_player_number,state[0])
#         return v1**2-(1 + 1.5*(self.get_number_of_filled_cells(state[0])/(state[0].shape[0]*state[0].shape[1])))*v2**2
#     def __init__(self, player_number: int, time: int):
#         """
#         :param player_number: Current player number
#         :param time: Time per move (seconds)
#         """
#         self.player_number = player_number
#         self.type = 'ai'
#         self.player_string = 'Player {}:ai'.format(player_number)
#         self.time = time
#         self.depth = 4
#         self.counter = 0
#         # Do the rest of your implementation here

    


#         # Do the rest of your implementation here
#         # raise NotImplementedError('Whoops I don\'t know what to do')
    
#     def get_number_of_filled_cells( self, board : np.array ) -> int :
#         """
#             returns the number of filled cells on the board
#         """
#         total_cells = board.shape[0] * board.shape[1]
#         zero_cells = np.count_nonzero(board==0)
#         return total_cells - zero_cells

#     def get_player_number( self, board : np.array ) -> int : 

#         return self.player_number
#         # filled_cells = get_number_of_filled_cells(board)
#         # return 1 when (filled_cells % 2 == 0) else 2

#     def apply_action( self, action : Tuple[int, bool],  state : Tuple[np.array, Dict[int, Integer]], player : int ) -> Tuple[np.array, Dict[int, Integer]] :
#         """
#             returns the new state after applying the action on the given state
#             player: player number playing the action
#         """
#         m, n = state[0].shape
#         my_player_number = self.player_number
#         is_popout = action[1]
#         column = action[0]
#         # next_state = (state[0]+0,state[1]) # to be returned by this function
#         next_state = deepcopy(state)
#         if is_popout:
#             next_state[1][player] = Integer(next_state[1][player].get_int()-1)
#             # next_state[1][player].decrement() # players pop out moves decreases
#             next_state[0][0][column] = 0  # first value in column will become zero
#             # shift values in the columns
#             for row in range(m-1,0,-1):
#                 next_state[0][row][column] = next_state[0][row-1][column] 

#         else:
#             empty_column = True
#             for row in range(m):
#                 if(state[0][row][column] != 0 ):
#                     # first non zero value in this column
#                     next_state[0][row-1][column] = player
#                     empty_column = False
#                     break
#             if( empty_column ):
#                 next_state[0][m-1][column] = player
#         return next_state



#     def expectation_node(self, state : Tuple[np.array, Dict[int, Integer]]) -> int :
#         """
#             returns the mean of value of all children
#         """
#         if( (self.expectimax_st + self.time ) - time.time() < 0.3 ):
#             raise Exception("Time out")       
#         adversary_number = 3 - self.player_number
#         valid_actions  = get_valid_actions(adversary_number, state)
#         total_number_of_valid_actions = len(valid_actions)
#         if( total_number_of_valid_actions == 0 ) or self.depth == 0:
#             v1 = get_pts(self.player_number, state[0])
#             v2 = get_pts(3-self.player_number,state[0])
#             return self.evaluation(state)
#         sum_of_children = 0
#         for action in valid_actions:
#             self.counter += 1
#             next_state = self.apply_action(action, state, adversary_number)
#             self.depth -= 1
#             child_value, _ = self.expectimax_node(next_state)
#             self.depth += 1
#             sum_of_children += child_value
        
#         return sum_of_children / total_number_of_valid_actions
#     def evaluation_node( self, state : Tuple[np.array, Dict[int, Integer]] , alpha , beta) : 
#         """
#             returns the Tuple [ max of all expectation node among all children, best_Action  ] 
#         """
#         if( (self.intelligent_st + self.time ) - time.time() < 0.5 ):
#             raise Exception("Time out")
#         my_player_number = 3-self.player_number
#         valid_actions  = get_valid_actions(my_player_number, state)
#         total_number_of_valid_actions = len(valid_actions)
#         if( total_number_of_valid_actions == 0 ) or (self.depth == 0):
#             # print(state)
#             # return (get_pts(my_player_number, state[0]), None)
#             v1 = get_pts(my_player_number, state[0])
#             v2 = get_pts(3-my_player_number,state[0])
#             return self.evaluation(state)
#         best_value = None
#         for action in valid_actions:
#             self.counter += 1
#             next_state = self.apply_action(action, state, my_player_number)
#             self.depth -= 1
#             child_value = self.minimax_node(next_state, alpha, beta)[0]
#             self.depth += 1
#             if( best_value is None ):
#                 # best_action = action
#                 best_value = child_value
#             elif( child_value < best_value ):
#                 best_value = child_value
#                 # best_action = action
#             if (beta is None) :
#                 beta = best_value
#             elif (best_value < beta):
#                 beta = best_value
#             if (alpha is not None):
#                 if beta <= alpha:
#                     break
#         return best_value



#     def expectimax_node( self, state : Tuple[np.array, Dict[int, Integer]] ) : 
#         """
#             returns the Tuple [ max of all expectation node among all children, best_Action  ] 
#         """
#         if( (self.expectimax_st + self.time ) - time.time() < 0.3 ):
#             raise Exception("Time out")

#         my_player_number = self.player_number
#         valid_actions  = get_valid_actions(my_player_number, state)
#         total_number_of_valid_actions = len(valid_actions)
#         if( total_number_of_valid_actions == 0 ) or (self.depth == 0):
#             # print(state)
#             # return (get_pts(my_player_number, state[0]), None)
#             # print(valid_actions,self.depth)
#             return (self.evaluation(state),None)
#         best_value, best_action = None, None       
#         for action in valid_actions:
#             self.counter += 1
#             next_state = self.apply_action(action, state, my_player_number)
#             self.depth -= 1
#             child_value = self.expectation_node(next_state)
#             self.depth += 1
#             if( best_value is None ):
#                 best_action = action
#                 best_value = child_value
#             elif( child_value > best_value ):
#                 best_value = child_value
#                 best_action = action
#         return (best_value, best_action)
#     def minimax_node( self, state : Tuple[np.array, Dict[int, Integer]] , alpha = None, beta = None) : 
#         """
#             returns the Tuple [ max of all expectation node among all children, best_Action  ] 
#         """
#         if( (self.intelligent_st + self.time ) - time.time() < 0.5 ):
#             raise Exception("Time out")
#         my_player_number = self.player_number
#         valid_actions  = get_valid_actions(my_player_number, state)
#         total_number_of_valid_actions = len(valid_actions)
#         if( total_number_of_valid_actions == 0 ) or (self.depth == 0):
#             # if( total_number_of_valid_actions == 0 ):
#             #     print("herewego")
#             return (self.evaluation(state),None,valid_actions)
#         best_value, best_action = None, None 
#         for action in valid_actions:
#             # print(action,self.depth)
#             # self.counter += 1
#             next_state = self.apply_action(action, state, my_player_number)
#             self.depth -= 1
#             child_value = self.evaluation_node(next_state, alpha, beta)
#             self.depth += 1
#             if( best_value is None ):
#                 best_action = action
#                 best_value = child_value
#             elif( child_value > best_value ):
#                 best_value = child_value
#                 best_action = action
#             if (alpha is None) :
#                 alpha = best_value
#             elif (best_value > alpha):
#                 alpha = best_value
#             if (beta is not None):
#                 if beta <= alpha:
#                     break

                
#         # if best_action is None:
#         #     print(best_value, len(valid_actions))
#         return (best_value, best_action,valid_actions)
#     def get_minimax_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
#         """
#         Given the current state of the board, return the next move based on
#         the Expecti max algorithm.
#         This will play against the random player, who chooses any valid move
#         with equal probability
#         :param state: Contains:
#                         1. board
#                             - a numpy array containing the state of the board using the following encoding:
#                             - the board maintains its same two dimensions
#                                 - row 0 is the top of the board and so is the last row filled
#                             - spaces that are unoccupied are marked as 0
#                             - spaces that are occupied by player 1 have a 1 in them
#                             - spaces that are occupied by player 2 have a 2 in them
#                         2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
#         :return: action (0 based index of the column and if it is a popout move)
#         """
#         # Do the rest of your implementation here
#         self.depth = 1
#         self.intelligent_st = time.time()
#         # ans = self.minimax_node(state)
#         # print(self.counter,self.depth,time.time()-self.intelligent_st)
#         # while self.counter < 5000 and self.depth < 100 and time.time()-st < self.time/10:
#         while self.depth < 100:
#             self.counter = 0
#             self.depth += 1
#             try:
#                 ans = self.minimax_node(state)
#             except:
#                 # print("Time about to end")
#                 break
#             # if self.depth %  == 0:
#             # print(self.counter,self.depth,time.time()-st)
#             self.counter = 0
#         # print(ans)
#         # time.sleep(1)
#         return ans[1] 
        


#     def get_expectimax_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
#         """
#         Given the current state of the board, return the next move based on
#         the Expecti max algorithm.
#         This will play against the random player, who chooses any valid move
#         with equal probability
#         :param state: Contains:
#                         1. board
#                             - a numpy array containing the state of the board using the following encoding:
#                             - the board maintains its same two dimensions
#                                 - row 0 is the top of the board and so is the last row filled
#                             - spaces that are unoccupied are marked as 0
#                             - spaces that are occupied by player 1 have a 1 in them
#                             - spaces that are occupied by player 2 have a 2 in them
#                         2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
#         :return: action (0 based index of the column and if it is a popout move)
#         """
#         # Do the rest of your implementation here
#         self.depth = 1
#         self.expectimax_st = time.time()        
#         # ans = self.expectimax_node(state)
#         # print(ans)
#         # print(self.counter,self.depth)
#         # while self.counter < 1000 and self.depth < 100:
#         while self.depth < 100:
#             self.depth += 1
#             self.counter = 0
#             try:
#                 ans = self.expectimax_node(state)
#             except:
#                 print("Time about to end !!!")
#                 break
#             # if x[1] is not None:
#             #     ans = x
#             # else:
#             #     break
#             # print(ans)
#             # print(self.counter,self.depth)
#         # self.counter = 0
#         # time.sleep(1)
#         print(ans)
#         return ans[1]
#         # raise NotImplementedError('Whoops I don\'t know what to do')
#     def get_intelligent_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
#         """
#         Given the current state of the board, return the next move
#         This will play against either itself or a human player
#         :param state: Contains:
#                         1. board
#                             - a numpy array containing the state of the board using the following encoding:
#                             - the board maintains its same two dimensions
#                                 - row 0 is the top of the board and so is the last row filled
#                             - spaces that are unoccupied are marked as 0
#                             - spaces that are occupied by player 1 have a 1 in them
#                             - spaces that are occupied by player 2 have a 2 in them
#                         2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
#         :return: action (0 based index of the column and if it is a popout move)
#         """
#         ans = self.get_minimax_move(state)
#         # print(self.counter)
#         # self.counter = 0
#         # print(ans)
#         return ans
#         # raise NotImplementedError('Whoops I don\'t know what to do')

from cgitb import reset
from operator import mod
import random
from shutil import move
import numpy as np
from typing import List, Tuple, Dict
from connect4.utils import get_pts, get_valid_actions, Integer
from collections import deque

import time

INF = 99999999
NEG_INF = -99999999

class QBoard:
    def __init__(self, board, num_popouts, playerID):
        self.num_rows = board.shape[0]
        self.num_cols = board.shape[1]
        
        self.player_number = playerID
        self.qboard = [deque([board[j][i] for j in range(self.num_rows-1, -1, -1) if board[j][i] != 0]) for i in range(self.num_cols)]
        self.stks = [[] for i in range(self.num_cols)]

        self.pop_left = {1: num_popouts[1].get_int(), 2: num_popouts[2].get_int()}

    def get_np_board(self):
        board = np.zeros((self.num_rows, self.num_cols), dtype = int)
        for i in range(self.num_cols):
            j = self.num_rows - 1	
            for e in self.qboard[i]:
                board[j][i] = e
                j -= 1
        return board

    def filled(self):
        sz = 0
        for col in self.qboard:
            sz += len(col)
        return sz/(self.num_cols*self.num_rows)

    def make_move(self, action, playerID):
        if action[1]:
            #pop out move
            item = self.qboard[action[0]].popleft()
            self.stks[action[0]].append(item)
            self.pop_left[playerID] -= 1
        else:
            #insert move
            self.qboard[action[0]].append(playerID)

    def rev_move(self, action, playerID):
        if action[1]:
            #pop out move
            item = self.stks[action[0]].pop()
            self.qboard[action[0]].appendleft(item)
            self.pop_left[playerID] += 1
        else:
            #insert move
            self.qboard[action[0]].pop()

    def valid_actions(self, playerID):
        valid = []
        for i in range(self.num_cols):
            if len(self.qboard[i]) < self.num_rows:
                valid.append((i, False))
        
        if self.pop_left[playerID] > 0:
            for i in range(self.num_cols):
                if i%2 == playerID - 1:
                    if len(self.qboard[i]) > 0:
                        valid.append((i, True))
        return valid

    def simple_score(self, playerID):
        board = self.get_np_board()
        return get_pts(playerID, board) - get_pts(3 - playerID, board)

        
    def score(self, playerID):
        board = self.get_np_board()
        my_pow = get_pts(playerID, board) 
        ene_pow = get_pts(3 - playerID, board)
        p = self.filled()

        # cent = abs((self.num_cols-1)/2 - move[0]) + len(self.qboard[move[0]])
        # if(len(self.qboard[move[0]]) == 0):
        #     cent += 10
        sc = 0
        for i in range(self.num_cols):
            if board[self.num_rows-1][i] == playerID:
                sc += self.pop_left[1 + i%2]
            elif board[self.num_rows-1][i] == 3-playerID:
                sc -= self.pop_left[1 + i%2]

        return (my_pow-ene_pow - (1-p)*(0.2*ene_pow), my_pow - ene_pow)
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
        self.num_rows = 0
        self.num_cols = 0

        self.start_time = 0
        self.prev_depth = 4


    def ABmini_max(self, qb, move, max_depth, curr_depth, alpha, beta):
        if(self.start_time + self.time - time.time() < 2):
            raise Exception("time up!")

        if(curr_depth == max_depth):
            return qb.score(self.player_number)

        # update board
        if(curr_depth%2 == 0):
            qb.make_move(move, self.player_number)
            valid_actions = qb.valid_actions(3-self.player_number)

        else:
            qb.make_move(move, 3 - self.player_number)
            valid_actions = qb.valid_actions(self.player_number)

        num_actions = len(valid_actions)

        best = 0
        ret = (0,0)
        if num_actions > 0:
            if (curr_depth%2 == 0):
                # min of all actions
                best = INF
                for action in valid_actions:
                    val = self.ABmini_max(qb, action, max_depth, curr_depth+1, alpha, beta)
                    if(val[1] < best):
                        best = val[1]
                        beta = best
                        ret = val

                    if alpha >= beta:
                        break
            else:
                # max of all actions
                best = NEG_INF
                for action in valid_actions:
                    val = self.ABmini_max(qb, action, max_depth, curr_depth+1, alpha, beta)
                    if(best < val[0]):
                        ret = val
                        best = val[0]
                        alpha = best

                    if(alpha >= beta):
                        break
        else:
            ret = qb.score(self.player_number)

        # recerse updates
        if(curr_depth%2 == 0):
            qb.rev_move(move, self.player_number)
        else:
            qb.rev_move(move, 3 - self.player_number)

        # print(ret)
        return ret
    def evalu(self, qb, move, playerID, p):
        ind = move[0]
        col = qb.qboard[ind]
        ret = abs(self.num_cols/2 - ind - 1)*2
        #  + abs(self.num_rows/2 - len(col))
        if (len(col) == 0):
            ret += 5*qb.pop_left[playerID]*(1.1 - p)

        if move[1] and len(col) > 0 and col[0] == playerID:
            ret += 40

        return ret/3
        
    def get_intelligent_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        self.start_time = time.time()

        board, num_popouts = state
        self.num_rows = board.shape[0]
        self.num_cols = board.shape[1]

        best_action = None
        best_exp = NEG_INF

        qb = QBoard(board, num_popouts, self.player_number)

        valid_actions = qb.valid_actions(self.player_number)
        max_depth = 4
        
        while(max_depth <= self.num_cols*self.num_rows):
            alpha = NEG_INF
            beta = INF
            # print(max_depth)
            for action in valid_actions:
                try:
                    exp = self.ABmini_max(qb, action, max_depth, 0, alpha, beta)
                except:
                    # print(best_action, best_exp, "best")
                    return best_action
                p = qb.filled()
                ext = self.evalu(qb, action, self.player_number, p)
                # print(action, exp)
                val = exp[0] - (1-p)*ext
                if val > best_exp:
                    test = action
                    best_action = action
                    best_exp = val
                    alpha = val
            max_depth += 1

        # print(best_action, best_exp, "best")
        return best_action

    def expectimax(self, qb, move, max_depth, curr_depth):
        if(self.start_time + self.time - time.time() < 1):
            raise Exception("time up!")

        if(curr_depth == max_depth):
            return qb.simple_score(self.player_number)

        if(curr_depth%2 == 0):
            qb.make_move(move, self.player_number)
            valid_actions = qb.valid_actions(3-self.player_number)

        else:
            qb.make_move(move, 3 - self.player_number)
            valid_actions = qb.valid_actions(self.player_number)

        num_actions = len(valid_actions)

        best = 0
        if num_actions > 0:
            if (curr_depth%2 == 0):
                # expectaion of all actions
                tot = 0
                for action in valid_actions:
                    tot += self.expectimax(qb, action, max_depth, curr_depth+1)
                best = tot/num_actions
            else:
                # max of all actions
                best = NEG_INF
                for action in valid_actions:
                    best = max(best, self.expectimax(qb, action, max_depth, curr_depth+1))
        else:
            best = qb.simple_score(self.player_number)

        # recerse updates
        if(curr_depth%2 == 0):
            qb.rev_move(move, self.player_number)
        else:
            qb.rev_move(move, 3 - self.player_number)

        return best


    def get_expectimax_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        self.start_time = time.time()

        board, num_popouts = state
        self.num_rows = board.shape[0]
        self.num_cols = board.shape[1]

        qb = QBoard(board, num_popouts, self.player_number)
        valid_actions = qb.valid_actions(self.player_number)

        best_action = None
        best_exp = NEG_INF

        max_depth = 4
        while(max_depth <= self.num_cols*self.num_rows):
            for action in valid_actions:
                try:
                    exp = self.expectimax(qb, action, max_depth, 0)
                except:
                    # print(best_action, best_exp, "best")

                    return best_action
                if exp > best_exp:
                    best_action = action
                    best_exp = exp
            max_depth += 1
        # print(best_action, best_exp, "best")
        return best_action