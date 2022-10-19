from copy import deepcopy
from os import stat
import random
from math import log1p
from typing import Dict, List, Tuple, Union
import time
import numpy as np
from connect4.utils import Integer, get_pts, get_valid_actions
from connect4.config import win_pts
from queue import PriorityQueue
import traceback

class AIPlayer:
    
    def get_row_score(self, player_number: int, row: Union[np.array, List[int]]):
        score = 0
        n = len(row)
        j = 0
        while j < n:
            if row[j] == player_number:
                count = 0
                while j < n and row[j] == player_number:
                    count += 1
                    j += 1
                k = len(self.win_pts) - 1
                score += self.win_pts[count % k] + (count // k) * self.win_pts[k]
            else:
                j += 1
        return score


    def get_diagonals_primary(self, board: np.array) -> List[int]:
        m, n = board.shape
        for k in range(n + m - 1):
            diag = []
            for j in range(max(0, k - m + 1), min(n, k + 1)):
                i = k - j
                diag.append(board[i, j])
            yield diag


    def get_diagonals_secondary(self, board: np.array) -> List[int]:
        m, n = board.shape
        for k in range(n + m - 1):
            diag = []
            for x in range(max(0, k - m + 1), min(n, k + 1)):
                j = n - 1 - x
                i = k - x
                diag.append(board[i][j])
            yield diag


    def get_pts(self,player_number: int, board: np.array) -> int:
        """
        :return: Returns the total score of player (with player number) on the board
        """
        score = 0
        score2 = 0
        m, n = board.shape
        # score in rows
        for i in range(m):
            score += self.get_row_score(player_number, board[i])
            score2 += self.get_row_score(3-player_number, board[i])
        # score in columns
        for j in range(n):
            score += self.get_row_score(player_number, board[:, j])
            score2 += self.get_row_score(3-player_number, board[:, j])
        # scores in diagonals_primary
        for diag in self.get_diagonals_primary(board):
            score += self.get_row_score(player_number, diag)
            score2 += self.get_row_score(3-player_number, diag)

        # scores in diagonals_secondary
        for diag in self.get_diagonals_secondary(board):
            score += self.get_row_score(player_number, diag)
            score2 += self.get_row_score(3-player_number, diag)
        return (score, score2)
    def evaluation(self,state):
        my_player_number = self.player_number
        filled_cells = self.get_number_of_filled_cells(state[0])
        frc = (filled_cells/(state[0].shape[0]*state[0].shape[1]))
        # self.win_pts = [i**((2-frc)) for i in win_pts]
        self.win_pts = win_pts
        (v1,v2) = self.get_pts(my_player_number, state[0])
        v3 = v1 - self.v1bar
        v4 = v2 - self.v2bar
        # (v3, v4) = self.get_pts(my_player_number, state[0])
        # v2 = self.get_pts(3-my_player_number,state[0])
        # return v1**2-(1 - frc*(frc-1))*v2**2
        if self.mode == 'random':
            return ((v1+1e-8)/(v2+1e-8))**3
        else:
            # if(self.v1bar > self.v2bar):
            (k1,k3) = (1,2+frc) if v3 > 0 else (-2-frc,-1)
            (k2,k4) = (2+frc, 1) if v4 > 0 else (-1,-2-frc)
            a = log1p(abs(v3))
            b = log1p(abs(v4))
            return (k1*a - k2*b,k3*a-k4*b)
    def overall(self,state,action, frc):
        numcol = state[0].shape[1]
        numrows = state[0].shape[0]
        # frc = self.get_number_of_filled_cells(state[0])
        mid = abs(numcol/2 - action[0]-1)
        ret = -(mid)
        # for j in range(numcol):
        #     for i in range(numrows):
        #         if state[0][i][j] != 0:
        #             ret += mid*(numrows-i)
        #             break

        if state[0][numrows-1][action[0]] == 0:
            ret -= 5*state[1][self.player_number].get_int()*(1+frc) 
        if action[1] and state[0][numrows-1][action[0]] == self.player_number:
            ret -= 40
        return ret
    
    def __init__(self, player_number: int, time: int):
        """
        :param player_number: Current player number
        :param time: Time per move (seconds)
        """
        self.player_number = player_number
        self.type = 'ai'
        self.v1bar = 0
        self.v2bar = 0
        self.player_string = 'Player {}:ai'.format(player_number)
        self.time = time
        self.depth = 3
        # self.depth = 
        # self.depth = 6
        self.bmu = 0
        self.maxd = 0
        self.counter = 0
        self.newc = 0
        self.cn1 = 0
        self.cn2 = 0
        self.store_action = PriorityQueue()
        self.win_pts = win_pts
        self.mode = 'random'
        # Do the rest of your implementation here

    


    def get_number_of_filled_cells( self, board : np.array ) -> int :
        """
            returns the number of filled cells on the board
        """
        total_cells = board.shape[0] * board.shape[1]
        zero_cells = np.count_nonzero(board==0)
        return total_cells - zero_cells

    def get_player_number( self, board : np.array ) -> int : 

        return self.player_number
        # filled_cells = get_number_of_filled_cells(board)
        # return 1 when (filled_cells % 2 == 0) else 2

    def apply_action( self, action : Tuple[int, bool],  state : Tuple[np.array, Dict[int, Integer]], player : int ) -> Tuple[np.array, Dict[int, Integer]] :
        """
            returns the new state after applying the action on the given state
            player: player number playing the action
        """
        m, n = state[0].shape
        my_player_number = self.player_number
        is_popout = action[1]
        column = action[0]
        # next_state = (state[0]+0,state[1]) # to be returned by this function
        next_state = deepcopy(state)
        if is_popout:
            next_state[1][player] = Integer(next_state[1][player].get_int()-1)
            # next_state[1][player].decrement() # players pop out moves decreases
            next_state[0][0][column] = 0  # first value in column will become zero
            # shift values in the columns
            for row in range(m-1,0,-1):
                next_state[0][row][column] = next_state[0][row-1][column] 

        else:
            empty_column = True
            for row in range(m):
                if(state[0][row][column] != 0 ):
                    # first non zero value in this column
                    next_state[0][row-1][column] = player
                    empty_column = False
                    break
            if( empty_column ):
                next_state[0][m-1][column] = player
        return next_state



    def expectation_node(self, state : Tuple[np.array, Dict[int, Integer]]) -> int :
        """
            returns the mean of value of all children
        """
        if( (self.expectimax_st + self.time ) - time.time() < 0.3 ):
            raise Exception("Time out")       
        adversary_number = 3 - self.player_number
        valid_actions  = get_valid_actions(adversary_number, state)
        total_number_of_valid_actions = len(valid_actions)
        if( total_number_of_valid_actions == 0 ) or self.depth == 0:
            v1 = get_pts(self.player_number, state[0])
            v2 = get_pts(3-self.player_number,state[0])
            return self.evaluation(state)
        sum_of_children = 0
        for action in valid_actions:
            self.counter += 1
            next_state = self.apply_action(action, state, adversary_number)
            self.depth -= 1
            child_value, _ = self.expectimax_node(next_state)
            self.depth += 1
            sum_of_children += child_value
        
        return sum_of_children / total_number_of_valid_actions
    def evaluation_node( self, state : Tuple[np.array, Dict[int, Integer]],store , alpha , beta) : 
        """
            returns the Tuple [ max of all expectation node among all children, best_Action  ] 
        """
        if( (self.intelligent_st + self.time ) - time.time() < 1 ):
            print(self.counter)
            raise Exception("Time out")
        my_player_number = 3-self.player_number
        valid_actions  = get_valid_actions(my_player_number, state)
        total_number_of_valid_actions = len(valid_actions)
        if( total_number_of_valid_actions == 0 ) or (self.depth == 0):
            # print(state)
            # return (get_pts(my_player_number, state[0]), None)
            # v1 = get_pts(my_player_number, state[0])
            # v2 = get_pts(3-my_player_number,state[0])
            return (self.evaluation(state),[])
        best_value = None
        best_move = None
        explored = {-1}
        storec = PriorityQueue()
        itt = 0
        oldbestval = None
        while not store.empty():
            self.counter += 1
            itt += 1
            k = store.get()
            if (oldbestval is None):
                oldbestval = k[0]
            action = k[1]
            explored.add(action)
            cnd = True
            # if self.maxd - self.depth < 2:
            #     if(k[0] - oldbestval) > 50:
            #         # print("blah",k[0],oldbestval)
            #         cnd = False
                    # cnd  F
            if self.maxd - self.depth < 2 and itt > 3:
                # print("Painn", self.depth, self.maxd - self.depth, itt)
                self.cn1 += 1
            if self.maxd - self.depth < 4 and itt > 6:
                self.cn2 += 1
            if cnd:
                next_state = self.apply_action(action, state, my_player_number)
                self.depth -= 1
                x = self.minimax_node(next_state, k[2], alpha, beta)
                child_value = x[0]
                best_move = x[1]
                best_move.append(action)
                self.depth += 1
                storec.put((child_value[1],k[1],k[2]))
                if( best_value is None ):
                    # best_action = action
                    best_value = child_value
                elif( child_value[1] < best_value[1] ):
                    self.newc += 1
                    best_value = child_value
                    # best_action = action
                if (beta is None) :
                    beta = best_value[1]
                elif (best_value[1] < beta):
                    beta = best_value[1]
                if (alpha is not None):
                    if beta <= alpha:
                        break
            else:
                storec.put((1e9,k[1],k[2]))

        if( alpha is not None):
            if (beta is not None):
                if  beta <= alpha:
                    return (best_value,best_move)
        while not storec.empty():
            t = storec.get()
            store.put(t)
        for action in valid_actions:
            if action not in explored:
                explored.add(action)
                newpq = PriorityQueue()
                self.counter += 1
                next_state = self.apply_action(action, state, my_player_number)
                self.depth -= 1
                x = self.minimax_node(next_state, newpq, alpha, beta)
                child_value = x[0]
                store.put((child_value[1],action,newpq))
                self.depth += 1
                if( best_move is None ):
                    best_move = x[1]
                    best_move.append(action)
                    # best_action = action
                    best_value = child_value
                elif( child_value[1] < best_value[1] ):
                    best_value = child_value
                    best_move = x[1]
                    best_move.append(action)
                    # best_action = action
                if (beta is None) :
                    beta = best_value[1]
                elif (best_value[1] < beta):
                    beta = best_value[1]
                if (alpha is not None):
                    if beta <= alpha:
                        break
        return (best_value,best_move)



    def expectimax_node( self, state : Tuple[np.array, Dict[int, Integer]] ) : 
        """
            returns the Tuple [ max of all expectation node among all children, best_Action  ] 
        """
        if( (self.expectimax_st + self.time ) - time.time() < 0.3 ):
            raise Exception("Time out")

        my_player_number = self.player_number
        valid_actions  = get_valid_actions(my_player_number, state)
        total_number_of_valid_actions = len(valid_actions)
        if( total_number_of_valid_actions == 0 ) or (self.depth == 0):
            # print(state)
            # return (get_pts(my_player_number, state[0]), None)
            # print(valid_actions,self.depth)
            return (self.evaluation(state),None)
        best_value, best_action = None, None       
        for action in valid_actions:
            self.counter += 1
            next_state = self.apply_action(action, state, my_player_number)
            self.depth -= 1
            child_value = self.expectation_node(next_state)
            self.depth += 1
            if( best_value is None ):
                best_action = action
                best_value = child_value
            elif( child_value > best_value ):
                best_value = child_value
                best_action = action
        return (best_value, best_action)
    def minimax_node( self, state : Tuple[np.array, Dict[int, Integer]],store , alpha = None, beta = None) : 
        """
            returns the Tuple [ max of all expectation node among all children, best_Action  ] 
        """
        if( (self.intelligent_st + self.time ) - time.time() < 1 ):
            print(self.counter)
            raise Exception("Time out")
        my_player_number = self.player_number
        valid_actions  = get_valid_actions(my_player_number, state)
        total_number_of_valid_actions = len(valid_actions)
        if( total_number_of_valid_actions == 0 ) or (self.depth == 0):
            # if( total_number_of_valid_actions == 0 ):
            #     print("herewego")
            return (self.evaluation(state),[],valid_actions)
        best_value, best_action = None, None
        explored = {-1}
        # storec = deepcopy(store)
        storec = PriorityQueue()
        itt = 0
        oldbestval = None
        while not store.empty():
            self.counter += 1
            itt += 1
            k = store.get()
            if oldbestval is None:
                oldbestval = k[0]
            # print(k)
            action = k[1]
            explored.add(action)
            cnd = True
            # if self.maxd - self.depth < 2:
            #     if(k[0] - oldbestval) < -50:
            #         # print("blah",k[0],oldbestval)
            #         cnd = False
            if self.maxd - self.depth < 2 and itt > 3:
                # print("Painn", self.depth, self.maxd - self.depth, itt)
                self.cn1 += 1
                # cnd = False
            if self.maxd - self.depth < 4 and itt > 6:
                self.cn2 += 1
                # print("Pain", self.depth, self.maxd - self.depth, itt)
            if cnd:
                next_state = self.apply_action(action, state, my_player_number)
                self.depth -= 1
                x = self.evaluation_node(next_state, k[2], alpha, beta)
                child_value = x[0]
                self.depth += 1
                storec.put((-child_value[0],k[1],k[2]))
                if( best_value is None ):
                    best_action = x[1]
                    best_action.append(action)
                    best_value = child_value
                elif( child_value[0] > best_value[0] ):
                    best_value = child_value
                    # best_action = action
                    best_action = x[1]
                    best_action.append(action)
                    self.newc += 1
                    if self.depth == self.maxd:
                        # print("best move updated")
                        self.bmu += 1
                    # if self.depth >= 4:
                    #     if itt > 3:
                    #         print(self.depth-self.maxd, itt)
                    
                if (alpha is None) :
                    alpha = best_value[0]
                elif (best_value[0] > alpha):
                    alpha = best_value[0]
                if (beta is not None):
                    if beta <= alpha:
                        break
            else:
                storec.put((1e9,k[1],k[2]))
        while not storec.empty():
            t = storec.get()
            store.put(t)
        if( alpha is not None):
            if (beta is not None):
                if  beta <= alpha:
                    # print(len(valid_actions)+1-len(explored))
                    return (best_value, best_action,valid_actions)

        for action in valid_actions:
            # print(action,self.depth)
            self.counter += 1
            if action not in explored:
                explored.add(action)
                newpq = PriorityQueue()
                next_state = self.apply_action(action, state, my_player_number)
                # dec = False
                # if self.depth > 2:
                #     self.depth -= 1

                self.depth -= 1
                x = self.evaluation_node(next_state, newpq, alpha, beta)
                child_value = x[0]
                self.depth += 1
                store.put((-child_value[0],action,newpq))

                if( best_value is None ):
                    best_action = x[1]
                    best_action.append(action)
                    best_value = child_value
                elif( child_value[0] > best_value[0] ):
                    best_value = child_value
                    # best_action = action
                    best_action = x[1]
                    best_action.append(action)
                if (alpha is None) :
                    alpha = best_value[0]
                elif (best_value[0] > alpha):
                    alpha = best_value[0]
                if (beta is not None):
                    if beta <= alpha:
                        break

                
        # if best_action is None:
        #     print(best_value, len(valid_actions))
        return (best_value, best_action,valid_actions)
    def get_minimax_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:

        # Do the rest of your implementation here
        # self.depth = 4
        # state[1][1] = Integer(min(state[1][1].get_int(),1))
        # state[1][2] = Integer(min(state[1][2].get_int(),1))
        self.intelligent_st = time.time()
        (self.v1bar,self.v2bar) = self.get_pts(self.player_number, state[0])
        valid_actions  = get_valid_actions(self.player_number, state)
        best_action = None
        best_val = None
        while self.depth < 30:
            self.maxd = self.depth
            for action in valid_actions:
                try:
                    ns = self.apply_action(action,state,self.player_number)
                    frc = self.get_number_of_filled_cells(state[0])
                    ans = self.minimax_node(ns,self.store_action)
                    # x = list(reversed(ans[1]))[0]
                    t = self.overall(state, action, frc)
                    y = t*(1-frc)+ans[0][0]*100
                    print(action, t, y)
                    if best_action is None:
                        best_action = action
                        best_val = y
                    if y > best_val:
                        best_action = action
                        best_val = y
                    # print(self.counter,end="; ")
                    # print(self.counter,self.newc,self.depth,time.time()-self.intelligent_st)
                except Exception as e:
                    # print(traceback.format_exc())
                    print(best_val,best_action)
                    return best_action
                    # print(e.)
                    # print(e,self.bmu,self.cn1,self.cn2,self.maxd)
                    break
            self.depth += 1
                
                    

                 


        # ans = self.minimax_node(state,self.store_action)
        # print(self.counter,self.depth,time.time()-self.intelligent_st)
        # while self.counter < 5000 and self.depth < 100 and time.time()-st < self.time/10:

        # print(ans)
        # time.sleep(1)
        # print(ans)
        # print(ans[0])
        # print(x)
        # ns  = state
        # pn = self.player_number
        # print(pn)
        # self.win_pts = win_pts
        # for elem in x:
        #     # print(t[0],t[1])
        #     ns = self.apply_action(elem, ns, pn)
        #     (v3, v4) = self.get_pts(self.player_number, ns[0])
        #     print(ns[0],ns[1][1].get_int(),ns[1][2].get_int(),v3,v4)
        #     pn = 3 - pn
        # print(self.evaluation(ns),v3,v4)
        # print(ns[0])
        return best_action 
        


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
        # Do the rest of your implementation here
        self.depth = 1
        self.expectimax_st = time.time()    
        self.mode = 'random'    
        # ans = self.expectimax_node(state)
        # print(ans)
        # print(self.counter,self.depth)
        # while self.counter < 1000 and self.depth < 100:
        while self.depth < 100:
            self.depth += 1
            self.counter = 0
            try:
                ans = self.expectimax_node(state)
            except:
                print("Time about to end !!!")
                break
            # if x[1] is not None:
            #     ans = x
            # else:
            #     break
            # print(ans)
            # print(self.counter,self.depth)
        # self.counter = 0
        # time.sleep(1)
        
        
        return ans[1]
        # raise NotImplementedError('Whoops I don\'t know what to do')
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
        self.mode = 'ai'
        ans = self.get_minimax_move(state)
        # print(self.counter)
        # self.counter = 0
        # print(ans)
        return ans
        # raise NotImplementedError('Whoops I don\'t know what to do')

