from copy import deepcopy
import random
from typing import Dict, List, Tuple
import time
import numpy as np
from connect4.utils import Integer, get_pts, get_valid_actions


class AIPlayer:
    def evaluation(self,state):
        my_player_number = self.player_number
        v1 = get_pts(my_player_number, state[0])
        v2 = get_pts(3-my_player_number,state[0])
        return v1**2-(1 + 1.5*(self.get_number_of_filled_cells(state[0])/(state[0].shape[0]*state[0].shape[1])))*v2**2
    def __init__(self, player_number: int, time: int):
        """
        :param player_number: Current player number
        :param time: Time per move (seconds)
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.time = time
        self.depth = 4
        self.counter = 0
        # Do the rest of your implementation here

    


        # Do the rest of your implementation here
        # raise NotImplementedError('Whoops I don\'t know what to do')
    
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
    def evaluation_node( self, state : Tuple[np.array, Dict[int, Integer]] , alpha , beta) : 
        """
            returns the Tuple [ max of all expectation node among all children, best_Action  ] 
        """
        my_player_number = 3-self.player_number
        valid_actions  = get_valid_actions(my_player_number, state)
        total_number_of_valid_actions = len(valid_actions)
        if( total_number_of_valid_actions == 0 ) or (self.depth == 0):
            # print(state)
            # return (get_pts(my_player_number, state[0]), None)
            v1 = get_pts(my_player_number, state[0])
            v2 = get_pts(3-my_player_number,state[0])
            return self.evaluation(state)
        best_value = None
        for action in valid_actions:
            self.counter += 1
            next_state = self.apply_action(action, state, my_player_number)
            self.depth -= 1
            child_value = self.minimax_node(next_state, alpha, beta)[0]
            self.depth += 1
            if( best_value is None ):
                # best_action = action
                best_value = child_value
            elif( child_value < best_value ):
                best_value = child_value
                # best_action = action
            if (beta is None) :
                beta = best_value
            elif (best_value < beta):
                beta = best_value
            if (alpha is not None):
                if beta <= alpha:
                    break
        return best_value



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
    def minimax_node( self, state : Tuple[np.array, Dict[int, Integer]] , alpha = None, beta = None) : 
        """
            returns the Tuple [ max of all expectation node among all children, best_Action  ] 
        """
        my_player_number = self.player_number
        valid_actions  = get_valid_actions(my_player_number, state)
        total_number_of_valid_actions = len(valid_actions)
        if( total_number_of_valid_actions == 0 ) or (self.depth == 0):
            # if( total_number_of_valid_actions == 0 ):
            #     print("herewego")
            return (self.evaluation(state),None,valid_actions)
        best_value, best_action = None, None 
        for action in valid_actions:
            # print(action,self.depth)
            # self.counter += 1
            next_state = self.apply_action(action, state, my_player_number)
            self.depth -= 1
            child_value = self.evaluation_node(next_state, alpha, beta)
            self.depth += 1
            if( best_value is None ):
                best_action = action
                best_value = child_value
            elif( child_value > best_value ):
                best_value = child_value
                best_action = action
            if (alpha is None) :
                alpha = best_value
            elif (best_value > alpha):
                alpha = best_value
            if (beta is not None):
                if beta <= alpha:
                    break

                
        # if best_action is None:
        #     print(best_value, len(valid_actions))
        return (best_value, best_action,valid_actions)
    def get_minimax_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
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
        # self.depth = 5
        st = time.time()
        ans = self.minimax_node(state)
        print(self.counter,self.depth,time.time()-st)
        # while self.counter < 5000 and self.depth < 100 and time.time()-st < self.time/10:
        while self.depth < 100 and time.time()-st < self.time/6:
            self.counter = 0
            self.depth += 1
            ans = self.minimax_node(state)
            # if self.depth %  == 0:
            # print(self.counter,self.depth,time.time()-st)
        self.counter = 0
        print(ans)
        time.sleep(1)
        return ans[1] 
        


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
        # print(ans)
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
        ans = self.get_minimax_move(state)
        # print(self.counter)
        # self.counter = 0
        print(ans)
        return ans
        # raise NotImplementedError('Whoops I don\'t know what to do')

