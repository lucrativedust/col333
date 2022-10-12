from cgitb import small
from queue import PriorityQueue
import random

class SentenceCorrector(object):
    
    def __init__(self, cost_fn, conf_matrix):
        self.conf_matrix = conf_matrix
        self.cost_fn = cost_fn
        self.inverse_conf_matrix = {}
        #### invert the confusion matrix according to problem specification
        for char in self.conf_matrix:
            for replacement_char in self.conf_matrix[char]:
                if( replacement_char not in self.inverse_conf_matrix):
                    self.inverse_conf_matrix[replacement_char] = []
                self.inverse_conf_matrix[replacement_char].append(char)
        # You should keep updating following variable with best string so far.
        self.best_state = None  

    def optimize_word_3( self, word, word_idx, beam_size=30000, beam_depth=1, best_n = 20,epsilon=1.05):
        """
            do a beam search on a word to explore all possibilities
        """
         ### we maintain a priority queue to store the best possible replacements of each word
        self.best_word = word
        self.best_word_cost = self.cost_fn(self.current_state)

        self.bwq[word_idx].put((self.best_word_cost,self.best_word))
        self.d[word_idx][self.best_word] = self.best_word_cost
        words = self.current_state.split() ### makes use of all words corrected till now
        # self.wbeams[word_idx] = [word] ## gets updated after every iteration of the beam
        lw = len(word)
        f = 1+(epsilon*lw)
        for depth in range(beam_depth):
            ## varialbles for next depth
            queue = PriorityQueue()
            new_beam = []
            
            ### process all solutions in the beam
            for current_solution in self.wbeams[word_idx] :
                
                ## evaluate all neighbours
                for char_idx in range(len(word)):
                    if( current_solution[char_idx] in self.conf_matrix):
                        for replace_current_char in self.conf_matrix[current_solution[char_idx]]:
                            new_word = current_solution[:char_idx] + replace_current_char + current_solution[char_idx+1:]
                            words[word_idx] = new_word
                            new_string = ' '.join(words)
                            if new_word not in self.d[word_idx]:
                                ft = self.cost_fn(new_string)
                                self.d[word_idx][new_word] = ft
                                queue.put((ft, new_word))
                                self.bwq[word_idx].put((ft, new_word))
                            # else:
                            #     t = self.d[word_idx][new_word]
                            #     if t > ft:
                            #         self.d[word_idx][new_word] = ft
                            #         queue.put((ft, new_word))
                            #         self.bwq[word_idx].put((ft, new_word))


                
            next_beam_size = 0
            ### choose the best neighbours for the next iteration of the beam search
            while ( next_beam_size < beam_size and ( not queue.empty())):
                best_word_tuple = queue.get()
                if best_word_tuple[0] < f*self.best_cost: ### only consider those neighbours which are within some bound
                    new_beam.append(best_word_tuple[1])
                    next_beam_size += 1
                    if( best_word_tuple[0] < self.best_cost):
                        self.best_cost = best_word_tuple[0]
                        self.best_word = best_word_tuple[1]
                else:
                    break
            self.wbeams[word_idx] = new_beam
        ans = [] ### returns the list of best possible possibilities of each word
        a = self.bwq[word_idx].get()
        ans.append(a[1])
        u = []
        u.append(a)
        for _ in range(best_n-1):
            if not self.bwq[word_idx].empty():
                t = self.bwq[word_idx].get()
                if a[0]*1.15 > t[0]: ### consider only those answers which are within some bounds dependent on the best state
                    ans.append(t[1])
                    u.append(t)
                else:
                    break
            else:
                break
        # for elem in u:
        #     self.bwq[word_idx].put(elem)
        self.best_words[word_idx] = ans ### stores the best possible replacement for each word
                
    
    def per_word_optimization( self, start_state ,beam_size = 40, beam_depth = 5,small_beam_depth = 3,epsilon=1.05) :
        self.best_state = start_state
        self.best_cost = self.cost_fn(self.best_state)
        # ans = ""
        self.current_state = self.start_state
        words = start_state.split()
        self.best_words = [[] for _ in words]
        
        for word_idx, word in enumerate(words) :
            self.optimize_word_3( word, word_idx,beam_depth=small_beam_depth,epsilon=epsilon) ### call beam search on each word
            optimized_word = self.best_word
            words[word_idx] = optimized_word ### update the optimized word for the future words
            self.current_state = ' '.join(words)
            self.best_state = self.current_state #### we keep updating the sentence with the best replacement of each word
        ### beam search on the word space where we know the best replacement for each word
        if len(self.beam) == 0:
            self.beam = [self.best_state]
        print("Beam search",small_beam_depth,end=" ")
        for _ in range(beam_depth):
            prq = PriorityQueue()
            for cs in self.beam:
                cs_words = cs.split(' ')
                for i,word in enumerate(cs_words):
                    for pos_rep in self.best_words[i]:
                        new_sol = ' '.join(cs_words[:i]+[pos_rep]+cs_words[i+1:])
                        cost = self.cost_fn(new_sol)
                        prq.put((cost, new_sol))
            next_beam = []
            for _ in range(beam_size):
                possol = prq.get()
                next_beam.append(possol[1])
                if (possol[0] < self.cost_fn(self.best_state)):
                    self.best_cost = possol[0]
                    self.best_state = possol[1]
            self.beam = next_beam




    
        
    def search(self, start_state):
        """
        :param start_state: str Input string with spelling errors
        """
        self.start_state = start_state
        self.best_state = start_state
        self.conf_matrix = self.inverse_conf_matrix
        self.best_cost = self.cost_fn(self.best_state)
        words = start_state.split()
        self.beam = []
        self.wbeams = [[word] for word in words]
        self.d = [{} for _ in words]
        self.bwq = [PriorityQueue() for _ in words]

        # print(self.start_state)
        # You should keep updating self.best_state with best string so far.
        # self.per_word_optimization(start_state) #### main routine of our algorithm
        self.per_word_optimization(start_state=start_state,beam_depth=4,small_beam_depth=1,epsilon=0.035)
        print(self.wbeams[13])
        print("Hola 1",end=" ")
        self.per_word_optimization(start_state=start_state,beam_depth=1,small_beam_depth=1,epsilon=0.035)
        print(self.wbeams[13])
        print("Hola 2",end=" ")
        self.per_word_optimization(start_state=self.best_state,small_beam_depth=1,epsilon=0.005)
        print(self.wbeams[13])
        for i,x in enumerate(words):
            if len(self.wbeams[i]) > 0:
                print(i,len(self.wbeams[i]),end=" ")
        print("Done")



        return 
