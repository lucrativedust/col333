from pickletools import optimize
from queue import PriorityQueue
from tracemalloc import start


class SentenceCorrector(object):
    
    def __init__(self, cost_fn, conf_matrix):
        self.conf_matrix = conf_matrix
        self.cost_fn = cost_fn
        self.inverse_conf_matrix = {}
        for char in self.conf_matrix:
            for replacement_char in self.conf_matrix[char]:
                if( replacement_char not in self.inverse_conf_matrix):
                    self.inverse_conf_matrix[replacement_char] = []
                self.inverse_conf_matrix[replacement_char].append(char)
        # print(self.inverse_conf_matrix)
        # You should keep updating following variable with best string so far.
        self.best_state = None  
        
    def optimize_word_helper( self, word, idx, word_idx):
        """
            iterates over all letters and checks all possible replacements by DFS
        """
        if( idx == len(word) ) :
            words = self.start_state.split()
            ### update string with new word
            words[word_idx] = word
            new_string = ' '.join(words)
            ### check if this improves cost
            if( self.cost_fn(new_string) < self.best_word_cost ) :
                self.best_word  = word
                self.best_word_cost = self.cost_fn(new_string)
                self.best_state = new_string ### since timer is likely to run out we keep storing the best answers
            return
        
        self.optimize_word_helper( word, idx+1, word_idx)
        
        for replace_current_char in self.conf_matrix[word[idx]]:
            self.optimize_word_helper(word[:idx] + replace_current_char + word[idx+1:], idx+1, word_idx)
            
        return
    
    def optimize_word_2( self, word, word_idx):
        """
            replaces only character and finds best possibility
        """
        self.best_word = word 
        self.best_word_cost = self.cost_fn(self.start_state)
        words = self.start_state.split()
        for i in range(len(word)):
            for replace_current_char in self.conf_matrix[word[i]]:
                new_word = word[:i] + replace_current_char + word[i+1:]
                words[word_idx] = new_word
                new_string = ' '.join(words)
                if( self.cost_fn(new_string) < self.best_word_cost  ) : 
                    self.best_word = new_word
                    self.best_word_cost = self.cost_fn(new_string)

    def optimize_word_3( self, word, word_idx, beam_size=425, beam_depth=4, best_n = 5):
        """
            do a beam search on a word to explore all possibilities
        """
        bwq = PriorityQueue()
        self.best_word = word
        self.best_word_cost = self.cost_fn(self.current_state)
        bwq.put((self.best_word_cost,self.best_word))
        # print(f"\nbest word is {word}, {self.best_word_cost}")
        d = {self.best_word}
        # d.add(self.best_word)
        # self.best_word_cost = self.cost_fn(self.current_state)
        # words = self.start_state.split() ## holds all the words in the string
        words = self.current_state.split() ### makes use of all words corrected till now
        beam = [word]
        for depth in range(beam_depth):
            ## varialbles for next depth
            queue = PriorityQueue()
            new_beam = []
            
            ### process all solutions in the beam
            for current_solution in beam :
                
                ## evaluate all neighbours
                for char_idx in range(len(word)):
                    if( current_solution[char_idx] in self.conf_matrix):
                        for replace_current_char in self.conf_matrix[current_solution[char_idx]]:
                            new_word = current_solution[:char_idx] + replace_current_char + current_solution[char_idx+1:]
                            words[word_idx] = new_word
                            new_string = ' '.join(words)

                            if new_word not in d:
                                d.add(new_word)
                                ft = self.cost_fn(new_string)
                                # if ft < 1.05*self.best_word_cost:
                                queue.put((ft, new_word))
                                bwq.put((ft, new_word))
                            #optimize here
                
            next_beam_size = 0
            # print(queue.qsize())   
            while ( next_beam_size < beam_size and ( not queue.empty())):
                best_word_tuple = queue.get()
                if best_word_tuple[0] < 1.05*self.best_word_cost:
                    new_beam.append(best_word_tuple[1])
                    next_beam_size += 1
                    if( best_word_tuple[0] < self.best_word_cost):
                        self.best_word_cost = best_word_tuple[0]
                        self.best_word = best_word_tuple[1]
                else:
                    break
            beam = new_beam
        ans = []
        a = bwq.get()
        ans.append(a[1])
        for _ in range(best_n-1):
            if not bwq.empty():
                t = bwq.get()
                if a[0]*1.2 > t[0]:
                    ans.append(t[1])
                else:
                    break
            else:
                break
                # print(t,end=" ")
        self.best_words.append(ans)
        # print()

        # return list of best words
                
    def optimize_word( self, word, word_idx):
        self.best_word = word
        self.best_word_cost = self.cost_fn(self.start_state)
        self.optimize_word_helper( word, 0, word_idx)
    
    def per_word_optimization( self, start_state ,beam_size = 10, beam_depth = 10) :
        self.best_state = start_state
        # ans = ""
        self.current_state = self.start_state
        self.best_words = []
        words = start_state.split()
        for word_idx, word in enumerate(start_state.split()) :
            # self.optimize_word(word, word_idx)
            # self.optimize_word_2(word, word_idx)
            self.optimize_word_3( word, word_idx)
            # print(lr,end=" ")
            optimized_word = self.best_word
            words[word_idx] = optimized_word ### update the optimized word for the future words
            self.current_state = ' '.join(words)
            # # print(self.current_state)
            self.best_state = self.current_state
            # if( ans !=  "" ) : 
            #     ans += " "
            # ans += optimized_word
            # print(f"{word} => {optimized_word}")
        print("Starting the main search")
        # for elem in self.best_words:
        #     if len(elem) > 3:
        #         print(elem,end=" ")
        # print(self.best_words[8])
        # self.best_state = ans
        beam = [self.best_state]
        for _ in range(beam_depth):
            prq = PriorityQueue()
            for cs in beam:
                cs_words = cs.split(' ')
                for i,word in enumerate(cs_words):
                    for pos_rep in self.best_words[i][1:]:
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
            beam = next_beam




    
    def beam_search_on_sentence(self, start_state, beam_size = 10, beam_depth = 30 ):
        
        self.best_state = start_state
        self.best_cost = self.cost_fn(start_state)
        
        beam = [start_state]
        
        for depth in range(beam_depth):
            
            queue = PriorityQueue()
            # self.conf_matrix = self.inverse_conf_matrix
            for current_solution in beam : 
                for char_idx in range(len(current_solution)):
                    if( current_solution[char_idx] != ' '):
                        if( current_solution[char_idx] in self.conf_matrix):
                            for replacement_char in self.conf_matrix[current_solution[char_idx]]:
                                new_solution = current_solution[:char_idx] + replacement_char + current_solution[char_idx+1:]
                                queue.put((self.cost_fn(new_solution), new_solution))

            next_beam = []

            for i in range(beam_size):
                best_solution_tuple  = queue.get()
                next_beam.append(best_solution_tuple[1])
                if( best_solution_tuple[0] < self.best_cost ) :
                    self.best_cost = best_solution_tuple[0]
                    self.best_state = best_solution_tuple[1]
            beam = next_beam
        
        
    def search(self, start_state):
        """
        :param start_state: str Input string with spelling errors
        """
        self.start_state = start_state
        self.conf_matrix = self.inverse_conf_matrix
        # You should keep updating self.best_state with best string so far.
        self.per_word_optimization(start_state)
        # self.per_word_optimization()
        # self.beam_search_on_sentence(start_state)
        return 
