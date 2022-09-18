from pickletools import optimize
from queue import PriorityQueue
from tracemalloc import start


class SentenceCorrector(object):
    
    def __init__(self, cost_fn, conf_matrix):
        self.conf_matrix = conf_matrix
        self.cost_fn = cost_fn

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

    def optimize_word_3( self, word, word_idx, beam_size = 10, beam_depth=5):
        """
            do a beam search on a word to explore all possibilities
        """
        self.best_word = word
        self.best_word_cost = self.cost_fn(self.start_state)
        # words = self.start_state.split() ## holds all the words in the string
        words = self.current_state.split() ### makes use of all words corrected till now
        beam = [word]
        for depth in range(beam_depth):
            ## varialbles for next depth
            queue = PriorityQueue()
            new_beam = []
            
            ### process all solutions in the beam
            for current_solution in beam :
                
                ### check if a solution gives a better answers 
                # words[word_idx] = current_solution
                # new_string = ' '.join(words)
                # if( self.cost_fn(new_string) < self.best_word_cost) :
                #     self.best_word_cost = self.cost_fn(new_string)
                #     self.best_word = current_solution
                
                ## evaluate all neighbours
                for char_idx in range(len(word)):
                    for replace_current_char in self.conf_matrix[current_solution[char_idx]]:
                        new_word = current_solution[:char_idx] + replace_current_char + current_solution[char_idx+1:]
                        words[word_idx] = new_word
                        new_string = ' '.join(words)
                        queue.put((self.cost_fn(new_string), new_word))
                
            next_beam_size = 0 
            while( next_beam_size < beam_size and ( not queue.empty())):
                best_word_tuple = queue.get()
                if(best_word_tuple[1] not in new_beam):
                    next_beam_size += 1
                    new_beam.append(best_word_tuple[1])
                    if( best_word_tuple[0] < self.best_word_cost):
                        self.best_word_cost = best_word_tuple[0]
                        self.best_word = best_word_tuple[1]
            beam = new_beam
                
    def optimize_word( self, word, word_idx):
        self.best_word = word
        self.best_word_cost = self.cost_fn(self.start_state)
        self.optimize_word_helper( word, 0, word_idx)
    
    def per_word_optimization( self, start_state ) :
        self.best_state = start_state
        ans = ""
        self.current_state = self.start_state
        words = start_state.split()
        for word_idx, word in enumerate(start_state.split()) :
            # self.optimize_word(word, word_idx)
            # self.optimize_word_2(word, word_idx)
            self.optimize_word_3( word, word_idx)
            optimized_word = self.best_word
            words[word_idx] = optimized_word ### update the optimized word for the future words
            self.current_state = ' '.join(words)
            self.best_state = self.current_state
            if( ans !=  "" ) : 
                ans += " "
            ans += optimized_word
            # print(f"{word} => {optimized_word}")
        self.best_state = ans
        
    def search(self, start_state):
        """
        :param start_state: str Input string with spelling errors
        """
        self.start_state = start_state
        # You should keep updating self.best_state with best string so far.
        self.per_word_optimization(start_state)
        return 
