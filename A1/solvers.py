from pickletools import optimize
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
                self.best_state = new_string
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

    def optimize_word( self, word, word_idx):
        self.best_word = word
        self.best_word_cost = self.cost_fn(self.start_state)
        self.optimize_word_helper( word, 0, word_idx)
    
    def per_word_optimization( self, start_state ) :
        self.best_state = start_state
        ans = ""
        for word_idx, word in enumerate(start_state.split()) :
            # self.optimize_word(word, word_idx)
            self.optimize_word_2(word, word_idx)
            optimized_word = self.best_word
            if( ans !=  "" ) : 
                ans += " "
            ans += optimized_word
            print(f"{word} => {optimized_word}")
        self.best_state = ans
        
    def search(self, start_state):
        """
        :param start_state: str Input string with spelling errors
        """
        self.start_state = start_state
        # You should keep updating self.best_state with best string so far.
        self.per_word_optimization(start_state)
        return 
