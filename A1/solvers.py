from pickletools import optimize


class SentenceCorrector(object):
    def __init__(self, cost_fn, conf_matrix):
        self.conf_matrix = conf_matrix
        self.cost_fn = cost_fn

        # You should keep updating following variable with best string so far.
        self.best_state = None  


        
    def optimize_word_helper( self, word, idx):
        
        if( idx == len(word) ) :
            if( self.cost_fn(word) < self.best_word_cost ) :
                self.best_word  = word
                self.best_word_cost = self.cost_fn(word)
            return
        
        self.optimize_word_helper( word, idx+1)
        
        for replace_current_char in self.conf_matrix[word[idx]]:
            self.optimize_word_helper(word[:idx] + replace_current_char + word[idx+1:], idx+1)
            
        return

    def optimize_word( self, word):
        self.best_word = None
        self.best_word_cost = 1e12
        self.optimize_word_helper( word, 0)
        
    def search(self, start_state):
        """
        :param start_state: str Input string with spelling errors
        """
        # You should keep updating self.best_state with best string so far.
        self.best_state = start_state
        ans = ""
        for word in start_state.split() :
            print(word) 
            self.optimize_word(word)
            optimized_word = self.best_word
            if( ans !=  "" ) : 
                ans += " "
            ans += optimized_word
            print(optimized_word)
        self.best_state = ans
        # self.optimize_word("decpite")
        # print(self.best_word)
        # print(self.cost_fn("despite"))
        return 
        # raise Exception("Not Implemented.")