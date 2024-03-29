import pickle
lm_file = "./data/lm_model.pkl"
with open(lm_file, 'rb') as fp:
    lm_model = pickle.load(fp)
    
def print_cost(sentence):
    print(f"Cost: {lm_model(sentence)}")
    
sentences = [ "officials seen arriving by reuter correspondents imcduded west german finance minister gerhard stoltenberg and bundesbank president karl otto poehl french finance minister edouard balladur and his central banker jacques de larosiere", 
              "officials seen arriving by reuter correspondents included west german finance minister gerhard stoltenberg and bundesbank president karl otto poehl french finance minister edouard balladur and his central banker jacques de larosiere",
              "officials seen arriving by reuter correspondents imcduded west german finance mirister gerhard stoltenberg and bundesbank president karl otto poehl french finance minister edouard balladur and his central banker jacques de larosiere"]

for sentence in sentences:
    print_cost(sentence)