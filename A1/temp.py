import pickle
lm_file = "./data/lm_model.pkl"
with open(lm_file, 'rb') as fp:
    lm_model = pickle.load(fp)
    
def print_cost(sentence):
    print(f"Cost: {lm_model(sentence)}")
    
sentences = [ "although the growth looks robust on the surface the figures as readity ere not that good he said", 
              "although the growth looks robust on the surface the figures in readity ere not that good he said",
              "officials seen arriving by reuter correspondents imcduded west german finance mirister gerhard stoltenberg and bundesbank president karl otto poehl french finance minister edouard balladur and his central banker jacques de larosiere"]

for sentence in sentences:
    print_cost(sentence)