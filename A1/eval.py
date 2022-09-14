import pickle
lm_file = "./data/lm_model.pkl"
input_file = "./data/input.txt"
pred_file = "./data/pred.txt"
target_file = "./data/output.txt"
input = open( input_file, 'r')
prediction = open( pred_file, 'r')
target = open( target_file, 'r')

with open(lm_file, 'rb') as fp:
    lm_model = pickle.load(fp)

lm_model.unk_prob = 1e-20
lm_model.set_mode('spell_check')

input_sentences = input.readlines()
prediction_sentences = prediction.readlines()
target_sentences = target.readlines()

num_of_sentences = len(input_sentences)
print(f"Number Of Sentences: {num_of_sentences}")

total_words = 0 
total_correct = 0
total_correct_initially = 0

for sentence in range(1, num_of_sentences+1):
    input_sentence = input_sentences[sentence-1]
    prediction_sentence = prediction_sentences[sentence-1]
    target_sentence = target_sentences[sentence-1]
    print(f"Sentence #{sentence}: Input Score: {lm_model(input_sentence):.2f} | Prediction Score: {lm_model(prediction_sentence):.2f} | Target Score: {lm_model(target_sentence):.2f}" )
    input_words = input_sentence.split()
    pred_words = prediction_sentence.split()
    target_words = target_sentence.split()
    total_words += len(input_words)
    for i in range(len(input_words)):
        total_correct += (1 if pred_words[i] == target_words[i] else 0  )
        total_correct_initially += ( 1 if target_words[i] == input_words[i] else 0)
        
print(f"Initial Accuracy {total_correct_initially/total_words*100} %")
print(f"Algorithm Accuracy {total_correct/total_words*100} %")
 
input.close()
prediction.close()
target.close()