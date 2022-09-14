input = open("./data/input.txt", 'r')
prediction = open("./data/pred.txt", 'r')
target = open("./data/output.txt", 'r')


input_sentences = input.readlines()
prediction_sentences = prediction.readlines()
target_sentences = target.readlines()

num_of_sentences = len(input_sentences)
print(f"Number Of Sentences: {num_of_sentences}")

total_words = 0 
total_correct = 0
total_correct_initially = 0

for sentence in range(1, num_of_sentences+1):
    input_words = input_sentences[sentence-1].split()
    pred_words = prediction_sentences[sentence-1].split()
    target_words = target_sentences[sentence-1].split()
    total_words += len(input_words)
    for i in range(len(input_words)):
        total_correct += (1 if pred_words[i] == target_words[i] else 0  )
        total_correct_initially += ( 1 if target_words[i] == input_words[i] else 0)
        
print(f"Initial Accuracy {total_correct_initially/total_words*100} %")
print(f"Algorithm Accuracy {total_correct/total_words*100} %")
 
input.close()
prediction.close()
target.close()