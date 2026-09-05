import torch
from bigram import words_tr, stoi, P
log_likelihood = 0.0
n = 0
words_local = ["sena", "seda", "erva"]
for word in words_tr:
    word = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(word, word[1:]):
        i_of_ch1 = stoi[ch1]
        i_of_ch2 = stoi[ch2]
        prob = P[i_of_ch1, i_of_ch2]
        log_prob = torch.log(prob)
        log_likelihood+= log_prob
        n+=1
        print(f"{ch1}{ch2} prob:{prob:.4f} log:{log_prob:.4f}")

print(f"likelihood:{log_likelihood:.4f}") 
llm = -log_likelihood
av_loss = llm/n
print(f"llm:{llm:.4f}")
print(f"{av_loss}")