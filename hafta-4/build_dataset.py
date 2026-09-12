import random
import torch

def build_dataset(words):
    X, Y = [], []
    chars = sorted(set(''.join(words)))
    chars.insert(0, '.')
    stoi = {s : i for i, s in enumerate(chars)}
    base_num = 3


    for word in words:
        adjent = [0 for _ in range(base_num)]
   
        word += '.'
        for ch in word:
            ix = stoi[ch]
            X.append(adjent)
            Y.append(ix)
            adjent = adjent[1:] + [ix]
        
    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y
