import torch
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()

chars = sorted(set(''.join(words)))
chars.insert(0, '.')

stoi = {s : i for i, s in enumerate(chars)}

base_num = 3
X, Y = [], []

for word in words[:5]:
   # print(word)
    adjent = [0 for _ in range(base_num)]
   
    word += '.'
    for ch in word:
       # print(f"{adjent} -------> {ch}")
        ix = stoi[ch]
        X.append(adjent)
        Y.append(ix)
        adjent = adjent[1:] + [ix]

F = torch.nn.functional

XS = torch.tensor(X)
YS = torch.tensor(Y)

C = torch.randn(len(chars), 2)

emb = C[XS]

g = torch.Generator().manual_seed(2147483647)

#print(embd.shape) 32, 3, 2
W1 = torch.randn((6,100), generator=g)
B1 = torch.randn((100), generator=g)

a = emb.view(-1, 6)

h = torch.tanh(a @ W1 + B1)

W2= torch.randn((100, len(chars)), generator=g)
B2 = torch.randn((len(chars)), generator=g)

parameteres = [C, W1, B1, W2, B2]

#Old method
#logits = h @ W2 + B2
#count = logits.exp()
#prob = count / count.sum(1, keepdim=True)
#loss = -prob[torch.arange(32), YS].log().mean()

#New Method
logits = h @ W2 + B2


loss = F.cross_entropy(logits, YS)
print(loss.item())




