import torch
import matplotlib.pyplot as plt
import build_dataset
import random


words = open('names.txt', 'r').read().splitlines()

chars = sorted(set(''.join(words)))
chars.insert(0, '.')

stoi = {s : i for i, s in enumerate(chars)}

base_num = 3
X, Y = [], []

for word in words:
   # print(word)
    adjent = [0 for _ in range(base_num)]
   
    word += '.'
    for ch in word:
       # print(f"{adjent} -------> {ch}")
        ix = stoi[ch]
        X.append(adjent)
        Y.append(ix)
        adjent = adjent[1:] + [ix]

random.seed(42)
random.shuffle(words)

n1 = int(len(words) * 0.8)
n2 = int(len(words) * 0.9)

Xtr, Ytr = build_dataset.build_dataset(words[:n1])
Xdev, Ydev = build_dataset.build_dataset(words[n1:n2])
Xte, Yte = build_dataset.build_dataset(words[n2:])

F = torch.nn.functional

XS = torch.tensor(X)
YS = torch.tensor(Y)

C = torch.randn(len(chars), 2)

emb = C[Xtr]

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
Xi = torch.randint(0, Xtr.shape[0], (32,))




for p in parameteres:
   p.requires_grad=True
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre
lri =[]
losses = []

for i in range(10000):
   #lr = lrs[i]
   Xi = torch.randint(0, Xtr.shape[0], (32,))
   emb = C[Xtr[Xi]] # (32, 3, 2)
   h = torch.tanh(emb.view(-1, 6) @ W1 + B1) #(32, 100)
   logits = h @ W2 + B2 #(32, 27)

   loss = F.cross_entropy(logits, Ytr[Xi])

   for p in parameteres:
      p.grad = None
   loss.backward()

   #lri.append(lre[i])
   #losses.append(loss.item())
   lr = 0.01
   for p in parameteres:
      p.data += - lr * p.grad

embd = C[Xtr]
h = torch.tanh(embd.view(-1, 6) @ W1 + B1)
logits = h @W2 +B2
loss = F.cross_entropy(logits, Ytr)
print(f"Train Loss---->{loss.item()}")
 
embd = C[Xdev]
h = torch.tanh(embd.view(-1, 6) @ W1 + B1)
logits = h @ W2 + B2
loss = F.cross_entropy(logits, Ydev)
print(f"Develop Loss--->{loss.item()}")

#plt.plot(lri, losses)
#plt.show()










