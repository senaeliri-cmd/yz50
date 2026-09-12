import torch
import matplotlib.pyplot as plt
import build_dataset
import random


words = open('names.txt', 'r').read().splitlines()

chars = sorted(set(''.join(words)))
chars.insert(0, '.')

stoi = {s : i for i, s in enumerate(chars)}
itos = {i : s for i, s in enumerate(chars)}

emb_size = 4
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

C = torch.randn(len(chars), emb_size)

emb = C[Xtr]

g = torch.Generator().manual_seed(2147483647)

#print(embd.shape) 32, 3, emb_size(4)
b = emb_size * base_num
W1 = torch.randn((b,200), generator=g)
B1 = torch.randn((200), generator=g)

a = emb.view(-1, emb_size * base_num)

h = torch.tanh(a @ W1 + B1)

W2= torch.randn((200, len(chars)), generator=g)
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
   h = torch.tanh(emb.view(-1, b) @ W1 + B1) #(32, 200)
   logits = h @ W2 + B2 #(32, 27)

   loss = F.cross_entropy(logits, Ytr[Xi])

   for p in parameteres:
      p.grad = None
   loss.backward()

   #lri.append(lre[i])
   #losses.append(loss.item())
   lr = 0.1 if i < 5000 else 0.01
   for p in parameteres:
      p.data += - lr * p.grad

emb = C[Xtr]
h = torch.tanh(emb.view(-1, b) @ W1 + B1)
logits = h @W2 +B2
loss = F.cross_entropy(logits, Ytr)
print(f"Train Loss---->{loss.item()}")
 
emb = C[Xdev]
h = torch.tanh(emb.view(-1, b) @ W1 + B1)
logits = h @ W2 + B2
loss = F.cross_entropy(logits, Ydev)
print(f"Develop Loss--->{loss.item()}")

#plt.plot(lri, losses)
#plt.show()


for _ in range(10):
   context = [0] * base_num
   out = []
   while True:
      emb = C[context]
      h = torch.tanh(emb.view(-1, b) @ W1 + B1)
      logits = h @ W2 + B2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
      if(ix == 0):break
      out.append(itos[ix])
      context = context[1:] + [ix]
   print(''.join(out))

print("emb:", emb.shape)
print("h:", h.shape)
print("logits:", logits.shape)
print("probs:", probs.shape)
###
#Train Loss---->2.365797519683838
#Develop Loss--->2.3608527183532715
#akka
#tarnita
#mellea
#kawginalely
#avyni
#kanir
#rak
#gabteeniac
#ean
#niskan
##
#
#
#