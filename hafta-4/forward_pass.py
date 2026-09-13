import torch
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F


words = open('names_tr_augmented.txt', 'r').read().splitlines()

chars = sorted(set(''.join(words)))
chars.insert(0, '.')

stoi = {s : i for i, s in enumerate(chars)}
itos = {i : s for i, s in enumerate(chars)}

emb_size = 10
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

def build_dataset(words):
    X, Y = [], []
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


Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])



XS = torch.tensor(X)
YS = torch.tensor(Y)

C = torch.randn(len(chars), emb_size)

emb = C[Xtr]

g = torch.Generator().manual_seed(2147483647)

#print(embd.shape) 32, 3, emb_size(4)
b = emb_size * base_num
n_hidden = 200

a = emb.view(-1, emb_size * base_num)


W1 = torch.randn((b,n_hidden), generator=g) * (5/3) / (b**0.5)
#B1 = torch.randn((n_hidden), generator=g) * 0.1
W2= torch.randn((n_hidden, len(chars)), generator=g) * 0.01
B2 = torch.randn((len(chars)), generator=g) * 0

bngain = torch.ones(1, n_hidden)
bnbias = torch.zeros(1, n_hidden)
bn_running_mean = torch.zeros(1, n_hidden)
bn_running_std = torch.ones(1, n_hidden)

parameteres = [C, W1, W2, B2, bngain, bnbias]

for p in parameteres:
   p.requires_grad=True

#Old method
#logits = h @ W2 + B2
#count = logits.exp()
#prob = count / count.sum(1, keepdim=True)
#loss = -prob[torch.arange(32), YS].log().mean()

#New Method
batch_size = 32



lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre
lri =[]
losses = []



def train_dataset():
   global bn_running_mean, bn_running_std
   for i in range(200000):
      #lr = lrs[i]
      Xi = torch.randint(0, Xtr.shape[0], (batch_size,))
      emb = C[Xtr[Xi]] # (32, 3, 2)
      hpreact = emb.view(-1, b) @ W1
      bnmean = hpreact.mean(0, keepdim=True)
      bnstd = hpreact.std(0, keepdim=True)

      with torch.no_grad():
         bn_running_mean = 0.999 *bn_running_mean + 0.001* bnmean
         bn_running_std = 0.999 * bn_running_std + 0.001 * bnstd

      hpreact = (bngain * (hpreact - bnmean) / (bnstd)) + (bnbias)
      h = torch.tanh(hpreact) #(32, 200)
      logits = h @ W2 + B2 #(32, 27)

      loss = F.cross_entropy(logits, Ytr[Xi])

      for p in parameteres:
         p.grad = None
      loss.backward()

      #lri.append(lre[i])
      #losses.append(loss.item())
      lr = 0.1 if i < 100000 else 0.01
      for p in parameteres:
         p.data += - lr * p.grad
   checkpoints = {
   'C': C,
   'W1' : W1,
   'W2' : W2,
   'B2' : B2,
   'bngain' : bngain,
   'bnbias' :bnbias,
   'bn_running_mean' : bn_running_mean,
   'bn_running_std' : bn_running_std
   }
   torch.save(checkpoints, "model.pt")



def main():
   checkpoints = torch.load("model.pt")
   sample_name(checkpoints['C'],checkpoints['W1'],checkpoints['W2'],checkpoints['B2'],checkpoints['bngain'],checkpoints['bnbias'], 
   checkpoints['bn_running_mean'], checkpoints['bn_running_std'])
   split_loss('train', checkpoints['C'],checkpoints['W1'],checkpoints['W2'],checkpoints['B2'],checkpoints['bngain'],checkpoints['bnbias'], 
   checkpoints['bn_running_mean'], checkpoints['bn_running_std'])
   split_loss('develop', checkpoints['C'],checkpoints['W1'],checkpoints['W2'],checkpoints['B2'],checkpoints['bngain'],checkpoints['bnbias'], 
   checkpoints['bn_running_mean'], checkpoints['bn_running_std'])

#plt.figure(figsize=(20,10))
#plt.imshow(h.abs() > 0.99, cmap='gray', interpolation='nearest')

#plt.hist(h.view(-1).tolist(), 50)
#plt.plot(lri, losses)
#plt.show()

def sample_name(C, W1, W2, B2, bngain, bnbias, bn_running_mean, bn_running_std):
   for _ in range(10):
      context = [0] * base_num
      out = []
      while True:
         emb = C[context]
         hpreact = emb.view(-1, b)@ W1
         hpreact= bngain * (hpreact-bn_running_mean) / bn_running_std +bnbias
         h = torch.tanh(hpreact)
         logits = h @ W2 + B2
         probs = F.softmax(logits, dim=1)
         ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
         if(ix == 0):break
         out.append(itos[ix])
         context = context[1:] + [ix]
      print(''.join(out))

@torch.no_grad()
def split_loss(split, C, W1, W2, B2, bngain, bnbias, bn_running_mean, bn_running_std):
   x, y = {
      'train': (Xtr, Ytr), 
      'develop' : (Xdev, Ydev),
      'test' : (Xte, Yte)
   }[split]
   emb = C[x]
   hpreact = emb.view(-1, b) @ W1
   hpreact = bngain * (hpreact- bn_running_mean)/bn_running_std + bnbias
   h = torch.tanh(hpreact)
   
   logits = h @ W2 + B2
   loss = F.cross_entropy(logits, y)
   print(f"{split} ---> {loss.item()}")

#train_dataset()
main()

# Bigram Loss  -> 2.499

# Baseline - 10,000 iterations
# Train Loss   -> 2.3658
# Dev Loss     -> 2.3609

# Baseline - 200,000 iterations
# Train Loss   -> 2.1123
# Dev Loss     -> 2.1563

# Initial loss before output-layer initialization fix
# Initial Loss -> 21.0983

# Initial loss after output-layer initialization fix
# Initial Loss -> 3.8490

# After fixing output-layer initialization
# Train Loss   -> 2.0738
# Dev Loss     -> 2.1384

# After reducing tanh saturation with scaled W1 initialization
# Train Loss   -> 2.0390
# Dev Loss     -> 2.1054

# After batch-nom
# Train Loss   -> 1.9278
# Dev Loss     -> 2.0260
