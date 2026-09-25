import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn(fan_in, fan_out)/ fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight]+ ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dim, eps = 1e-5, lr = 0.1):
        self.eps = eps
        self.momentum = lr
        self.training = True
        # will be learned
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # will not be learned
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self,x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / (xvar + self.eps)**0.5
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = self.running_mean * (1-self.momentum) + xmean * self.momentum
                self.running_var = self.running_var * (1-self.momentum) + xvar * self.momentum
        return self.out
    
    def parameters(self):
        return [self.gamma] + [self.beta] 

class Tanh():
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out 
    def parameters(self):
        return []




g = torch.Generator().manual_seed(42)

words = open("names.txt", "r").read().splitlines()

chars = sorted(set(''.join(words)))

chars.insert(0, '.')

stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for i, s in enumerate(chars)}

base_num = 3
emb_size = 10

batch_size = 32

X, Y = [],[]

for word in words:
    context = [0 for _ in range(base_num)]
    word =  word + '.'
    
    for ch in word:
        i_s = stoi[ch]
        Y.append(i_s)
        X.append(context)

        context = context[1:] + [i_s]

X = torch.tensor(X)
Y = torch.tensor(Y)

def build_dataset(words):
    X_l, Y_l = [],[]
    for word in words:
        context = [0 for _ in range(base_num)]
        word =  word + '.'
        
        for ch in word:
            i_s = stoi[ch]
            Y_l.append(i_s)
            X_l.append(context)

            context = context[1:] + [i_s]
    return torch.tensor(X_l), torch.tensor(Y_l)

n1 = int(len(words) * 0.8)
n2 = int(len(words) * 0.9)

Xtr, Ytr = build_dataset(words[:n1])
Xdev,Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

n_hidden = 200
max_steps= 200000

C = torch.randn((len(chars), emb_size))
layers = [Linear(emb_size* base_num,n_hidden, bias= False), BatchNorm1d(n_hidden), Tanh(), Linear(n_hidden, len(chars)),]
with torch.no_grad():
    layers[-1].weight *= 0.1

parameters = [C] + [parameter for layer in layers for parameter in layer.parameters()]

for parameter in parameters:
    parameter.requires_grad = True

lossi = []
for i in range(max_steps):
    ix = torch.randint(Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    
    emb = C[Xb]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    
    loss = F.cross_entropy(x, Yb)
    lossi.append(loss.log10().item())

    for p in parameters:
        p.grad = None
    
    loss.backward()

    lr = 0.1 if i < max_steps/2 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    if i % 10000 == 0:
        print(f"{i:7_d}/{max_steps:7d}: {loss.item()}")

    
for layer in layers:
    layer.training = False

def sample_name(C, W1, W2, B2, bngain, bnbias, bn_running_mean, bn_running_std):
    for _ in range(10):
        context = [0] * base_num
        out = []
        while True:
            emb = C[torch.tensor(context)]
            x = emb.view(emb.shape[0], -1)
            for layer in layers:
                x = layer(x)
            probs = F.softmax(x, dim=1)
            ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
            if(ix == 0):
                break
            out.append(itos[ix])
            context = context[1:] + [ix]
        print(''.join(out))

@torch.no_grad()
def split_loss(split):
   x, y = {
      'train': (Xtr, Ytr), 
      'develop' : (Xdev, Ydev),
      'test' : (Xte, Yte)
   }[split]
   emb = C[x]
   x = emb.view(emb.shape[0], -1)
   for layer in layers:
        x = layer(x)

   loss = F.cross_entropy(x, y)
   print(f"{split} ---> {loss.item()}")
print(split_loss('train'))
print(split_loss('develop'))
plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
plt.show()
#3-character MLP: train ≈ 2.018, dev ≈ 2.323