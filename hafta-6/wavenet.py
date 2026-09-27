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
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0,1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
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

class Embedding():
    def __init__(self, emb_size, emb_dim):
        self.weights = torch.randn(emb_size, emb_dim)
    
    def __call__(self, iX):
        self.out = self.weights[iX]
        return self.out
    
    def parameters(self):
        return [self.weights]

class FlattenConsecutive():
    def __init__(self, n_context):
        self.n = n_context

    def __call__(self, x):
        B, C, V = x.shape
        self.out = x.view(B, C//self.n, V * self.n)
        if self.out.shape[1] == 1:
            self.out = self.out.squeeze(1)
        return self.out

    def parameters(self):
        return []

class Flatten():
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out
    def parameters(self):
        return []
class Sequential():
    def __init__(self, x):
        self.layers = x
    def __call__(self, x):
        self.out = x
        for layer in self.layers:
            self.out = layer(self.out)
        return self.out

    def parameters(self):
        self.parameters = [p for layer in self.layers for p in self.parameters()]
        return self.parameters



g = torch.Generator().manual_seed(42)

words = open("names_tr.txt", "r").read().splitlines()

chars = sorted(set(''.join(words)))

chars.insert(0, '.')

stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for i, s in enumerate(chars)}

base_num = 8
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

n_hidden = 68
max_steps= 200000

model = Sequential([Embedding(len(chars), emb_size),
                    FlattenConsecutive(2), Linear(emb_size * 2, n_hidden, bias= False), BatchNorm1d(n_hidden), Tanh(), 
                    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias= False), BatchNorm1d(n_hidden), Tanh(), 
                    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias= False), BatchNorm1d(n_hidden), Tanh(), 
                    Linear(n_hidden , len(chars)),]) 

with torch.no_grad():
    model.layers[-1].weight *= 0.1

parameters = [parameter for layer in model.layers for parameter in layer.parameters()]
print(sum(p.nelement() for p in parameters))

for parameter in parameters:
    parameter.requires_grad = True

lossi = []
for i in range(max_steps):
    ix = torch.randint(Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    x = Xb
    for layer in model.layers:
        x = layer(x)
        
    
    loss = F.cross_entropy(x, Yb)
    lossi.append(loss.log10().item())

    for p in parameters:
        p.grad = None
    
    loss.backward()

    lr = 0.1 if i < max_steps/2 else 0.01
    for p in parameters:
        p.data += -lr * p.grad
    
    #if i % 10000 == 0:
       # print(f"{i:7_d}/{max_steps:7d}: {loss.item()}")

    
for layer in model.layers:
    layer.training = False

def sample_name():
    for _ in range(10):
        context = [0] * base_num
        out = []
        while True:
            x = torch.tensor(context).view(1, -1)
            for layer in model.layers:
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

   for layer in model.layers:
        x = layer(x)

   loss = F.cross_entropy(x, y)
   print(f"{split} ---> {loss.item()}")
print(split_loss('train'))
print(split_loss('develop'))
sample_name()
#plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))
#plt.show()

# hafta-4 3-character MLP:                   train ≈ 2.021, dev ≈ 2.329
# 3-character MLP :                          train ≈ 2.018, dev ≈ 2.323 parameters: 12097
# 8-character MLP:                           train ≈ 1.878, dev ≈ 2.245 parameters: 22097
# 8-character wavenet:                       train ≈ 1.882, dev ≈ 2.251 
# 8-character fixed batchnorm1d:             train ≈ 1.875, dev ≈ 2.238 parameters: 22397
# 8-character 24 n_emb 128 n_hidden wavenet: train ≈ 1.743, dev ≈ 2.226

# Embedding : (32, 8, 10)
# FlattenConsecutive : (32, 4, 20)
# Linear : (32, 4, 68)
# BatchNorm1d : (32, 4, 68)
# Tanh : (32, 4, 68)
# FlattenConsecutive : (32, 2, 136)
# Linear : (32, 2, 68)
# BatchNorm1d : (32, 2, 68)
# Tanh : (32, 2, 68)
# FlattenConsecutive : (32, 136)
# Linear : (32, 68)
# BatchNorm1d : (32, 68)
# Tanh : (32, 68)
# Linear : (32, 27)
# Amacımız tüm önceki karakterlerden gelen bilgileri tek seferde sıkıştırmak yerine,
# kademeli olarak ikişer ardışık karakterin bilgisini birleştirerek ilerlemek.
# Bu sayede receptive field her katmanda büyüyor (2 -> 4 -> 8) ve daha uzak
# geçmişteki karakterlerden gelen bilgileri daha az katmanla modele dahil edebiliyoruz.


# BatchNorm1d ilk yazıldığında girdisi Flatten ile 2D bir tensöre dönüştürülüyordu.
# Ancak FlattenConsecutive kullanmaya başladıktan sonra BatchNorm1d'ye
# (batch, context, feature) şeklinde 3D tensorlar da gelmeye başladı.
#
# Örneğin ilk BatchNorm'a (32, 4, 68) boyutunda bir tensor geliyor:
# 32 -> batch size
# 4  -> context pozisyonu
# 68 -> neuron/feature sayısı
#
# BatchNorm her bir neuron için batch ve context boyunca ortalama almalıdır.
# Yani her neuron için 32 * 4 = 128 değerden tek bir mean ve variance hesaplanmalıdır.
#
# Sadece dim=0 üzerinden ortalama aldığımızda context dimensionı korunuyordu.
# Bu yüzden her neuron için tek bir ortalama yerine 4 farklı ortalama elde ediyorduk.
#
# Bunu düzeltmek için input 2D ise dim=0,
# input 3D ise dim=(0, 1) üzerinden mean ve variance hesapladık.
# Böylece her neuron/feature için tek bir mean ve variance elde edilmiş oldu.
#
# Before fix: dev loss ≈ 2.245
# After fix:  dev loss ≈ 2.238

model_name = ["3 MLP", "8 düz MLP", "8 WaveNet"]

loss_model = [2.323,  2.245,  2.238]
p_model = [12097, 22097, 22397]

table_data = [[model_name[i], p_model[i], loss_model[i],] for i in range(len(model_name))]


fig, ax = plt.subplots()

ax.axis("off")
ax.table(
    cellText=table_data,
    colLabels=["Model", "Parameter Count", "Dev Loss"],
    loc="center"
)
#
#plt.show()

# Turkish Model Comparison
#
# Week 4 - 3-context MLP:      dev loss ≈ 2.329
# Week 6 - 8-context WaveNet:  dev loss ≈ 2.312
#
# Sample generated names:
# erminer
# gülbey
# akköl
# gülser
# alpay
# cavsol
# demirdoğa
# nazlı
# pesuzan
# uçe
#
# Türkçe modelde context uzunluğunu 3'ten 8'e çıkarıp WaveNet yapısını
# kullandığımızda dev loss 2.329'dan 2.312'ye düştü.
# Bu yaklaşık 0.017'lik küçük bir iyileşme sağladı.
# Daha uzun context sayesinde model, bir sonraki karakteri tahmin ederken
# yalnızca son 3 karakter yerine son 8 karaktere kadar olan geçmiş bilgiyi
# kullanabildi. Böylece özellikle daha uzun isimlerdeki karakter örüntülerini
# ve bağımlılıklarını daha iyi modelleyebildi.