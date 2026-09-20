import torch
import torch.nn.functional as F

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
n2 = int(len(words) * 0.1)

Xtr, Ytr = build_dataset(words[:n1])
Xdev,Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt -t.grad).abs().max().item()
    print(f"{s:15s} | exact:{str(ex):5s} | approximate:{str(app):5s} | maxdiff:{maxdiff}")
    
n_hidden = 200

C = torch.randn((len(chars), emb_size))

g = torch.Generator().manual_seed(2147483647)

W1 = torch.randn((base_num * emb_size, n_hidden),   generator=g) * 5/3 *(base_num ** 0.5)
B1 = torch.randn(n_hidden,                          generator=g) * 0.1
W2 = torch.randn(n_hidden, len(chars),              generator=g) * 0.1
B2 = torch.randn(len(chars),                        generator=g) * 0.1

bn_gain = torch.randn(1, n_hidden) * 0.1 + 1.0
bn_bias = torch.randn(1, n_hidden) * 0.1

parameteres = [W1, B1, W2, B2, C, bn_gain, bn_bias]

for p in parameteres:
    p.requires_grad = True

bn_runningmean = torch.ones(1, n_hidden)
bn_runningbias = torch.zeros(1, n_hidden)


ix = torch.randint(0, Xtr.shape[0], (batch_size, ), generator=g)
X_minib, Y_minib = Xtr[ix], Ytr[ix]

emb = C[X_minib]
#emb shape: ([32, 3, 10]), C: ([27, 10]) X_minib: ([32, 3])
embcat = emb.view(emb.shape[0], -1) 

hprebn = embcat @ W1 + B1 # embcat size(32, 30) hprebn size (32,200)

bnmean = (1/batch_size) * (hprebn.sum(0, keepdim=True)) # bnmean size(1,200)
bndiff = hprebn -bnmean # bndiff size(32,200)
bndiff2 = bndiff ** 2 #bndiff2 size (32, 200)
bnvar = 1/(batch_size-1)*(bndiff2).sum(0, keepdim=True) # bnvar size (1, 200)
bnvar_inv =(bnvar + 1e-5)**-0.5 # bnvar_inv size (1, 200)
bnraw =bndiff *bnvar_inv
hpreact = bn_gain * bnraw + bn_bias

h = torch.tanh(hpreact)

logits = h @ W2 + B2

logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdim=True)
counts_sum_inv = counts_sum**-1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(batch_size), Y_minib].mean()

for p in parameteres:
    p.grad = None
for t in [logprobs, probs, counts_sum_inv, counts_sum, counts, norm_logits, logit_maxes, h, hpreact, hprebn, bnraw, bndiff, bndiff2,
             embcat, emb, bnvar, bnmean, bnvar_inv, logits]:
    t.retain_grad()
loss.backward()


dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(batch_size), Y_minib] = -1.0/(batch_size)
dprobs = (1.0/probs)*dlogprobs
dcounts = counts_sum_inv * dprobs
dcounts_sum_inv = (dprobs * counts).sum(1, keepdim=True)
dcounts_sum = dcounts_sum_inv * (-counts_sum**-2)
dcounts = torch.ones_like(counts)* dcounts_sum + dprobs * counts_sum_inv
dnorm_logits = dcounts * counts
#dlogits = dnorm_logits.clone()
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
#dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1])*dlogit_maxes
dlogits = F.softmax(logits, 1)
dlogits[range(batch_size), Y_minib] -=1
dlogits/= batch_size
dlogits = (probs - F.one_hot(Y_minib, num_classes=probs.shape[1])) / batch_size
dB2 = dlogits.sum(0, keepdim=True)
dW2 = h.T@dlogits
dh = dlogits@W2.T
dhpreact = dh * (1-h**2)
dbn_gain = (bnraw * dhpreact).sum(0)
dbn_bias = dhpreact.sum(0)
dbnraw = (dhpreact * bn_gain)
dbndiff = (dbnraw * bnvar_inv)
dbnvar_inv = (dbnraw * bndiff).sum(0, keepdim=True)
dbnvar = ((dbnvar_inv) * -0.5 * (bnvar + 1e-5)**-1.5)
dbndiff2 = dbnvar * (batch_size-1)**-1.0 * torch.ones_like(bndiff2)
dbndiff += dbndiff2 *2*bndiff
dhprebn = dbndiff.clone()
dbnmean = -dbndiff.clone().sum(0, keepdim=True)
dhprebn += (batch_size**-1)* torch.ones_like(hprebn) *dbnmean
dembcat = dhprebn@W1.T
dW1 = embcat.T@dhprebn
dB1 = dhprebn.clone().sum(0, keepdim=True)
demb = dembcat.view(emb.shape)
dC = torch.zeros_like(C)
for k in range(X_minib.shape[0]):
    for s in range(X_minib.shape[1]):
        ix = X_minib[k,s]
        dC[ix] += demb[k, s]



cmp("logprobs", dlogprobs, logprobs)
cmp("dprobs", dprobs, probs)
cmp("dcounts_sum", dcounts_sum, counts_sum)
cmp("dcounts", dcounts, counts)
cmp("dnorm_logits", dnorm_logits, norm_logits)
cmp("dlogits", dlogits, logits)
cmp("dlogit_maxes", dlogit_maxes, logit_maxes)
cmp("B2", dB2, B2)
cmp("W2", dW2, W2)
cmp("h", dh, h)
cmp("dhpreact", dhpreact, hpreact)
cmp("dbn_gain", dbn_gain, bn_gain)
cmp("dbn_bias", dbn_bias, bn_bias)
cmp("dbnraw", dbnraw, bnraw)
cmp("dbndiff", dbndiff, bndiff)
cmp("dbndiff2", dbndiff2, bndiff2)
cmp("dbnvar_inv", dbnvar_inv, bnvar_inv)
cmp("dbnvar", dbnvar, bnvar)
cmp("dhprebn", dhprebn, hprebn)
cmp("dbnmean", dbnmean, bnmean)
cmp("dembcat", dembcat, embcat)
cmp("dW1", dW1, W1)
cmp("dB1", dB1, B1)
cmp("demb", demb, emb)
cmp("dC", dC, C)
cmp("dlogits", dlogits,logits)
