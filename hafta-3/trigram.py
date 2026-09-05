import torch
import matplotlib.pyplot as plt
import numpy as np
#print(27*27) 729

words = open('names.txt', 'r').read().splitlines()

chars = list(sorted(set(''.join(words))))
chars.insert(0, '.')
print(f"{len(chars)} {len(words)}")


s = [(c1,c2) for c1 in chars for c2 in chars]
d_itos = {i  : j for i, j in enumerate(s)}

d_stoi = {j : i for i, j in enumerate(s)}

stoi = {j:i  for i, j in enumerate(chars)}

itos = {i : j for i,j in enumerate(chars)}

xxs, yys = [], []
for word in words:
    word = '..' + word +'.'
    for ch1, ch2, ch3 in zip(word, word[1:], word[2:]):
        i1 = stoi[ch1]
        i2 = stoi[ch2]
        i3 = stoi[ch3]
        #print(f"{ch1}{ch2}{ch3}")
        xxs.append(d_stoi[(ch1, ch2)])
        yys.append(i3)

xxs = torch.tensor(xxs)
yys = torch.tensor(yys)
num = xxs.nelement()



g = torch.Generator().manual_seed(2350290)
F = torch.nn.functional

xenc = F.one_hot(xxs, num_classes=729).float()
print(xenc.shape)
W = torch.randn((729, 27), generator=g, requires_grad= True)


for _ in range(40):
    W = torch.load("W_trigram.pt")
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)

    loss = -probs[torch.arange(num), yys].log().mean()
    W.grad = None
    loss.backward()
    
    W.data += -30.0 * W.grad
    torch.save(W, "W_trigram.pt")


for _ in range(10):
    out=[]
    xi = d_stoi[('.', '.')]

    while True:
        p = probs[xi]
        old_s = d_itos[xi]
        ci = torch.multinomial(p, num_samples=1, replacement= True,generator= g).item()
        out.append(itos[ci])
        xi = d_stoi[(old_s[-1], itos[ci])]
        if(ci == 0):
            break
    print(''.join(out))