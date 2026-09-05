import torch
from bigram import words, stoi
xs, ys = [], []

for word in words:
    word = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(word, word[1:]):
        i_of_ch1 = stoi[ch1]
        i_of_ch2 = stoi[ch2]

        xs.append(i_of_ch1)
        ys.append(i_of_ch2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement() 
print(num)
print(len(words)) 
    

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=g, requires_grad= True)
F = torch.nn.functional

xenc = F.one_hot(xs, num_classes=27).float()


for k in range(200):
    logits = xenc @ W
    its = logits.exp()
    probs = its / its.sum(1, keepdim=True)

    loss = -probs[torch.arange(num), ys].log().mean()
    print(loss.item())
    W.grad = None
    loss.backward()
    W.data += -20.0 * W.grad


