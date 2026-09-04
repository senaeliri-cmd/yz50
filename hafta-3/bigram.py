import torch
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()


chars = sorted(set(''.join(words)))
stoi = {s:i +1 for i, s in enumerate(chars)}
stoi["."] = 0


d = torch.zeros((27, 27), dtype=torch.int32)

for word in words:
    word = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(word, word[1:]):
        i_of_ch1 = stoi[ch1]
        i_of_ch2 = stoi[ch2]
        d[i_of_ch1, i_of_ch2] += 1
    
itos = {i:s for s, i in stoi.items()}

#visiuals
plt.figure(figsize=(16, 16))
plt.imshow(d, cmap='Blues')
for i in range(27):
    for j in range(27):
        ch1xch2 = itos[i] + itos[j]
        plt.text(j, i, ch1xch2, ha="center", va = "bottom", color= 'gray')
        plt.text(j, i, d[i, j].item(), ha="center", va = "top",color='black' )

plt.axis('off')
#plt.show()

P = d.float()
P = P/P.sum(1, keepdim=True)

g = torch.Generator().manual_seed(2147483647)

for _ in range(3):
    xi = 0
    out = []
    while True:
        p = P[xi]

        xi = torch.multinomial(p, num_samples=1, replacement= True,generator= g).item()
        out.append(itos[xi])

        if(xi == 0):
            break
    print(''.join(out))
        