import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


batch_size = 32
block_size = 8
eval_itarate = 200
learning_rate = 1e-2
eval_interval = 300
max_iterations = 3000
n_emb = 32
head_size = 16


text = open("input.txt", 'r').read()

chars = list(sorted(set(text)))

vocab_size = len(chars)

stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for i, s in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[n] for n in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = 9*(len(text) // 10)
train_data = data[:n]
val_data = data[n:]

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.position_embedding_table = nn.Embedding(block_size, n_emb)
        
        self.sa_head = Head(head_size)
        self.lm_head = nn.Linear(head_size, vocab_size)


    def forward(self, idx, targets = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T))
        x = tok_emb + pos_emb
        z = self.sa_head(x)
        logits = self.lm_head(z)
        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_limited = idx[:,-block_size:]
            logits, loss = self(idx_limited)
            logits = logits[:,-1,:]
            probs = torch.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size)
        self.query = nn.Linear(n_emb, head_size)
        self.value = nn.Linear(n_emb, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
    def __call__(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2,-1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))

        wei = F.softmax(wei, -1)
        out = wei @ v
        return out

def get_batch(split):
    split_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(split_data)-block_size, (batch_size,))
    x = torch.stack([split_data[i: i + block_size] for i in ix])
    y = torch.stack([split_data[i+1 : i + block_size +1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss():
    out={}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_itarate)
        for i in range(eval_itarate):
            xb, yb = get_batch(split)
            logits, loss = m(xb,yb)
            losses[i] = loss.item()
            av_loss = losses.mean()
            out[split] = av_loss
    m.train()
    return out


m = BigramLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

print(decode(m.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))



for iteration in range(max_iterations):
    if iteration % eval_itarate == 0:
        losses = estimate_loss()
        print(f"train loss: {losses['train']},  val loss: {losses['val']}")

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context= torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))