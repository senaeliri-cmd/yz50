import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


batch_size = 64
block_size = 256
eval_itarate = 200
learning_rate = 3e-4
eval_interval = 500
max_iterations = 5000
n_emb = 384
head_size = 16
dropout = 0.2
device = "cuda"


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
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb, device=device)
        self.position_embedding_table = nn.Embedding(block_size, n_emb, device=device)
        
        self.blocks = nn.Sequential(
            Block(n_head=6, n_emb=n_emb),
            Block(n_emb=n_emb, n_head=6),
            Block(n_emb=n_emb, n_head=6),
            Block(n_emb=n_emb, n_head=6),
            Block(n_emb=n_emb, n_head=6),
            Block(n_emb=n_emb, n_head=6),
            nn.LayerNorm(n_emb))
        self.lm_head = nn.Linear(n_emb, vocab_size, device=device)


    def forward(self, idx, targets = None):
        idx = idx.to(device)
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T,device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        idx = idx.to(self.token_embedding_table.weight.device)
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
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2,-1) * C ** -head_size
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))

        wei = F.softmax(wei, -1)
        out = wei @ v
        return out

class MultiHead(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.project = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.project(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.set = nn.Sequential(nn.Linear(n_emb, 6 * n_emb), 
                            nn.ReLU(),
                            nn.Linear(n_emb *6 , n_emb),
                            nn.Dropout(dropout))
    def forward(self, x):
        x = self.set(x)
        return x

class Block(nn.Module):
    def __init__(self, n_emb, n_head):
        super().__init__()
        head_size = n_emb // n_head

        self.sa = MultiHead(n_head, head_size)
        self.ffwd = FeedForward(n_emb)

        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


def get_batch(split):
    split_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(split_data)-block_size, (batch_size,))
    x = torch.stack([split_data[i: i + block_size] for i in ix])
    y = torch.stack([split_data[i+1 : i + block_size +1] for i in ix])
    x = x.to(device)
    y = y.to(device)
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


model = BigramLanguageModel(vocab_size)
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
print(decode(m.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))



for iteration in range(max_iterations):
    if iteration % eval_interval == 0:
        losses = estimate_loss()
        print(f"train loss: {losses['train']},  val loss: {losses['val']}")

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))