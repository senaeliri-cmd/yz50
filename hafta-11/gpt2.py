
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_embd = config.n_embd
        self.n_head = config.n_head

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate = "tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class ChatGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_head: int = 12
    n_layer: int = 12
    n_embd: int = 768

class ChatGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)

        # weight share scheme
        self.transformer.wte.weight = self.lm_head.weight

        # fix init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if(hasattr(module, "NANO_GPT_SCALE")):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        assert T <= self.config.block_size

        pos = torch.arange(T, device=idx.device)

        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)

        x = pos_emb + tok_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        from transformers import GPT2LMHeadModel
        print(f"loading weights from {model_type}...")

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = ChatGPTConfig(**config_args)
        model = cls(config)

        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = [k for k in sd_hf.keys()
                      if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf
                      if not k.endswith('.attn.bias')]

        transposed = [
            'attn.c_attn.weight',
            'attn.c_proj.weight',
            'mlp.c_fc.weight',
            'mlp.c_proj.weight'
        ]

        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched key length {len(sd_keys_hf)} != {len(sd_keys)}"
        )

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape

                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape

                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

import tiktoken
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)

        self.current_pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos : self.current_pos + B*T + 1]

        self.current_pos += B * T + 1

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        if(self.current_pos + B* T + 1 > len(self.tokens)):
            self.current_pos = 0
        
        return x, y

# Device
import time

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

torch.manual_seed(42)

train_loader = DataLoaderLite(B=4, T=32)

config = ChatGPTConfig(
    block_size=128,
    n_layer=4,
    n_head=4,
    n_embd=256
)

model = ChatGPT(config).to(device)


x, y = train_loader.next_batch()
x, y = x.to(device), y.to(device)

model.eval()
with torch.no_grad():
    logits, loss = model(x, y)

print(f"Initial loss: {loss.item():.4f}")
print(f"Expected loss: {math.log(50257):.4f}")

# Training
model.train()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)

max_steps = 600
   
import code; code.interact(local=locals())
for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(x, y)
    
    loss.backward()

    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0) * 1000

    

    if step % 50 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")

print("Training complete!")