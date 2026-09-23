"""Decoder-only Transformer in the style of karpathy/nanoGPT.

Differences from experiment 001's hand-rolled GPT:

- one linear map produces query, key, and value, then splits into heads
- the feed-forward network uses GELU
- the token embedding and the output head share one weight matrix

Shapes below: B batch, T time, C embedding width.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Batched causal self-attention.

    Scores are divided by sqrt(head_size) so softmax stays diffuse. Future
    positions are masked with a lower triangle.
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("bias", mask)

    def forward(self, x):
        batch, time, channels = x.size()
        query, key, value = self.c_attn(x).split(channels, dim=2)
        query = query.view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        key = key.view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        value = value.view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        weights = (query @ key.transpose(-2, -1)) * (self.head_size ** -0.5)
        weights = weights.masked_fill(self.bias[:, :, :time, :time] == 0, float("-inf"))
        weights = self.attn_dropout(F.softmax(weights, dim=-1))
        out = weights @ value
        out = out.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_dropout(self.c_proj(out))


class MLP(nn.Module):
    """Position-wise feed-forward network, 4x wider, with GELU."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-norm residual block: attention, then the feed-forward network."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """Character-level GPT with tied token embedding and output head."""

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout_p = dropout

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)
        # Residual projections start smaller so the sum of many blocks stays stable.
        scale = 0.02 / math.sqrt(2 * n_layer)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=scale)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """idx is (B, T). targets, when set, is the next character at each position."""
        _, time = idx.shape
        if time > self.block_size:
            raise ValueError(f"sequence length {time} exceeds block_size {self.block_size}")
        positions = torch.arange(time, device=idx.device)
        x = self.drop(self.token_embedding(idx) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            batch, time, channels = logits.shape
            loss = F.cross_entropy(logits.view(batch * time, channels), targets.view(batch * time))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Append max_new_tokens characters, cropping context to block_size."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def build_model(vocab_size, config):
    return GPTLanguageModel(
        vocab_size=vocab_size,
        block_size=config["block_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        dropout=config["dropout"],
    )
