"""Decoder-only Transformer from the nanoGPT lecture.

Each block is pre-norm: LayerNorm, then a residual of causal self-attention,
then LayerNorm, then a residual of a position-wise MLP. Tokens cannot attend
to future positions, so the model can be trained on every next-character
target in a block at once.

Shapes used below: B batch, T time (context length), C channels (n_embd).
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """One head of causal self-attention.

    Query and key decide how strongly each position attends to earlier
    positions. Value is what gets averaged. Dividing by sqrt(head_size) keeps
    the scores from saturating softmax when the head is wide.
    """

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril[t, i] is 1 when i <= t. Positions with 0 are future tokens.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, time, _ = x.shape
        key = self.key(x)
        query = self.query(x)
        weights = query @ key.transpose(-2, -1) * (key.shape[-1] ** -0.5)  # (B, T, T)
        weights = weights.masked_fill(self.tril[:time, :time] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        value = self.value(x)
        return weights @ value  # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """Several heads in parallel, concatenated, then projected back to n_embd.

    Each head can learn a different notion of "what to look at". The projection
    mixes those head outputs into the residual stream.
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise MLP. The hidden layer is 4x wider, matching the lecture."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One Transformer block: communication, then computation.

    LayerNorm is applied before each sublayer (pre-norm). The residual add
    lets the block keep the previous representation and only add a change.
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """Character-level GPT.

    A token embedding and a learned position embedding are added, passed
    through `n_layer` blocks, then a linear head produces logits over the
    character vocabulary. The head is not tied to the token embedding.
    """

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout_p = dropout

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Small normal init from the lecture repo. The video itself skipped this;
        # without it the same architecture converges more slowly.
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """idx is (B, T) integer token ids. targets, when given, is (B, T) too.

        targets[b, t] is the character that follows idx[b, t], so every
        position in the block contributes one next-character prediction.
        """
        _, time = idx.shape
        token_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(time, device=idx.device))  # (T, C)
        x = self.blocks(token_emb + pos_emb)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab)

        loss = None
        if targets is not None:
            batch, time, channels = logits.shape
            loss = F.cross_entropy(logits.view(batch * time, channels), targets.view(batch * time))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Append max_new_tokens characters.

        The context is cropped to block_size because the position embedding
        table only has that many rows.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
