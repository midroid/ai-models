"""Decoder-only Transformer in the style of karpathy/minGPT.

The library this follows is mingpt/model.py: a token embedding, a learned
position embedding, pre-norm blocks, and an untied output head. Attention is
the explicit matmul with a causal mask, and the feed-forward network uses the
tanh approximation of GELU.

Shapes below: B batch, T time, C embedding width.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


# Names follow upstream mingpt. The smoke uses gpt-micro. gpt-mini is the
# chargpt default and is not the recorded run.
MODEL_TYPES = {
    "gpt-nano": dict(n_layer=3, n_head=3, n_embd=48),
    "gpt-micro": dict(n_layer=4, n_head=4, n_embd=128),
    "gpt-mini": dict(n_layer=6, n_head=6, n_embd=192),
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
}


class NewGELU(nn.Module):
    """Tanh approximation of GELU used by GPT-2.

    https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """Batched causal self-attention with an explicit score matrix.

    One linear map produces query, key, and value. Scores are divided by
    sqrt(head size). A lower-triangular buffer blocks future positions.
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("bias", mask)

    def forward(self, x):
        batch, time, channels = x.size()
        query, key, value = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = channels // self.n_head
        query = query.view(batch, time, self.n_head, head_size).transpose(1, 2)
        key = key.view(batch, time, self.n_head, head_size).transpose(1, 2)
        value = value.view(batch, time, self.n_head, head_size).transpose(1, 2)
        weights = (query @ key.transpose(-2, -1)) * (head_size ** -0.5)
        weights = weights.masked_fill(self.bias[:, :, :time, :time] == 0, float("-inf"))
        weights = self.attn_dropout(F.softmax(weights, dim=-1))
        out = weights @ value
        out = out.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_dropout(self.c_proj(out))


class Block(nn.Module):
    """Pre-norm residual block: attention, then the feed-forward network."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(n_embd, 4 * n_embd),
                c_proj=nn.Linear(4 * n_embd, n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(dropout),
            )
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        mlp = self.mlp
        x = x + mlp.dropout(mlp.c_proj(mlp.act(mlp.c_fc(self.ln_2(x)))))
        return x


class GPT(nn.Module):
    """minGPT language model. The output head does not share the token embedding."""

    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, n_embd),
                wpe=nn.Embedding(block_size, n_embd),
                drop=nn.Dropout(dropout),
                h=nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]),
                ln_f=nn.LayerNorm(n_embd),
            )
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def transformer_parameters(self):
        """Parameter count upstream prints. The untied lm_head is excluded."""
        return sum(p.numel() for p in self.transformer.parameters())

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(self, learning_rate, weight_decay, betas):
        """AdamW. Weight decay applies to Linear weights only.

        Biases, LayerNorm scales, and embedding tables are not decayed.
        """
        decay = set()
        no_decay = set()
        whitelist = (nn.Linear,)
        blacklist = (nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, _param in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                if param_name.endswith("bias"):
                    no_decay.add(full_name)
                elif param_name.endswith("weight") and isinstance(module, whitelist):
                    decay.add(full_name)
                elif param_name.endswith("weight") and isinstance(module, blacklist):
                    no_decay.add(full_name)
        param_dict = dict(self.named_parameters())
        missing = param_dict.keys() - (decay | no_decay)
        overlap = decay & no_decay
        if overlap:
            raise RuntimeError(f"parameters in both decay groups: {overlap}")
        if missing:
            raise RuntimeError(f"parameters missing from decay groups: {missing}")
        groups = [
            {"params": [param_dict[name] for name in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[name] for name in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)

    def forward(self, idx, targets=None):
        _batch, time = idx.size()
        if time > self.block_size:
            raise ValueError(f"sequence length {time} exceeds block size {self.block_size}")
        positions = torch.arange(0, time, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(positions))
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Append max_new_tokens sampled characters. idx is (B, T)."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def build_model(vocab_size, config):
    """Build a GPT from a preset dict. `model_type` selects depth and width."""
    spec = MODEL_TYPES[config["model_type"]]
    dropout = config.get("dropout", 0.1)
    return GPT(
        vocab_size=vocab_size,
        block_size=config["block_size"],
        n_layer=spec["n_layer"],
        n_head=spec["n_head"],
        n_embd=spec["n_embd"],
        dropout=dropout,
    )
