"""Decoder-only Hindi-Sanskrit model.

Grouped-query attention, SwiGLU, RMSNorm with a learned scale, and rotary
positions. The rotary helper matches experiment 008: base 100_000 and the
nanochat pair rotation. The input embedding and the output head share one
weight matrix.

Shapes: B batch, T time, C width, H query heads, K key-value heads, D head dim.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


MODEL_10M = {
    "vocab_size": 12000,
    "n_embd": 256,
    "n_layer": 12,
    "n_head": 8,
    "n_kv_head": 2,
    "head_dim": 32,
    "intermediate": 640,
    "block_size": 1024,
    "dropout": 0.05,
    "rope_theta": 100_000,
    "rms_eps": 1e-5,
    "tie_embeddings": True,
}

PARAMETER_COUNT = 10_942_720


def count_parameters(config):
    """Trainable weights for the tied decoder. RoPE tables are buffers."""
    vocab = config["vocab_size"]
    width = config["n_embd"]
    layers = config["n_layer"]
    query = config["n_head"] * config["head_dim"]
    key_value = config["n_kv_head"] * config["head_dim"]
    intermediate = config["intermediate"]
    embed = vocab * width
    attention = width * query + 2 * width * key_value + query * width
    feed_forward = 3 * width * intermediate
    norms = 2 * width
    return embed + layers * (attention + feed_forward + norms) + width


def rms_norm(x, eps=1e-6):
    """Root-mean-square norm along the last dimension. No learned scale."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps)


def apply_rotary_emb(x, cos, sin):
    """Rotate the last dimension in pairs.

    `cos` and `sin` broadcast over (B, T, heads, width/2). The sign matches
    nanochat: the rotation is the transpose of the textbook one, and only the
    relative rotation between query and key matters.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1)


def build_rope(seq_len, head_dim, base=100_000):
    """Cos and sin tables of shape (1, seq_len, 1, head_dim/2)."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be even for rotary positions")
    channel = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (channel / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos()[None, :, None, :]
    sin = freqs.sin()[None, :, None, :]
    return cos, sin


def attend(query, key, value):
    """query (B, Tq, H, D), key (B, Tk, H, D), value (B, Tk, H, D)."""
    query_t = query.transpose(1, 2)
    key_t = key.transpose(1, 2)
    value_t = value.transpose(1, 2)
    if query.size(1) == key.size(1):
        out = F.scaled_dot_product_attention(query_t, key_t, value_t, is_causal=True)
    elif query.size(1) == 1:
        out = F.scaled_dot_product_attention(query_t, key_t, value_t, is_causal=False)
    else:
        raise RuntimeError("cached attention expects one new token or a full causal block")
    return out.transpose(1, 2).contiguous()


def repeat_kv(states, n_repeat):
    """(B, T, K, D) -> (B, T, K * n_repeat, D)."""
    if n_repeat == 1:
        return states
    return states.repeat_interleave(n_repeat, dim=2)


def unique_parameters(module):
    """Parameters once each. The tied head shares storage with the embedding."""
    seen = set()
    unique = []
    for param in module.parameters():
        if not param.requires_grad:
            continue
        pointer = param.data_ptr()
        if pointer in seen:
            continue
        seen.add(pointer)
        unique.append(param)
    return unique


class RMSNorm(nn.Module):
    """rms_norm from experiment 008, multiplied by a learned scale."""

    def __init__(self, width, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return rms_norm(x, self.eps) * self.weight


class KVCache:
    """Keys and values for the key-value heads, written at an absolute position."""

    def __init__(self, n_layer, batch, max_seq, n_kv_head, head_dim, device, dtype):
        self.pos = 0
        self.k = torch.zeros(n_layer, batch, max_seq, n_kv_head, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros(n_layer, batch, max_seq, n_kv_head, head_dim, device=device, dtype=dtype)


class GroupedQueryAttention(nn.Module):
    """Separate Q, K, and V. K and V are the narrower grouped set."""

    def __init__(self, n_embd, n_head, n_kv_head, head_dim, layer_idx, dropout):
        super().__init__()
        if n_head % n_kv_head != 0:
            raise ValueError(f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})")
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.n_repeat = n_head // n_kv_head
        self.q_proj = nn.Linear(n_embd, n_head * head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.o_proj = nn.Linear(n_head * head_dim, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)
        self.last_q_shape = None
        self.last_k_shape = None

    def _shape(self, projected, batch, time, heads):
        return projected.view(batch, time, heads, self.head_dim)

    def _rope(self, states, positions, cos, sin):
        return apply_rotary_emb(states, cos[:, positions], sin[:, positions])

    def _project(self, x, positions, cos, sin):
        batch, time, _ = x.shape
        query = self._rope(self._shape(self.q_proj(x), batch, time, self.n_head), positions, cos, sin)
        key = self._rope(self._shape(self.k_proj(x), batch, time, self.n_kv_head), positions, cos, sin)
        value = self._shape(self.v_proj(x), batch, time, self.n_kv_head)
        self.last_q_shape = tuple(query.shape)
        self.last_k_shape = tuple(key.shape)
        return query, key, value

    def _merge(self, heads):
        batch, time, _, _ = heads.shape
        return self.drop(self.o_proj(heads.reshape(batch, time, -1)))

    def forward(self, x, cos, sin):
        time = x.size(1)
        query, key, value = self._project(x, slice(0, time), cos, sin)
        key = repeat_kv(key, self.n_repeat)
        value = repeat_kv(value, self.n_repeat)
        return self._merge(attend(query, key, value))

    def forward_cached(self, x, cache, pos, cos, sin):
        batch, time, _ = x.shape
        query, key, value = self._project(x, slice(pos, pos + time), cos, sin)
        cache.k[self.layer_idx, :, pos : pos + time] = key
        cache.v[self.layer_idx, :, pos : pos + time] = value
        end = pos + time
        key = repeat_kv(cache.k[self.layer_idx, :, :end], self.n_repeat)
        value = repeat_kv(cache.v[self.layer_idx, :, :end], self.n_repeat)
        return self._merge(attend(query, key, value))


class SwiGLU(nn.Module):
    """Three-matrix feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, n_embd, intermediate, dropout):
        super().__init__()
        self.gate = nn.Linear(n_embd, intermediate, bias=False)
        self.up = nn.Linear(n_embd, intermediate, bias=False)
        self.down = nn.Linear(intermediate, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    """Pre-norm residual block."""

    def __init__(self, n_embd, n_head, n_kv_head, head_dim, intermediate, layer_idx, dropout, eps):
        super().__init__()
        self.norm1 = RMSNorm(n_embd, eps)
        self.attn = GroupedQueryAttention(n_embd, n_head, n_kv_head, head_dim, layer_idx, dropout)
        self.norm2 = RMSNorm(n_embd, eps)
        self.mlp = SwiGLU(n_embd, intermediate, dropout)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))

    def forward_cached(self, x, cache, pos, cos, sin):
        x = x + self.attn.forward_cached(self.norm1(x), cache, pos, cos, sin)
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    """Tied decoder. Labels use -100 for positions that must not contribute loss."""

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        vocab = config["vocab_size"]
        width = config["n_embd"]
        n_head = config["n_head"]
        head_dim = config["head_dim"]
        if width != n_head * head_dim:
            raise ValueError(f"n_embd ({width}) must equal n_head * head_dim ({n_head * head_dim})")
        self.vocab_size = vocab
        self.block_size = config["block_size"]
        self.n_layer = config["n_layer"]
        self.n_head = n_head
        self.n_kv_head = config["n_kv_head"]
        self.head_dim = head_dim
        self.wte = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList(
            [
                Block(
                    width,
                    n_head,
                    config["n_kv_head"],
                    head_dim,
                    config["intermediate"],
                    layer_idx,
                    config["dropout"],
                    config["rms_eps"],
                )
                for layer_idx in range(config["n_layer"])
            ]
        )
        self.norm_f = RMSNorm(width, config["rms_eps"])
        self.lm_head = nn.Linear(width, vocab, bias=False)
        self.tie_weights()
        cos, sin = build_rope(self.block_size, head_dim, base=config["rope_theta"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init_weights)
        scale = 0.02 / math.sqrt(2 * config["n_layer"])
        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(param, mean=0.0, std=scale)
        self.tie_weights()

    def tie_weights(self):
        if self.config.get("tie_embeddings", True):
            self.lm_head.weight = self.wte.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self):
        return sum(param.numel() for param in unique_parameters(self))

    def kv_bytes_per_token(self):
        """Bytes of KV cache for one token, across every layer, in float32."""
        return self.n_layer * 2 * self.n_kv_head * self.head_dim * 4

    def _logits(self, x):
        return self.lm_head(self.norm_f(x))

    def forward(self, idx, labels=None):
        """Full sequence. labels, when set, align with idx and may contain -100."""
        _, time = idx.shape
        if time > self.block_size:
            raise ValueError(f"sequence length {time} exceeds block_size {self.block_size}")
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        logits = self._logits(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    def new_cache(self, batch, device, dtype):
        return KVCache(
            n_layer=self.n_layer,
            batch=batch,
            max_seq=self.block_size,
            n_kv_head=self.n_kv_head,
            head_dim=self.head_dim,
            device=device,
            dtype=dtype,
        )

    def _infer(self, idx, cache):
        _, time = idx.shape
        pos = cache.pos
        if pos + time > self.block_size:
            raise ValueError(f"context length {pos + time} exceeds block_size {self.block_size}")
        x = self.wte(idx)
        for block in self.blocks:
            x = block.forward_cached(x, cache, pos, self.cos, self.sin)
        cache.pos = pos + time
        return self._logits(x)

    @torch.no_grad()
    def prefill(self, idx):
        cache = self.new_cache(idx.size(0), idx.device, self.wte.weight.dtype)
        return self._infer(idx, cache), cache

    @torch.no_grad()
    def decode_step(self, idx, cache):
        return self._infer(idx, cache), cache

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, eos_id=None, temperature=1.0):
        """Prefill, then one token at a time. temperature 0 is greedy."""
        if idx.size(1) > self.block_size:
            idx = idx[:, -self.block_size :]
        room = self.block_size - idx.size(1)
        steps = min(max_new_tokens, room)
        logits, cache = self.prefill(idx)
        for _ in range(steps):
            idx_next = _next_token(logits[:, -1, :], temperature)
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_id is not None and int(idx_next[0, 0]) == eos_id:
                break
            logits, cache = self.decode_step(idx_next, cache)
        return idx


def _next_token(row_logits, temperature):
    if temperature == 0:
        return row_logits.argmax(dim=-1, keepdim=True)
    probs = F.softmax(row_logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def build_model(config):
    return GPT(config)


def verify_cache(model, n_prompt=8, n_new=3):
    """Max absolute logit gap between the cache path and a full forward."""
    if n_prompt + n_new > model.block_size:
        raise ValueError("verification sequence exceeds block_size")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, model.vocab_size, (1, n_prompt), generator=generator).to(device)
    new_tokens = torch.randint(0, model.vocab_size, (1, n_new), generator=generator).to(device)
    with torch.no_grad():
        prefill_logits, cache = model.prefill(prompt)
        reference, _ = model(prompt)
        prefill_gap = (prefill_logits[:, -1, :] - reference[:, -1, :]).abs().max().item()
        decode_gap = 0.0
        grown = prompt
        for index in range(n_new):
            step = new_tokens[:, index : index + 1]
            step_logits, cache = model.decode_step(step, cache)
            grown = torch.cat((grown, step), dim=1)
            reference, _ = model(grown)
            decode_gap = max(
                decode_gap,
                (step_logits[:, -1, :] - reference[:, -1, :]).abs().max().item(),
            )
    if was_training:
        model.train()
    return {"prefill_gap": prefill_gap, "decode_gap": decode_gap}
