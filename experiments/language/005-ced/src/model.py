"""Character-level GPT with a causal encoder-decoder mode.

The inference shape follows nanochat: separate query, key, and value
projections, rotary positions, RMSNorm, and SDPA. A KV cache stores one
prefill of the prompt, then each new character is one decode step.

`ced=False` is that GPT. Every layer builds its own keys and values.

`ced=True` splits the stack in half. The encoder is the bottom half. The
decoder's keys and values are projections of the encoder output H, one
projection per decoder layer:

    K_l, V_l = H W_l

Training still runs the decoder on every character, because every position
needs a loss. Prefill runs the decoder only on the last character. That
matches the full forward: a decoder position reads H, and it does not read
another position's decoder residual. Sliding-window attention is left out so
this match stays exact.

Shapes below: B batch, T time, C embedding width, H heads.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


def rms_norm(x, eps=1e-6):
    """Root-mean-square norm along the last dimension. No learned scale."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps)


def apply_rotary_emb(x, cos, sin):
    """Rotate the last dimension in pairs.

    `cos` and `sin` broadcast over (B, T, heads, head_dim/2). The sign matches
    nanochat: the rotation is the transpose of the textbook one, and only the
    relative rotation between query and key matters.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1)


def build_rope(seq_len, head_dim, base=100_000):
    """Cos and sin tables of shape (1, seq_len, 1, head_dim/2).

    Row t is the rotation for character t. The table is `block_size` rows long,
    so a cached generation cannot grow past that context.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be even for rotary positions")
    channel = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (channel / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos()[None, :, None, :]
    sin = freqs.sin()[None, :, None, :]
    return cos, sin


class InferStats:
    """Token-layers executed by the most recent prefill or decode step.

    One token-layer is one token passing through one block's attention and MLP.
    Projecting decoder keys for the whole prompt does not add a token-layer:
    that projection has no attention and no MLP.
    """

    def __init__(self):
        self.encoder_token_layers = 0
        self.decoder_token_layers = 0

    @property
    def token_layers(self):
        return self.encoder_token_layers + self.decoder_token_layers


class KVCache:
    """Keys and values for every layer, written at an absolute position.

    `pos` is how many tokens are already stored. Layers in one forward all
    write at the same `pos`; the caller advances `pos` once, after the last
    layer.
    """

    def __init__(self, n_layer, batch, max_seq, n_head, head_dim, device, dtype):
        self.pos = 0
        # (layer, batch, time, head, head_dim). Time is allocated up front so
        # decode can append without reallocating.
        self.k = torch.zeros(n_layer, batch, max_seq, n_head, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros(n_layer, batch, max_seq, n_head, head_dim, device=device, dtype=dtype)


class CausalSelfAttention(nn.Module):
    """SDPA with separate Q, K, and V projections.

    `memory` is the tensor K and V are projected from. Encoder layers pass the
    same tensor as the query input. Decoder layers pass the encoder output H.
    """

    def __init__(self, n_embd, n_head, layer_idx):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.layer_idx = layer_idx
        self.c_q = nn.Linear(n_embd, n_embd, bias=False)
        self.c_k = nn.Linear(n_embd, n_embd, bias=False)
        self.c_v = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def _shape(self, projected, batch, time):
        return projected.view(batch, time, self.n_head, self.head_dim)

    def _rope(self, states, positions, cos, sin):
        # positions is a slice into the rotary tables. states is (B, T, H, D).
        return apply_rotary_emb(states, cos[:, positions], sin[:, positions])

    def _attend(self, query, key, value):
        """query (B, Tq, H, D), key and value (B, Tk, H, D).

        Equal lengths use a causal mask. A single new query attends to every
        cached key, which is the last row of that causal mask.
        """
        # SDPA wants (B, heads, time, head_dim). The cache stores time before heads.
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

    def _merge(self, heads):
        batch, time, _, _ = heads.shape
        return self.c_proj(heads.view(batch, time, -1))

    def forward(self, x, memory, cos, sin):
        """Full-sequence attention. x and memory both have length T, positions 0..T-1."""
        batch, time, _ = x.shape
        query = self._rope(self._shape(self.c_q(x), batch, time), slice(0, time), cos, sin)
        key = self._rope(self._shape(self.c_k(memory), batch, time), slice(0, time), cos, sin)
        # Values are not rotated. Rotary position lives on the query-key pair.
        value = self._shape(self.c_v(memory), batch, time)
        return self._merge(self._attend(query, key, value))

    def write_kv(self, memory, cache, pos, cos, sin):
        """Project K and V from `memory` and store them at `pos`. No query, no MLP."""
        batch, time, _ = memory.shape
        key = self._rope(self._shape(self.c_k(memory), batch, time), slice(pos, pos + time), cos, sin)
        value = self._shape(self.c_v(memory), batch, time)
        cache.k[self.layer_idx, :, pos : pos + time] = key
        cache.v[self.layer_idx, :, pos : pos + time] = value

    def forward_cached(self, x, cache, pos, cos, sin, kv_span=None):
        """Attention for tokens already backed by the cache.

        `kv_span is None` means these tokens are their own keys: write K and V,
        then attend over everything stored so far. A span means K and V were
        already written for that many tokens starting at `pos`, and `x` is the
        last query or queries in that span.
        """
        batch, time, _ = x.shape
        if kv_span is None:
            self.write_kv(x, cache, pos, cos, sin)
            query_pos = slice(pos, pos + time)
            end = pos + time
        else:
            if time > kv_span:
                raise ValueError(f"query length {time} exceeds kv span {kv_span}")
            query_pos = slice(pos + kv_span - time, pos + kv_span)
            end = pos + kv_span
        query = self._rope(self._shape(self.c_q(x), batch, time), query_pos, cos, sin)
        key = cache.k[self.layer_idx, :, :end]
        value = cache.v[self.layer_idx, :, :end]
        return self._merge(self._attend(query, key, value))


class MLP(nn.Module):
    """Position-wise feed-forward network, 4x wider, with GELU."""

    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    """Pre-norm residual block: attention, then the feed-forward network."""

    def __init__(self, n_embd, n_head, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, layer_idx)
        self.mlp = MLP(n_embd)

    def forward(self, x, memory, cos, sin):
        # Encoder layers pass memory=None, so keys and values come from x.
        # Decoder layers pass the encoder output H. The query still comes from x,
        # which changes as it moves through the decoder. H does not.
        source = x if memory is None else memory
        x = x + self.attn(rms_norm(x), rms_norm(source), cos, sin)
        x = x + self.mlp(rms_norm(x))
        return x

    def forward_cached(self, x, cache, pos, cos, sin, kv_span=None, memory=None):
        # No span: this layer's own hidden state is the key and the value.
        # A span is a decoder layer at inference. Keys for every character in the
        # span were projected from H. Only the last character is queried.
        if kv_span is None:
            y = self.attn.forward_cached(rms_norm(x), cache, pos, cos, sin, kv_span=None)
        else:
            self.attn.write_kv(rms_norm(memory), cache, pos, cos, sin)
            y = self.attn.forward_cached(rms_norm(x), cache, pos, cos, sin, kv_span=kv_span)
        x = x + y
        x = x + self.mlp(rms_norm(x))
        return x


class GPT(nn.Module):
    """Character-level transformer. `ced` selects the encoder-decoder split."""

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, ced):
        super().__init__()
        if ced and n_layer % 2 != 0:
            raise ValueError(f"ced requires an even n_layer, got {n_layer}")
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.ced = ced
        # With ced off, every layer is an "encoder" layer: it builds its own keys.
        # With ced on, the bottom half builds H and the top half reads it.
        self.n_encoder = n_layer // 2 if ced else n_layer
        self.n_decoder = n_layer // 2 if ced else 0

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, layer_idx) for layer_idx in range(n_layer)]
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        cos, sin = build_rope(block_size, n_embd // n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.stats = InferStats()
        self.apply(self._init_weights)
        # Each block adds two residuals (attention and MLP), so the residual
        # projections are scaled by 1/sqrt(2 * n_layer), matching GPT-2.
        scale = 0.02 / math.sqrt(2 * n_layer)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=scale)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def kv_bytes_per_token(self):
        """Bytes of KV cache for one token, across every layer, in float32."""
        head_dim = self.n_embd // self.n_head
        return self.n_layer * 2 * self.n_head * head_dim * 4

    def _logits(self, x):
        return self.lm_head(rms_norm(x))

    def forward(self, idx, targets=None):
        """Full sequence. targets, when set, is the next character at each position."""
        _, time = idx.shape
        if time > self.block_size:
            raise ValueError(f"sequence length {time} exceeds block_size {self.block_size}")
        x = self.wte(idx)
        for block in self.blocks[: self.n_encoder]:
            x = block(x, None, self.cos, self.sin)
        if self.ced:
            # Keep H fixed. Decoder keys are projections of this tensor, not of
            # the residual the decoder is about to update.
            memory = x
            for block in self.blocks[self.n_encoder :]:
                x = block(x, memory, self.cos, self.sin)
        logits = self._logits(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def _check_room(self, pos, time):
        if pos + time > self.block_size:
            raise ValueError(
                f"context length {pos + time} exceeds block_size {self.block_size}"
            )

    def _infer(self, idx, cache):
        """One prefill or decode step. Advances the cache once, after the last layer."""
        _, time = idx.shape
        pos = cache.pos
        self._check_room(pos, time)
        self.stats = InferStats()
        x = self.wte(idx)
        for block in self.blocks[: self.n_encoder]:
            x = block.forward_cached(x, cache, pos, self.cos, self.sin)
            self.stats.encoder_token_layers += time
        if self.ced:
            memory = x
            # One query, at the last character. It attends to keys projected
            # from every character of H. Earlier characters never enter the
            # decoder MLP. That is the prefill saving, and it matches a full
            # decoder forward because a decoder position does not read another
            # position's decoder residual.
            queries = memory[:, -1:, :]
            for block in self.blocks[self.n_encoder :]:
                queries = block.forward_cached(
                    queries,
                    cache,
                    pos,
                    self.cos,
                    self.sin,
                    kv_span=time,
                    memory=memory,
                )
                # Count the queried character, not the prompt characters whose
                # keys were only projected.
                self.stats.decoder_token_layers += 1
            x = queries
        # Every layer wrote at the same `pos`. Advance once, after the last layer.
        cache.pos = pos + time
        return self._logits(x)

    def new_cache(self, batch, device, dtype):
        return KVCache(
            n_layer=self.n_layer,
            batch=batch,
            max_seq=self.block_size,
            n_head=self.n_head,
            head_dim=self.n_embd // self.n_head,
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def prefill(self, idx):
        """Run the prompt once. Returns logits (B, T, vocab) and a cache.

        In CED mode the decoder portion of `logits` is only the last character,
        so the tensor has time 1. The GPT returns a logit row for every prompt
        character.
        """
        cache = self.new_cache(idx.size(0), idx.device, self.wte.weight.dtype)
        logits = self._infer(idx, cache)
        return logits, cache

    @torch.no_grad()
    def decode_step(self, idx, cache):
        """Append the characters in `idx` (usually one) and return their logits."""
        logits = self._infer(idx, cache)
        return logits, cache

    def generate(self, idx, max_new_tokens):
        """Prefill, then sample one character at a time. idx is (B, T)."""
        if idx.size(1) > self.block_size:
            idx = idx[:, -self.block_size :]
        room = self.block_size - idx.size(1)
        steps = min(max_new_tokens, room)
        logits, cache = self.prefill(idx)
        for _ in range(steps):
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            logits, cache = self.decode_step(idx_next, cache)
        return idx


def build_model(vocab_size, config):
    return GPT(
        vocab_size=vocab_size,
        block_size=config["block_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        ced=config["ced"],
    )


def verify_cache(model, n_prompt=8, n_new=3):
    """Compare prefill and decode logits with a full forward of the same tokens.

    The gap is measured at the last character. That is the only row CED prefill
    returns, and it is the row a decode step has to match. Returns the max
    absolute logit gap and the token-layer counts. The model is left in the
    training mode it started in.
    """
    if n_prompt + n_new > model.block_size:
        raise ValueError("verification sequence exceeds block_size")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    # A private generator so this check does not move the training seed.
    generator = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, model.vocab_size, (1, n_prompt), generator=generator).to(device)
    new_tokens = torch.randint(0, model.vocab_size, (1, n_new), generator=generator).to(device)

    with torch.no_grad():
        prefill_logits, cache = model.prefill(prompt)
        prefill_stats = {
            "encoder_token_layers": model.stats.encoder_token_layers,
            "decoder_token_layers": model.stats.decoder_token_layers,
        }
        reference, _ = model(prompt)
        prefill_gap = (prefill_logits[:, -1, :] - reference[:, -1, :]).abs().max().item()

        decode_gap = 0.0
        decode_token_layers = None
        grown = prompt
        for index in range(n_new):
            step = new_tokens[:, index : index + 1]
            step_logits, cache = model.decode_step(step, cache)
            if decode_token_layers is None:
                decode_token_layers = model.stats.token_layers
            grown = torch.cat((grown, step), dim=1)
            reference, _ = model(grown)
            decode_gap = max(
                decode_gap,
                (step_logits[:, -1, :] - reference[:, -1, :]).abs().max().item(),
            )

    if was_training:
        model.train()
    return {
        "prefill_gap": prefill_gap,
        "decode_gap": decode_gap,
        "prefill_encoder_token_layers": prefill_stats["encoder_token_layers"],
        "prefill_decoder_token_layers": prefill_stats["decoder_token_layers"],
        "decode_token_layers": decode_token_layers,
        "kv_bytes_per_token": model.kv_bytes_per_token(),
        "n_prompt": n_prompt,
    }
