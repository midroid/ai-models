"""Character-level GPT with multi-head attention or multi-head latent attention.

`mla=False` is the dense GPT from experiment 006: separate Q, K, and V, rotary
positions on the query-key pair, and a cache of full keys and values. `mla=True`
replaces that attention with the low-rank KV map from DeepSeek-V2, section 2.1.
The feed-forward network stays the single GELU MLP. The loss stays next-character
cross-entropy.

The latent is a joint compression of keys and values. Content keys are the
up-projection of that latent and are not rotated. Rotary position lives on a
separate key, shared by every head, which is cached beside the latent. Decode
up-projects the cached latent back to content keys and values. Absorbing those
up-projections into the query and the output projection is the same math; this
model leaves that rewrite out. Query compression is left out too: it does not
change what the cache stores.

Shapes below: B batch, T time, C embedding width, H heads, D content head width,
R the latent rank, P the rotary width.
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

    `cos` and `sin` broadcast over (B, T, heads, width/2). The sign matches
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


def attend(query, key, value):
    """query (B, Tq, H, Dq), key (B, Tk, H, Dq), value (B, Tk, H, Dv).

    Equal lengths use a causal mask. A single new query attends to every
    cached key, which is the last row of that causal mask. The value width
    may differ from the query width: latent attention concatenates a rotary
    slice onto Q and K only.
    """
    # SDPA wants (B, heads, time, dim). Callers store time before heads.
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


class InferStats:
    """Token-layers executed by the most recent prefill or decode step.

    One token-layer is one token passing through one block. Compressing the
    cache does not add a pass.
    """

    def __init__(self):
        self.encoder_token_layers = 0
        self.decoder_token_layers = 0

    @property
    def token_layers(self):
        return self.encoder_token_layers + self.decoder_token_layers


class KVCache:
    """Full keys and values for every layer, written at an absolute position.

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


class LatentKVCache:
    """Compressed KV state for every layer.

    Each token stores the latent `c_kv` and one rotary key shared by all heads.
    Content keys and values are rebuilt from the latent when a later query
    attends. The rotary key is stored already rotated, at the position it was
    written.
    """

    def __init__(self, n_layer, batch, max_seq, kv_rank, rope_dim, device, dtype):
        self.pos = 0
        self.c_kv = torch.zeros(n_layer, batch, max_seq, kv_rank, device=device, dtype=dtype)
        self.k_rope = torch.zeros(n_layer, batch, max_seq, rope_dim, device=device, dtype=dtype)


class CausalSelfAttention(nn.Module):
    """SDPA with separate Q, K, and V projections."""

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

    def _merge(self, heads):
        batch, time, _, _ = heads.shape
        return self.c_proj(heads.reshape(batch, time, -1))

    def forward(self, x, cos, sin):
        """Full-sequence attention. Positions are 0..T-1."""
        batch, time, _ = x.shape
        query = self._rope(self._shape(self.c_q(x), batch, time), slice(0, time), cos, sin)
        key = self._rope(self._shape(self.c_k(x), batch, time), slice(0, time), cos, sin)
        # Values are not rotated. Rotary position lives on the query-key pair.
        value = self._shape(self.c_v(x), batch, time)
        return self._merge(attend(query, key, value))

    def write_kv(self, x, cache, pos, cos, sin):
        """Project K and V and store them at `pos`. No query, no MLP."""
        batch, time, _ = x.shape
        key = self._rope(self._shape(self.c_k(x), batch, time), slice(pos, pos + time), cos, sin)
        value = self._shape(self.c_v(x), batch, time)
        cache.k[self.layer_idx, :, pos : pos + time] = key
        cache.v[self.layer_idx, :, pos : pos + time] = value

    def forward_cached(self, x, cache, pos, cos, sin):
        """Write K and V for these tokens, then attend over everything stored so far."""
        batch, time, _ = x.shape
        self.write_kv(x, cache, pos, cos, sin)
        query = self._rope(self._shape(self.c_q(x), batch, time), slice(pos, pos + time), cos, sin)
        end = pos + time
        key = cache.k[self.layer_idx, :, :end]
        value = cache.v[self.layer_idx, :, :end]
        return self._merge(attend(query, key, value))


class LatentAttention(nn.Module):
    """Multi-head attention whose keys and values share a low-rank latent.

    `w_dkv` maps the residual stream to `c_kv` of width `kv_rank`. `w_uk` and
    `w_uv` expand that latent to per-head content keys and values. Those
    content keys are not rotated: a rotation depends on absolute position, and
    the cached latent has to stay a property of the token alone.

    Position is a second key, `w_kr`, of width `rope_dim`, rotated and shared
    across heads. Each head has its own rotary query, `w_qr`. The score is
    the content dot product plus the rotary dot product, implemented by
    concatenating the two slices so SDPA scales by 1/sqrt(D + P).
    """

    def __init__(self, n_embd, n_head, layer_idx, kv_rank, rope_dim):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        if kv_rank < 1:
            raise ValueError(f"kv_rank ({kv_rank}) must be positive")
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim ({rope_dim}) must be even for rotary positions")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.kv_rank = kv_rank
        self.rope_dim = rope_dim
        self.layer_idx = layer_idx
        # Content queries stay full rank. The down-projection is only on K and V,
        # which is what the cache has to keep.
        self.w_q = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.w_qr = nn.Linear(n_embd, n_head * rope_dim, bias=False)
        self.w_dkv = nn.Linear(n_embd, kv_rank, bias=False)
        self.w_uk = nn.Linear(kv_rank, n_head * self.head_dim, bias=False)
        self.w_uv = nn.Linear(kv_rank, n_head * self.head_dim, bias=False)
        # One rotary key for the whole layer, not one per head.
        self.w_kr = nn.Linear(n_embd, rope_dim, bias=False)
        self.c_proj = nn.Linear(n_head * self.head_dim, n_embd, bias=False)

    def _heads(self, projected, batch, time, width):
        return projected.view(batch, time, self.n_head, width)

    def content_key(self, c_kv):
        """Up-project a latent to content keys. Position is not an input."""
        batch, time, _ = c_kv.shape
        return self._heads(self.w_uk(c_kv), batch, time, self.head_dim)

    def content_value(self, c_kv):
        """Up-project a latent to values."""
        batch, time, _ = c_kv.shape
        return self._heads(self.w_uv(c_kv), batch, time, self.head_dim)

    def rope_key(self, x, positions, cos, sin):
        """One rotary key per token, shape (B, T, 1, P), shared by every head."""
        raw = self.w_kr(x).unsqueeze(2)
        return apply_rotary_emb(raw, cos[:, positions], sin[:, positions])

    def keys_and_values(self, x, positions, cos, sin):
        """Latent, content keys, values, and the shared rotary key.

        Content keys depend only on `c_kv`. The same latent at two positions
        therefore produces the same content key. The rotary key does not.
        """
        c_kv = self.w_dkv(x)
        return c_kv, self.content_key(c_kv), self.content_value(c_kv), self.rope_key(x, positions, cos, sin)

    def queries(self, x, positions, cos, sin):
        """Content queries and per-head rotary queries."""
        batch, time, _ = x.shape
        q_content = self._heads(self.w_q(x), batch, time, self.head_dim)
        q_rope = self._heads(self.w_qr(x), batch, time, self.rope_dim)
        q_rope = apply_rotary_emb(q_rope, cos[:, positions], sin[:, positions])
        return q_content, q_rope

    def _combine(self, q_content, q_rope, k_content, k_rope, value):
        """Concatenate content and rotary slices, then attend.

        `k_rope` is (B, Tk, 1, P) and is broadcast across heads. Query and key
        width become D + P, so the SDPA scale is 1/sqrt(D + P). Values stay
        width D.
        """
        query = torch.cat((q_content, q_rope), dim=-1)
        key = torch.cat((k_content, k_rope.expand(-1, -1, self.n_head, -1)), dim=-1)
        return attend(query, key, value)

    def _merge(self, heads):
        batch, time, _, _ = heads.shape
        return self.c_proj(heads.reshape(batch, time, -1))

    def forward(self, x, cos, sin):
        """Full-sequence attention. Positions are 0..T-1. `cos` and `sin` are the rotary-width tables."""
        batch, time, _ = x.shape
        positions = slice(0, time)
        q_content, q_rope = self.queries(x, positions, cos, sin)
        _c_kv, k_content, value, k_rope = self.keys_and_values(x, positions, cos, sin)
        return self._merge(self._combine(q_content, q_rope, k_content, k_rope, value))

    def write_kv(self, x, cache, pos, cos, sin):
        """Store the latent and the rotated shared key. Content K and V are not stored."""
        time = x.size(1)
        c_kv, _k_content, _value, k_rope = self.keys_and_values(
            x, slice(pos, pos + time), cos, sin
        )
        cache.c_kv[self.layer_idx, :, pos : pos + time] = c_kv
        # rope_key keeps a head axis of 1 so the rotary table broadcasts. The
        # cache stores the vector itself. Rotating it here means a later read
        # does not need the position again.
        cache.k_rope[self.layer_idx, :, pos : pos + time] = k_rope.squeeze(2)

    def read_kv(self, cache, end):
        """Up-project every cached latent, and return the stored rotary keys."""
        c_kv = cache.c_kv[self.layer_idx, :, :end]
        k_rope = cache.k_rope[self.layer_idx, :, :end].unsqueeze(2)
        return self.content_key(c_kv), self.content_value(c_kv), k_rope

    def forward_cached(self, x, cache, pos, cos, sin):
        """Write the latent for these tokens, then attend over every latent stored so far."""
        batch, time, _ = x.shape
        self.write_kv(x, cache, pos, cos, sin)
        q_content, q_rope = self.queries(x, slice(pos, pos + time), cos, sin)
        k_content, value, k_rope = self.read_kv(cache, pos + time)
        return self._merge(self._combine(q_content, q_rope, k_content, k_rope, value))


class MLP(nn.Module):
    """Position-wise feed-forward network, 4x wider, with GELU."""

    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    """Pre-norm residual block: attention, then the dense feed-forward."""

    def __init__(self, n_embd, n_head, layer_idx, mla, kv_rank, rope_dim):
        super().__init__()
        self.mla = mla
        if mla:
            self.attn = LatentAttention(n_embd, n_head, layer_idx, kv_rank, rope_dim)
        else:
            self.attn = CausalSelfAttention(n_embd, n_head, layer_idx)
        self.mlp = MLP(n_embd)

    def forward(self, x, cos, sin):
        x = x + self.attn(rms_norm(x), cos, sin)
        return x + self.mlp(rms_norm(x))

    def forward_cached(self, x, cache, pos, cos, sin):
        x = x + self.attn.forward_cached(rms_norm(x), cache, pos, cos, sin)
        return x + self.mlp(rms_norm(x))


class GPT(nn.Module):
    """Character-level transformer. `mla` selects latent attention."""

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, mla, kv_rank, rope_dim):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.mla = mla
        self.kv_rank = kv_rank
        self.rope_dim = rope_dim

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList(
            [
                Block(n_embd, n_head, layer_idx, mla, kv_rank, rope_dim)
                for layer_idx in range(n_layer)
            ]
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # The multi-head path rotates the full content head. Latent attention
        # rotates only the decoupled key, so it needs a second, narrower table.
        head_dim = n_embd // n_head
        cos, sin = build_rope(block_size, head_dim)
        cos_rope, sin_rope = build_rope(block_size, rope_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos_rope", cos_rope, persistent=False)
        self.register_buffer("sin_rope", sin_rope, persistent=False)
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
        """Bytes of cache for one token, across every layer, in float32.

        Multi-head attention stores a key and a value per head. Latent
        attention stores the latent plus one shared rotary key.
        """
        if self.mla:
            return self.n_layer * (self.kv_rank + self.rope_dim) * 4
        head_dim = self.n_embd // self.n_head
        return self.n_layer * 2 * self.n_head * head_dim * 4

    def _logits(self, x):
        return self.lm_head(rms_norm(x))

    def _rope(self):
        if self.mla:
            return self.cos_rope, self.sin_rope
        return self.cos, self.sin

    def forward(self, idx, targets=None):
        """Full sequence. targets, when set, is the next character at each position.

        The second return is cross-entropy, or None when targets is omitted.
        """
        _, time = idx.shape
        if time > self.block_size:
            raise ValueError(f"sequence length {time} exceeds block_size {self.block_size}")
        cos, sin = self._rope()
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x, cos, sin)
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
        cos, sin = self._rope()
        x = self.wte(idx)
        for block in self.blocks:
            x = block.forward_cached(x, cache, pos, cos, sin)
            self.stats.encoder_token_layers += time
        # Every layer wrote at the same `pos`. Advance once, after the last layer.
        cache.pos = pos + time
        return self._logits(x)

    def new_cache(self, batch, device, dtype):
        if self.mla:
            return LatentKVCache(
                n_layer=self.n_layer,
                batch=batch,
                max_seq=self.block_size,
                kv_rank=self.kv_rank,
                rope_dim=self.rope_dim,
                device=device,
                dtype=dtype,
            )
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
        """Run the prompt once. Returns logits (B, T, vocab) and a cache."""
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
        mla=config["mla"],
        kv_rank=config["kv_rank"],
        rope_dim=config["rope_dim"],
    )


def verify_cache(model, n_prompt=8, n_new=3):
    """Compare prefill and decode logits with a full forward of the same tokens.

    The gap is measured at the last character. Latent attention rebuilds keys
    from the cached latent, so a cached step has to match the full forward the
    same way multi-head attention does. Returns the max absolute logit gap and
    the token-layer counts. The model is left in the training mode it started in.
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
