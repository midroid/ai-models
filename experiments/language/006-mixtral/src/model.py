"""Character-level GPT with a Mixtral-style mixture of experts.

`moe=False` is the dense GPT from experiment 005: one GELU feed-forward
network per layer. `moe=True` replaces that network with eight copies. A
router sends each token to two of them. Attention, rotary positions, the KV
cache, and the loss on the next character stay as they are.

The gate follows Mixtral. The top two router logits are kept, the rest are
treated as -inf, and softmax over those two scales the expert outputs. The
load-balancing term is the Switch auxiliary loss on the softmax over all
eight logits. The hard assignment is detached, so that term is what moves
router logits the language loss never touches.

Shapes below: B batch, T time, C embedding width, H heads, E experts, K = 2.
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

    One token-layer is one token passing through one block. The mixture still
    counts as one block: the router runs two experts, and that is recorded
    separately as expert load, not as extra token-layers.
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

    def forward(self, x, cos, sin):
        """Full-sequence attention. Positions are 0..T-1."""
        batch, time, _ = x.shape
        query = self._rope(self._shape(self.c_q(x), batch, time), slice(0, time), cos, sin)
        key = self._rope(self._shape(self.c_k(x), batch, time), slice(0, time), cos, sin)
        # Values are not rotated. Rotary position lives on the query-key pair.
        value = self._shape(self.c_v(x), batch, time)
        return self._merge(self._attend(query, key, value))

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
        return self._merge(self._attend(query, key, value))


class MLP(nn.Module):
    """Position-wise feed-forward network, 4x wider, with GELU.

    One of these is the dense block. The mixture uses eight copies as experts,
    so the expert itself is not a new nonlinearity.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class SparseMoE(nn.Module):
    """Top-2 mixture of `n_expert` feed-forward networks.

    Every token is sent to exactly two experts. There is no capacity limit
    and no dropped token. Expert outputs are added with the Mixtral gate:
    softmax over the two kept logits, and zero on the rest.
    """

    def __init__(self, n_embd, n_expert, top_k):
        super().__init__()
        if top_k < 1 or top_k > n_expert:
            raise ValueError(f"top_k ({top_k}) must lie in 1..{n_expert}")
        self.n_expert = n_expert
        self.top_k = top_k
        self.gate = nn.Linear(n_embd, n_expert, bias=False)
        self.experts = nn.ModuleList([MLP(n_embd) for _ in range(n_expert)])

    def route(self, flat):
        """Router distribution, top-k indices, and the renormalized gate.

        `probs` is the softmax over all experts, used by the auxiliary loss.
        `gates` is the softmax over the kept logits only, used to mix experts.
        """
        logits = self.gate(flat)
        probs = F.softmax(logits, dim=-1)
        top_val, top_idx = torch.topk(logits, self.top_k, dim=-1)
        gates = F.softmax(top_val, dim=-1)
        return probs, top_idx, gates

    def full_gates(self, top_idx, gates):
        """Dense (token, expert) weights. Unselected experts are exactly zero."""
        full = gates.new_zeros(top_idx.size(0), self.n_expert)
        full.scatter_(1, top_idx, gates)
        return full

    def assignment_frequency(self, top_idx, dtype):
        """Fraction of (token, slot) assignments landing on each expert.

        The fractions sum to 1. This is the Switch `f` vector. It is a count
        of hard choices, so callers that put it in a loss detach it.
        """
        assignment = F.one_hot(top_idx, num_classes=self.n_expert).to(dtype=dtype)
        return assignment.mean(dim=(0, 1))

    def aux_loss(self, probs, top_idx):
        """Switch load-balancing loss for one layer.

        `n_expert * sum(f * P)` equals 1 when the router probability `P` is
        uniform. `f` is detached so the language loss, not this count, is what
        trains an expert's weights.
        """
        frequency = self.assignment_frequency(top_idx, probs.dtype)
        density = probs.mean(dim=0)
        return self.n_expert * (frequency.detach() * density).sum()

    def dispatch(self, flat, top_idx, gates):
        """Run only the experts that were selected, and mix them by `gates`."""
        out = torch.zeros_like(flat)
        for index, expert in enumerate(self.experts):
            selected = top_idx == index
            if not selected.any():
                continue
            token_idx, slot_idx = selected.nonzero(as_tuple=True)
            expert_out = expert(flat.index_select(0, token_idx))
            weighted = expert_out * gates[token_idx, slot_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, weighted)
        return out

    def dense_combine(self, flat, top_idx, gates):
        """Run every expert on every token, then apply the same sparse weights.

        Unselected weights are zero, so this is the Mixtral sum written without
        the dispatch. The smoke uses it as the reference for `dispatch`.
        """
        full = self.full_gates(top_idx, gates)
        out = torch.zeros_like(flat)
        for index, expert in enumerate(self.experts):
            out = out + expert(flat) * full[:, index].unsqueeze(-1)
        return out

    def forward(self, x):
        """Return the mixed output, the auxiliary loss, and expert fractions."""
        batch, time, _ = x.shape
        flat = x.reshape(-1, x.size(-1))
        probs, top_idx, gates = self.route(flat)
        mixed = self.dispatch(flat, top_idx, gates)
        aux = self.aux_loss(probs, top_idx)
        fractions = self.assignment_frequency(top_idx, probs.dtype)
        return mixed.view(batch, time, -1), aux, fractions


class Block(nn.Module):
    """Pre-norm residual block: attention, then a dense or sparse feed-forward."""

    def __init__(self, n_embd, n_head, layer_idx, moe, n_expert, top_k):
        super().__init__()
        self.moe = moe
        self.attn = CausalSelfAttention(n_embd, n_head, layer_idx)
        self.mlp = SparseMoE(n_embd, n_expert, top_k) if moe else MLP(n_embd)

    def _feed_forward(self, x):
        if self.moe:
            return self.mlp(x)
        return self.mlp(x), x.new_zeros(()), None

    def forward(self, x, cos, sin):
        x = x + self.attn(rms_norm(x), cos, sin)
        y, aux, fractions = self._feed_forward(rms_norm(x))
        return x + y, aux, fractions

    def forward_cached(self, x, cache, pos, cos, sin):
        x = x + self.attn.forward_cached(rms_norm(x), cache, pos, cos, sin)
        y, aux, fractions = self._feed_forward(rms_norm(x))
        return x + y, aux, fractions


class GPT(nn.Module):
    """Character-level transformer. `moe` selects the sparse feed-forward."""

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, moe, n_expert, top_k):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.moe = moe
        self.n_expert = n_expert
        self.top_k = top_k

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList(
            [
                Block(n_embd, n_head, layer_idx, moe, n_expert, top_k)
                for layer_idx in range(n_layer)
            ]
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        cos, sin = build_rope(block_size, n_embd // n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.stats = InferStats()
        self.expert_fractions = []
        self.apply(self._init_weights)
        # Each block adds two residuals (attention and MLP), so the residual
        # projections are scaled by 1/sqrt(2 * n_layer), matching GPT-2.
        # Expert output projections are residual projections too.
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

    def expert_parameter_count(self):
        """Parameters inside one expert. Zero for the dense feed-forward."""
        if not self.moe:
            return 0
        expert = self.blocks[0].mlp.experts[0]
        return sum(p.numel() for p in expert.parameters())

    def num_active_parameters(self):
        """Parameters touched by one token.

        The dense model activates everything. The mixture activates the
        embedding, the attention, the router, and `top_k` of the experts.
        """
        total = self.num_parameters()
        if not self.moe:
            return total
        inactive = (self.n_expert - self.top_k) * self.expert_parameter_count() * self.n_layer
        return total - inactive

    def kv_bytes_per_token(self):
        """Bytes of KV cache for one token, across every layer, in float32."""
        head_dim = self.n_embd // self.n_head
        return self.n_layer * 2 * self.n_head * head_dim * 4

    def _logits(self, x):
        return self.lm_head(rms_norm(x))

    def _remember_fractions(self, fractions):
        # Detach so a later backward does not keep the routing graph alive
        # through a list stored on the module.
        self.expert_fractions = [row.detach() for row in fractions]

    def forward(self, idx, targets=None):
        """Full sequence. targets, when set, is the next character at each position.

        The second return is cross-entropy, or None when targets is omitted.
        The third return is the mean Switch auxiliary loss across layers. It
        is zero for the dense model.
        """
        _, time = idx.shape
        if time > self.block_size:
            raise ValueError(f"sequence length {time} exceeds block_size {self.block_size}")
        x = self.wte(idx)
        aux_total = x.new_zeros(())
        fractions = []
        for block in self.blocks:
            x, aux, fraction = block(x, self.cos, self.sin)
            aux_total = aux_total + aux
            if fraction is not None:
                fractions.append(fraction)
        self._remember_fractions(fractions)
        logits = self._logits(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, aux_total / self.n_layer

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
        fractions = []
        for block in self.blocks:
            x, _, fraction = block.forward_cached(x, cache, pos, self.cos, self.sin)
            self.stats.encoder_token_layers += time
            if fraction is not None:
                fractions.append(fraction)
        self._remember_fractions(fractions)
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
        moe=config["moe"],
        n_expert=config["n_expert"],
        top_k=config["top_k"],
    )


def verify_cache(model, n_prompt=8, n_new=3):
    """Compare prefill and decode logits with a full forward of the same tokens.

    The gap is measured at the last character. The mixture is position-wise,
    so a cached step has to match the full forward the same way the dense GPT
    does. Returns the max absolute logit gap and the token-layer counts. The
    model is left in the training mode it started in.
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
        reference, _, _ = model(prompt)
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
            reference, _, _ = model(grown)
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
