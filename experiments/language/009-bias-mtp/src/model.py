"""Character-level DeepSeekMoE with a router bias and a second predicted character.

The block is experiment 007: one shared expert, fifteen routed experts, top-3.
Attention, the KV cache, and the printed loss on the next character stay as
they are. `use_bias=False` keeps the Switch auxiliary loss on the optimizer.
`use_bias=True` drops that term. A buffer added only to the top-k choice
nudges overloaded experts down and starved experts up. The mix weights are
still the softmax of the raw logits, so the buffer is not a learned scale
and AdamW never steps on it.

The second-token head is training only. It reads the main hidden state and
the embedding of the next character, through one dense block, and is not
called from prefill or decode. The block is dense so this experiment's router
stays the four main layers.

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

    One token-layer is one token passing through one block. A DeepSeekMoE
    block still counts as one, even though it runs the shared expert and
    three routed experts. That load is recorded as expert fractions, not as
    extra token-layers.
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
    """Position-wise feed-forward network with GELU.

    `hidden` defaults to 4x the embedding, which is the Mixtral expert. The
    DeepSeekMoE experts pass half of that, so four of them match two full ones.
    """

    def __init__(self, n_embd, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = 4 * n_embd
        self.hidden = hidden
        self.c_fc = nn.Linear(n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class SparseMoE(nn.Module):
    """Top-k mixture of `n_expert` feed-forward networks.

    Every token is sent to exactly `top_k` experts. There is no capacity limit
    and no dropped token. Expert outputs are added with the Mixtral gate:
    softmax over the kept logits, and zero on the rest.

    `bias` is a buffer, not a parameter. When `use_bias` is set, top-k reads
    `logits + bias` and the gate still reads `logits`. The bias update lives
    outside the backward pass.
    """

    def __init__(self, n_embd, n_expert, top_k, hidden=None, use_bias=False):
        super().__init__()
        if top_k < 1 or top_k > n_expert:
            raise ValueError(f"top_k ({top_k}) must lie in 1..{n_expert}")
        self.n_expert = n_expert
        self.top_k = top_k
        self.use_bias = use_bias
        self.gate = nn.Linear(n_embd, n_expert, bias=False)
        self.experts = nn.ModuleList([MLP(n_embd, hidden) for _ in range(n_expert)])
        # Zeros so a fresh bias router matches top-k on the raw logits.
        self.register_buffer("bias", torch.zeros(n_expert))

    def route(self, flat):
        """Router distribution, top-k indices, and the renormalized gate.

        `probs` is the softmax over all experts, used by the auxiliary loss.
        `gates` is the softmax over the kept raw logits, used to mix experts.
        The bias can change which logits are kept. It does not scale them.
        """
        logits = self.gate(flat)
        probs = F.softmax(logits, dim=-1)
        # `scores` decide membership. `chosen` is gathered from `logits`, not
        # from `scores`, so a large bias can pull an expert into the top-k
        # without inflating the weight it is mixed with.
        scores = logits + self.bias if self.use_bias else logits
        _top_scores, top_idx = torch.topk(scores, self.top_k, dim=-1)
        chosen = logits.gather(-1, top_idx)
        gates = F.softmax(chosen, dim=-1)
        return probs, top_idx, gates

    def apply_bias_update(self, frequency, rate):
        """Move the bias from an assignment fraction that sums to 1.

        An expert above the uniform share `1 / n_expert` loses `rate`. An
        expert below it gains `rate`. An exact tie stays put. This is not a
        gradient step.
        """
        if frequency.shape != self.bias.shape:
            raise ValueError(
                f"frequency shape {tuple(frequency.shape)} != bias {tuple(self.bias.shape)}"
            )
        target = 1.0 / self.n_expert
        self.bias.add_(rate * torch.sign(target - frequency))

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
        """Return the mix, the auxiliary loss, routed fractions, and no shared fraction.

        The fourth value is None so this block and DeepSeekMoE share one
        return shape. Mixtral has no shared expert.
        """
        batch, time, _ = x.shape
        flat = x.reshape(-1, x.size(-1))
        probs, top_idx, gates = self.route(flat)
        mixed = self.dispatch(flat, top_idx, gates)
        aux = self.aux_loss(probs, top_idx)
        fractions = self.assignment_frequency(top_idx, probs.dtype)
        return mixed.view(batch, time, -1), aux, fractions, None


class DeepSeekMoE(nn.Module):
    """Fine-grained routed experts, plus experts that every token runs.

    Cutting each Mixtral expert in half and activating twice as many of them
    keeps the active parameter count fixed. The number of ways to pick the
    active set grows. One of those narrow experts is then taken out of the
    router and run on every token, so the common part of the block is not
    rebuilt inside several routed experts.

    With that cut, hidden size 256, one shared expert and top-3 routed
    experts are four narrow experts. That matches Mixtral's two experts of
    hidden size 512. The router has 15 rows, not 16: the shared expert is
    not one of them, so the gate cannot drop it. Its output is added with
    coefficient 1. The auxiliary loss sees only the routed softmax.
    """

    def __init__(self, n_embd, n_routed, top_k, n_shared, hidden, use_bias=False):
        super().__init__()
        if n_shared < 1:
            raise ValueError(f"n_shared ({n_shared}) must be at least 1")
        self.n_shared = n_shared
        self.shared = nn.ModuleList([MLP(n_embd, hidden) for _ in range(n_shared)])
        self.routed = SparseMoE(n_embd, n_routed, top_k, hidden, use_bias=use_bias)

    def shared_sum(self, flat):
        """Add the shared experts with weight 1. No router logit scales them."""
        total = torch.zeros_like(flat)
        for expert in self.shared:
            total = total + expert(flat)
        return total

    def forward(self, x):
        """Return the mix, the routed auxiliary loss, routed fractions, and 1.

        The last value is the shared-expert token fraction. Every token runs
        the shared expert, so it is 1.
        """
        batch, time, _ = x.shape
        flat = x.reshape(-1, x.size(-1))
        routed, aux, fractions, _shared = self.routed(x)
        mixed = routed + self.shared_sum(flat).view(batch, time, -1)
        return mixed, aux, fractions, x.new_ones(())


class Block(nn.Module):
    """Pre-norm residual block: attention, then a Mixtral or DeepSeekMoE mix."""

    def __init__(self, n_embd, n_head, layer_idx, ffn, n_routed, top_k, n_shared, hidden, use_bias=False):
        super().__init__()
        self.ffn = ffn
        self.attn = CausalSelfAttention(n_embd, n_head, layer_idx)
        if ffn == "dsmoe":
            self.mlp = DeepSeekMoE(n_embd, n_routed, top_k, n_shared, hidden, use_bias=use_bias)
        elif ffn == "mixtral":
            self.mlp = SparseMoE(n_embd, n_routed, top_k, hidden, use_bias=use_bias)
        else:
            raise ValueError(f"unknown feed-forward {ffn}")

    def forward(self, x, cos, sin):
        x = x + self.attn(rms_norm(x), cos, sin)
        y, aux, fractions, shared = self.mlp(rms_norm(x))
        return x + y, aux, fractions, shared

    def forward_cached(self, x, cache, pos, cos, sin):
        x = x + self.attn.forward_cached(rms_norm(x), cache, pos, cos, sin)
        y, aux, fractions, shared = self.mlp(rms_norm(x))
        return x + y, aux, fractions, shared


class MultiTokenHead(nn.Module):
    """Predict the character after the next one. Training only.

    The main head already predicts the next character from the hidden state.
    This head is shown that character's embedding and asked for the one after
    it, so the extra loss can move the main hidden states. The block is a
    dense GELU network, not another mixture: a second router would make the
    bias update count assignments from two places. A separate output head
    keeps the main head the same width as experiment 007. Prefill and decode
    never call this module, so it does not grow the KV cache.
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.proj = nn.Linear(2 * n_embd, n_embd, bias=False)
        # layer_idx is unused by the full-sequence path. The main cache is separate.
        self.attn = CausalSelfAttention(n_embd, n_head, layer_idx=0)
        self.mlp = MLP(n_embd, hidden=4 * n_embd)

    def forward(self, hidden, next_tokens, embed, cos, sin, head):
        """hidden (B, T, C) is the main stack output. next_tokens is the main target.

        Embedding `next_tokens` is teacher forcing. The head sees the true
        next character and scores the character after that.
        """
        future = embed(next_tokens)
        x = self.proj(torch.cat((hidden, future), dim=-1))
        x = x + self.attn(rms_norm(x), cos, sin)
        x = x + self.mlp(rms_norm(x))
        return head(rms_norm(x))


class GPT(nn.Module):
    """Character-level transformer. `use_bias` selects the router balance."""

    def __init__(
        self,
        vocab_size,
        block_size,
        n_embd,
        n_head,
        n_layer,
        ffn,
        n_routed,
        top_k,
        n_shared,
        hidden,
        use_bias=False,
    ):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.ffn = ffn
        self.n_routed = n_routed
        self.top_k = top_k
        self.n_shared = n_shared
        self.hidden = hidden
        self.use_bias = use_bias
        self.mtp = None
        self.mtp_head = None

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList(
            [
                Block(n_embd, n_head, layer_idx, ffn, n_routed, top_k, n_shared, hidden, use_bias)
                for layer_idx in range(n_layer)
            ]
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        cos, sin = build_rope(block_size, n_embd // n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.stats = InferStats()
        self.expert_fractions = []
        self.shared_fractions = []
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

    def attach_mtp(self):
        """Add the second-token block and head after the main stack is initialized.

        Callers that need the main weights to match a model without this head
        should restore the generator afterward. The new parameters are
        initialized here, including the residual scale on `c_proj`.
        """
        if self.mtp is not None:
            return
        self.mtp = MultiTokenHead(self.n_embd, self.n_head)
        self.mtp_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        self.mtp.apply(self._init_weights)
        self._init_weights(self.mtp_head)
        scale = 0.02 / math.sqrt(2 * self.n_layer)
        for name, param in self.mtp.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=scale)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def mtp_parameter_count(self):
        """Training-only parameters. Zero when the second-token head is absent."""
        total = 0
        if self.mtp is not None:
            total += sum(p.numel() for p in self.mtp.parameters())
        if self.mtp_head is not None:
            total += sum(p.numel() for p in self.mtp_head.parameters())
        return total

    def num_inference_parameters(self):
        """Parameters used by prefill and decode. The second-token head is excluded."""
        return self.num_parameters() - self.mtp_parameter_count()

    def _routed_module(self):
        mlp = self.blocks[0].mlp
        if isinstance(mlp, DeepSeekMoE):
            return mlp.routed
        return mlp

    def expert_parameter_count(self):
        """Parameters inside one routed expert."""
        expert = self._routed_module().experts[0]
        return sum(p.numel() for p in expert.parameters())

    def active_ffn_parameters(self):
        """Feed-forward parameters one token runs, in one layer.

        Shared experts always run. Only `top_k` routed experts run.
        """
        return (self.n_shared + self.top_k) * self.expert_parameter_count()

    def expert_parameters_per_layer(self):
        """Every expert copy in one layer, shared and routed."""
        return (self.n_shared + self.n_routed) * self.expert_parameter_count()

    def num_active_parameters(self):
        """Parameters touched by one token.

        Idle routed experts drop out. Shared experts stay. Dropping them
        would report a smaller active model than Mixtral for the same
        feed-forward work.
        """
        idle = (self.n_routed - self.top_k) * self.expert_parameter_count() * self.n_layer
        return self.num_inference_parameters() - idle

    def update_expert_bias(self, rate):
        """Apply one bias step from the routed fractions of the latest forward.

        Eval forwards also write those fractions. Call this only after a
        training step, or the printed loss would depend on how many eval
        batches were drawn.
        """
        if not self.use_bias:
            return
        if len(self.expert_fractions) != self.n_layer:
            raise RuntimeError("bias update requires a routed fraction for every layer")
        for fraction, block in zip(self.expert_fractions, self.blocks):
            block.mlp.routed.apply_bias_update(fraction, rate)

    def bias_extent(self):
        """Minimum and maximum bias across every routed expert, or None."""
        if not self.use_bias:
            return None
        rows = [block.mlp.routed.bias.detach() for block in self.blocks]
        stacked = torch.stack(rows)
        return float(stacked.min().item()), float(stacked.max().item())

    def kv_bytes_per_token(self):
        """Bytes of KV cache for one token, across every layer, in float32."""
        head_dim = self.n_embd // self.n_head
        return self.n_layer * 2 * self.n_head * head_dim * 4

    def _logits(self, x):
        return self.lm_head(rms_norm(x))

    def _remember_fractions(self, fractions, shared):
        # Detach so a later backward does not keep the routing graph alive
        # through a list stored on the module.
        self.expert_fractions = [row.detach() for row in fractions]
        self.shared_fractions = [row.detach() for row in shared]

    def forward(self, idx, targets=None, future=None):
        """Full sequence. targets, when set, is the next character at each position.

        The second return is cross-entropy, or None when targets is omitted.
        The third return is the mean Switch auxiliary loss across layers.
        The fourth is the second-token cross-entropy when `future` is set and
        this model has the training head, otherwise None.
        """
        _, time = idx.shape
        if time > self.block_size:
            raise ValueError(f"sequence length {time} exceeds block_size {self.block_size}")
        x = self.wte(idx)
        aux_total = x.new_zeros(())
        fractions = []
        shared = []
        for block in self.blocks:
            x, aux, fraction, shared_fraction = block(x, self.cos, self.sin)
            aux_total = aux_total + aux
            if fraction is not None:
                fractions.append(fraction)
            if shared_fraction is not None:
                shared.append(shared_fraction)
        self._remember_fractions(fractions, shared)
        logits = self._logits(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        mtp_loss = None
        if self.mtp is not None and targets is not None and future is not None:
            mtp_logits = self.mtp(x, targets, self.wte, self.cos, self.sin, self.mtp_head)
            mtp_loss = F.cross_entropy(mtp_logits.view(-1, mtp_logits.size(-1)), future.view(-1))
        return logits, loss, aux_total / self.n_layer, mtp_loss

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
        shared = []
        for block in self.blocks:
            x, _, fraction, shared_fraction = block.forward_cached(x, cache, pos, self.cos, self.sin)
            self.stats.encoder_token_layers += time
            if fraction is not None:
                fractions.append(fraction)
            if shared_fraction is not None:
                shared.append(shared_fraction)
        self._remember_fractions(fractions, shared)
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
    model = GPT(
        vocab_size=vocab_size,
        block_size=config["block_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        ffn=config["ffn"],
        n_routed=config["n_routed"],
        top_k=config["top_k"],
        n_shared=config["n_shared"],
        hidden=config["hidden"],
        use_bias=config.get("use_bias", False),
    )
    if config.get("mtp"):
        model.attach_mtp()
    return model


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
        reference, *_rest = model(prompt)
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
            reference, *_rest = model(grown)
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
