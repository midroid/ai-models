"""Check the mixture, the KV cache, then take 20 training steps.

    uv run python -m src.smoke

The cache check fails if prefill or decode logits differ from a full forward
by more than 1e-4. The mixture checks fail if the gate is not exactly top-2,
if sparse dispatch disagrees with the dense mix, if an unused expert receives
a gradient, or if the auxiliary loss disagrees with an independent count.
The training steps use the laptop width and the Tiny Shakespeare split.
"""

import math

import torch
from torch.nn import functional as F

from src.model import SparseMoE, build_model, verify_cache
from src.train import PRESETS, pick_device, train

GAP_LIMIT = 1e-4
AUX_LIMIT = 1e-4
SMOKE_STEPS = 20
# Step 0 of a fresh head sits on the uniform-guess loss, ln(65).
LOSS_BAND = 0.15


def assert_gap(name, gap, limit=GAP_LIMIT):
    if gap > limit:
        raise AssertionError(f"{name} gap {gap:.3e} exceeds {limit:.0e}")


def assert_near(name, value, target, limit):
    gap = abs(value - target)
    if gap > limit:
        raise AssertionError(f"{name} {value:.6f} is {gap:.3e} from {target:.6f}")


def check_cache(moe, device):
    config = dict(PRESETS["laptop"])
    config["moe"] = moe
    # Vocabulary size only affects the embedding and the head. 65 matches Tiny Shakespeare.
    model = build_model(65, config).to(device)
    model.eval()
    report = verify_cache(model)
    name = "moe" if moe else "gpt"
    n_prompt = report["n_prompt"]
    expected_prefill = model.n_layer * n_prompt
    if report["prefill_encoder_token_layers"] != expected_prefill:
        raise AssertionError(
            f"{name} prefill token-layers "
            f"{report['prefill_encoder_token_layers']} != {expected_prefill}"
        )
    if report["prefill_decoder_token_layers"] != 0:
        raise AssertionError(
            f"{name} prefill decoder token-layers {report['prefill_decoder_token_layers']} != 0"
        )
    if report["decode_token_layers"] != model.n_layer:
        raise AssertionError(
            f"{name} decode token-layers {report['decode_token_layers']} != {model.n_layer}"
        )
    assert_gap(f"{name} prefill", report["prefill_gap"])
    assert_gap(f"{name} decode", report["decode_gap"])
    print(
        f"check {name}  device {device}  "
        f"prefill gap {report['prefill_gap']:.3e}  decode gap {report['decode_gap']:.3e}  "
        f"prefill token-layers {report['prefill_encoder_token_layers']}  "
        f"decode token-layers {report['decode_token_layers']}  "
        f"kv bytes/token {report['kv_bytes_per_token']}  "
        f"parameters {model.num_parameters():,}  "
        f"active {model.num_active_parameters():,}"
    )
    return model


def check_counts(gpt, moe):
    """Active parameters drop the experts a token does not run."""
    if moe.num_parameters() <= gpt.num_parameters():
        raise AssertionError(
            f"moe total {moe.num_parameters()} does not exceed gpt {gpt.num_parameters()}"
        )
    one = moe.expert_parameter_count()
    expected_one = 8 * moe.n_embd * moe.n_embd
    if one != expected_one:
        raise AssertionError(f"one expert has {one} parameters, expected {expected_one}")
    for block in moe.blocks:
        for expert in block.mlp.experts:
            count = sum(p.numel() for p in expert.parameters())
            if count != one:
                raise AssertionError(f"expert parameter count {count} != {one}")
    inactive = (moe.n_expert - moe.top_k) * one * moe.n_layer
    active = moe.num_parameters() - inactive
    if moe.num_active_parameters() != active:
        raise AssertionError(
            f"active {moe.num_active_parameters()} != total - inactive experts {active}"
        )
    if not (gpt.num_parameters() < active < moe.num_parameters()):
        raise AssertionError(
            f"expected gpt {gpt.num_parameters()} < active {active} < total {moe.num_parameters()}"
        )
    if gpt.kv_bytes_per_token() != moe.kv_bytes_per_token():
        raise AssertionError("kv bytes per token differ between gpt and moe")
    router = moe.n_embd * moe.n_expert * moe.n_layer
    # The dense GPT already includes one expert-sized MLP per layer.
    expected_active = gpt.num_parameters() + one * moe.n_layer + router
    if active != expected_active:
        raise AssertionError(f"active {active} != dense plus one extra expert plus router {expected_active}")
    print(
        f"check counts  gpt {gpt.num_parameters():,}  "
        f"moe total {moe.num_parameters():,}  moe active {active:,}  "
        f"one expert {one:,}  router {router:,}"
    )


def _flat(moe, x):
    return x.reshape(-1, x.size(-1))


def check_gate_and_dispatch(device):
    """The sparse mix matches running every expert, and exactly two weights are set."""
    torch.manual_seed(0)
    moe = SparseMoE(n_embd=128, n_expert=8, top_k=2).to(device)
    x = torch.randn(2, 8, 128, device=device)
    flat = _flat(moe, x)
    _probs, top_idx, gates = moe.route(flat)
    full = moe.full_gates(top_idx, gates)
    nonzero = (full > 0).sum(dim=-1)
    zeros = (full == 0).sum(dim=-1)
    if not torch.equal(nonzero, torch.full((flat.size(0),), 2, device=device)):
        raise AssertionError(f"gate nonzero counts {nonzero.tolist()} are not all 2")
    if not torch.equal(zeros, torch.full((flat.size(0),), 6, device=device)):
        raise AssertionError("unselected experts are not exactly zero")
    weight_gap = (full.sum(dim=-1) - 1).abs().max().item()
    assert_gap("gate weight sum", weight_gap, AUX_LIMIT)

    sparse = moe.dispatch(flat, top_idx, gates)
    dense = moe.dense_combine(flat, top_idx, gates)
    mix_gap = (sparse - dense).abs().max().item()
    assert_gap("sparse dispatch", mix_gap)

    _out, aux, fractions = moe(x)
    assignment = F.one_hot(top_idx, num_classes=moe.n_expert).to(dtype=full.dtype)
    frequency = assignment.mean(dim=(0, 1))
    density = _probs.mean(dim=0)
    expected_aux = moe.n_expert * (frequency * density).sum()
    aux_gap = (aux - expected_aux).abs().item()
    assert_gap("aux loss", aux_gap, AUX_LIMIT)
    fraction_gap = (fractions - frequency).abs().max().item()
    assert_gap("expert fractions", fraction_gap, AUX_LIMIT)
    fraction_sum_gap = (fractions.sum() - 1).abs().item()
    assert_gap("expert fraction sum", fraction_sum_gap, AUX_LIMIT)
    print(
        f"check dispatch  device {device}  mix gap {mix_gap:.3e}  "
        f"aux gap {aux_gap:.3e}  gate sum gap {weight_gap:.3e}"
    )


def check_aux_values(device):
    """A flat router scores 1. A peaked router scores well above 1."""
    width = 8
    flat_router = SparseMoE(n_embd=width, n_expert=8, top_k=2).to(device)
    with torch.no_grad():
        flat_router.gate.weight.zero_()
    tokens = torch.randn(2, 4, width, device=device)
    _out, aux, fractions = flat_router(tokens)
    assert_near("uniform aux", aux.item(), 1.0, AUX_LIMIT)
    assert_near("uniform fraction sum", fractions.sum().item(), 1.0, AUX_LIMIT)

    peaked = SparseMoE(n_embd=width, n_expert=8, top_k=2).to(device)
    with torch.no_grad():
        peaked.gate.weight.zero_()
        peaked.gate.weight[0, 0] = 20.0
        peaked.gate.weight[1, 0] = 0.0
        for expert in range(2, 8):
            peaked.gate.weight[expert, 0] = -20.0
    probe = torch.zeros(4, 2, width, device=device)
    probe[..., 0] = 1
    _out, peaked_aux, _fractions = peaked(probe)
    if peaked_aux.item() <= 2:
        raise AssertionError(f"peaked aux {peaked_aux.item():.4f} is not above 2")
    print(f"check aux  uniform {aux.item():.4f}  peaked {peaked_aux.item():.4f}")


def _grad_norm(param):
    if param.grad is None:
        return None
    return param.grad.detach().abs().sum().item()


def check_expert_gradients(device):
    """Selected experts get a gradient. An expert with no tokens does not."""
    torch.manual_seed(0)
    width = 16
    covered = SparseMoE(n_embd=width, n_expert=8, top_k=2).to(device)
    with torch.no_grad():
        covered.gate.weight.zero_()
        for expert in range(8):
            covered.gate.weight[expert, expert] = 10.0
    tokens = torch.zeros(1, 8, width, device=device)
    tokens[0, torch.arange(8), torch.arange(8)] = 1
    mixed, _aux, _fractions = covered(tokens)
    mixed.sum().backward()
    for index, expert in enumerate(covered.experts):
        norm = _grad_norm(expert.c_fc.weight)
        if norm is None or norm == 0:
            raise AssertionError(f"covered expert {index} gradient is {norm}")

    unused = SparseMoE(n_embd=width, n_expert=8, top_k=2).to(device)
    with torch.no_grad():
        unused.gate.weight.zero_()
        unused.gate.weight[0, 0] = 10.0
        unused.gate.weight[1, 0] = 9.0
        for expert in range(2, 8):
            unused.gate.weight[expert, 0] = -10.0
    probe = torch.zeros(2, 4, width, device=device)
    probe[..., 0] = 1
    mixed, _aux, _fractions = unused(probe)
    mixed.sum().backward()
    for index, expert in enumerate(unused.experts):
        norm = _grad_norm(expert.c_fc.weight)
        if index < 2:
            if norm is None or norm == 0:
                raise AssertionError(f"selected expert {index} gradient is {norm}")
        elif norm not in (None, 0):
            raise AssertionError(f"unused expert {index} gradient is {norm}")
    print("check gradients  covered experts nonzero  unused experts silent")


def check_training(result, name):
    first = result["history"][0]["train_loss"]
    assert_near(f"{name} step-0 train loss", first, math.log(65), LOSS_BAND)
    for row in result["history"]:
        for key in ("train_loss", "val_loss", "train_aux", "val_aux"):
            if not math.isfinite(row[key]):
                raise AssertionError(f"{name} step {row['step']} {key} is not finite")
        for layer, fractions in enumerate(row["expert_fractions"]):
            total = sum(fractions)
            assert_near(f"{name} layer {layer} fraction sum", total, 1.0, 1e-4)
    if name == "gpt":
        for row in result["history"]:
            if row["train_aux"] != 0 or row["val_aux"] != 0:
                raise AssertionError(f"dense aux is not zero: {row}")
            if row["expert_fractions"]:
                raise AssertionError("dense model recorded expert fractions")
    print(
        f"check train {name}  steps {len(result['history'])} evals  "
        f"step 0 train {first:.4f}  "
        f"last train {result['history'][-1]['train_loss']:.4f}"
    )


def main():
    device = pick_device()
    gpt = check_cache(False, device)
    moe = check_cache(True, device)
    check_counts(gpt, moe)
    check_gate_and_dispatch(device)
    check_aux_values(device)
    check_expert_gradients(device)
    for model_name in ("gpt", "moe"):
        result = train(model_name=model_name, max_iters=SMOKE_STEPS, tag=f"{model_name}-smoke")
        check_training(result, model_name)
    print("smoke ok")


if __name__ == "__main__":
    main()
