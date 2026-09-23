"""Check both mixes, the KV cache, then take 20 training steps.

    uv run python -m src.smoke

The cache check fails if prefill or decode logits differ from a full forward
by more than 1e-4. The DeepSeekMoE checks fail if the active feed-forward
cost drifts from Mixtral, if the shared expert is routed, if sparse dispatch
disagrees with the dense mix, or if the auxiliary loss counts the shared expert.
"""

import math

import torch
from torch.nn import functional as F

from src.model import DeepSeekMoE, build_model, verify_cache
from src.train import FEED_FORWARDS, PRESETS, pick_device, train

GAP_LIMIT = 1e-4
AUX_LIMIT = 1e-4
SMOKE_STEPS = 20
# Step 0 of a fresh head sits on the uniform-guess loss, ln(65).
LOSS_BAND = 0.15
FULL_EXPERT = 131_072
HALF_EXPERT = 65_536
ACTIVE_FFN = 262_144
EXPERTS_PER_LAYER = 1_048_576
ROUTER_DELTA = (15 - 8) * 128 * 4


def assert_gap(name, gap, limit=GAP_LIMIT):
    if gap > limit:
        raise AssertionError(f"{name} gap {gap:.3e} exceeds {limit:.0e}")


def assert_near(name, value, target, limit):
    gap = abs(value - target)
    if gap > limit:
        raise AssertionError(f"{name} {value:.6f} is {gap:.3e} from {target:.6f}")


def model_config(name):
    config = dict(PRESETS["laptop"])
    spec = dict(FEED_FORWARDS[name])
    hidden_mult = spec.pop("hidden_mult")
    config.update(spec)
    config["hidden"] = hidden_mult * config["n_embd"]
    return config


def check_cache(name, device):
    # Vocabulary size only affects the embedding and the head. 65 matches Tiny Shakespeare.
    model = build_model(65, model_config(name)).to(device)
    model.eval()
    report = verify_cache(model)
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
        f"active {model.num_active_parameters():,}  "
        f"active ffn/layer {model.active_ffn_parameters():,}"
    )
    return model


def check_counts(mixtral, dsmoe):
    """Four half-width experts match two full experts. The router is the extra cost."""
    if mixtral.expert_parameter_count() != FULL_EXPERT:
        raise AssertionError(f"mixtral expert {mixtral.expert_parameter_count()} != {FULL_EXPERT}")
    if dsmoe.expert_parameter_count() != HALF_EXPERT:
        raise AssertionError(f"dsmoe expert {dsmoe.expert_parameter_count()} != {HALF_EXPERT}")
    if mixtral.active_ffn_parameters() != ACTIVE_FFN or dsmoe.active_ffn_parameters() != ACTIVE_FFN:
        raise AssertionError(
            f"active ffn/layer mixtral {mixtral.active_ffn_parameters()} "
            f"dsmoe {dsmoe.active_ffn_parameters()} != {ACTIVE_FFN}"
        )
    if mixtral.expert_parameters_per_layer() != EXPERTS_PER_LAYER:
        raise AssertionError("mixtral expert parameters per layer drifted")
    if dsmoe.expert_parameters_per_layer() != EXPERTS_PER_LAYER:
        raise AssertionError("dsmoe expert parameters per layer drifted")
    if (1 + 3) * HALF_EXPERT != 2 * FULL_EXPERT:
        raise AssertionError("half-width active count does not match two full experts")
    param_delta = dsmoe.num_parameters() - mixtral.num_parameters()
    active_delta = dsmoe.num_active_parameters() - mixtral.num_active_parameters()
    if param_delta != ROUTER_DELTA or active_delta != ROUTER_DELTA:
        raise AssertionError(
            f"parameter delta {param_delta} active delta {active_delta} != router {ROUTER_DELTA}"
        )
    if mixtral.num_parameters() != 4_477_184 or mixtral.num_active_parameters() != 1_331_456:
        raise AssertionError(
            f"mixtral counts {mixtral.num_parameters()} / {mixtral.num_active_parameters()}"
        )
    if dsmoe.num_parameters() != 4_480_768 or dsmoe.num_active_parameters() != 1_335_040:
        raise AssertionError(
            f"dsmoe counts {dsmoe.num_parameters()} / {dsmoe.num_active_parameters()}"
        )
    if mixtral.kv_bytes_per_token() != dsmoe.kv_bytes_per_token():
        raise AssertionError("kv bytes per token differ")
    if dsmoe.n_shared != 1 or dsmoe.n_routed != 15 or dsmoe.top_k != 3:
        raise AssertionError("dsmoe width is not 1 shared, 15 routed, top-3")
    print(
        f"check counts  mixtral {mixtral.num_parameters():,} active {mixtral.num_active_parameters():,}  "
        f"dsmoe {dsmoe.num_parameters():,} active {dsmoe.num_active_parameters():,}  "
        f"router delta {ROUTER_DELTA:,}"
    )


def check_gate_and_dispatch(device):
    """The shared expert is added at coefficient 1 and is absent from the router."""
    torch.manual_seed(0)
    moe = DeepSeekMoE(n_embd=128, n_routed=15, top_k=3, n_shared=1, hidden=256).to(device)
    if moe.routed.gate.out_features != 15:
        raise AssertionError("router does not have one row per routed expert")
    x = torch.randn(2, 8, 128, device=device)
    flat = x.reshape(-1, 128)
    _probs, top_idx, gates = moe.routed.route(flat)
    full = moe.routed.full_gates(top_idx, gates)
    if full.size(-1) != 15:
        raise AssertionError("gate vector includes something other than the routed experts")
    nonzero = (full > 0).sum(dim=-1)
    zeros = (full == 0).sum(dim=-1)
    if not torch.equal(nonzero, torch.full((flat.size(0),), 3, device=device)):
        raise AssertionError(f"gate nonzero counts {nonzero.tolist()} are not all 3")
    if not torch.equal(zeros, torch.full((flat.size(0),), 12, device=device)):
        raise AssertionError("unselected routed experts are not exactly zero")
    weight_gap = (full.sum(dim=-1) - 1).abs().max().item()
    assert_gap("gate weight sum", weight_gap, AUX_LIMIT)

    sparse = moe.routed.dispatch(flat, top_idx, gates)
    dense = moe.routed.dense_combine(flat, top_idx, gates)
    shared = moe.shared_sum(flat)
    mix_gap = ((sparse + shared) - (dense + shared)).abs().max().item()
    assert_gap("sparse dispatch", mix_gap)
    mixed, _aux, _fractions, _shared_fraction = moe(x)
    forward_gap = (mixed.reshape(-1, 128) - (sparse + shared)).abs().max().item()
    assert_gap("forward mix", forward_gap)

    with torch.no_grad():
        for param in moe.shared.parameters():
            param.zero_()
    zeroed, _, _, _ = moe(x)
    routed_only, _, _, _ = moe.routed(x)
    shared_gap = (zeroed - routed_only).abs().max().item()
    assert_gap("zeroed shared expert", shared_gap)
    print(
        f"check dispatch  device {device}  mix gap {mix_gap:.3e}  "
        f"forward gap {forward_gap:.3e}  zeroed-shared gap {shared_gap:.3e}"
    )


def check_aux_values(device):
    """A flat routed router scores 1. The shared expert is not in that score."""
    width = 16
    flat_router = DeepSeekMoE(n_embd=width, n_routed=15, top_k=3, n_shared=1, hidden=8).to(device)
    with torch.no_grad():
        flat_router.routed.gate.weight.zero_()
    tokens = torch.randn(2, 4, width, device=device)
    _out, aux, fractions, shared_fraction = flat_router(tokens)
    assert_near("uniform aux", aux.item(), 1.0, AUX_LIMIT)
    assert_near("uniform fraction sum", fractions.sum().item(), 1.0, AUX_LIMIT)
    assert_near("shared fraction", shared_fraction.item(), 1.0, AUX_LIMIT)
    if fractions.numel() != 15:
        raise AssertionError(f"routed fractions cover {fractions.numel()} experts")
    flat = tokens.reshape(-1, width)
    probs, top_idx, _gates = flat_router.routed.route(flat)
    assignment = F.one_hot(top_idx, num_classes=15).to(dtype=probs.dtype)
    expected = 15 * (assignment.mean(dim=(0, 1)) * probs.mean(dim=0)).sum()
    assert_gap("aux loss", (aux - expected).abs().item(), AUX_LIMIT)

    peaked = DeepSeekMoE(n_embd=width, n_routed=15, top_k=3, n_shared=1, hidden=8).to(device)
    with torch.no_grad():
        peaked.routed.gate.weight.zero_()
        peaked.routed.gate.weight[0, 0] = 20.0
        peaked.routed.gate.weight[1, 0] = 0.0
        for expert in range(2, 15):
            peaked.routed.gate.weight[expert, 0] = -20.0
    probe = torch.zeros(4, 2, width, device=device)
    probe[..., 0] = 1
    _out, peaked_aux, _fractions, _shared_fraction = peaked(probe)
    if peaked_aux.item() <= 2:
        raise AssertionError(f"peaked aux {peaked_aux.item():.4f} is not above 2")
    print(f"check aux  uniform {aux.item():.4f}  peaked {peaked_aux.item():.4f}")


def _grad_norm(param):
    if param.grad is None:
        return None
    return param.grad.detach().abs().sum().item()


def check_expert_gradients(device):
    """Selected routed experts and the shared expert get a gradient."""
    torch.manual_seed(0)
    width = 32
    covered = DeepSeekMoE(n_embd=width, n_routed=15, top_k=3, n_shared=1, hidden=8).to(device)
    with torch.no_grad():
        covered.routed.gate.weight.zero_()
        for expert in range(15):
            covered.routed.gate.weight[expert, expert] = 10.0
    tokens = torch.zeros(1, 15, width, device=device)
    tokens[0, torch.arange(15), torch.arange(15)] = 1
    mixed, _aux, _fractions, _shared = covered(tokens)
    mixed.sum().backward()
    for index, expert in enumerate(covered.routed.experts):
        norm = _grad_norm(expert.c_fc.weight)
        if norm is None or norm == 0:
            raise AssertionError(f"covered routed expert {index} gradient is {norm}")
    shared_norm = _grad_norm(covered.shared[0].c_fc.weight)
    if shared_norm is None or shared_norm == 0:
        raise AssertionError(f"shared expert gradient is {shared_norm}")

    unused = DeepSeekMoE(n_embd=width, n_routed=15, top_k=3, n_shared=1, hidden=8).to(device)
    with torch.no_grad():
        unused.routed.gate.weight.zero_()
        unused.routed.gate.weight[0, 0] = 10.0
        unused.routed.gate.weight[1, 0] = 9.0
        unused.routed.gate.weight[2, 0] = 8.0
        for expert in range(3, 15):
            unused.routed.gate.weight[expert, 0] = -10.0
    probe = torch.zeros(2, 4, width, device=device)
    probe[..., 0] = 1
    mixed, _aux, _fractions, _shared = unused(probe)
    mixed.sum().backward()
    for index, expert in enumerate(unused.routed.experts):
        norm = _grad_norm(expert.c_fc.weight)
        if index < 3:
            if norm is None or norm == 0:
                raise AssertionError(f"selected routed expert {index} gradient is {norm}")
        elif norm not in (None, 0):
            raise AssertionError(f"unused routed expert {index} gradient is {norm}")
    if _grad_norm(unused.shared[0].c_fc.weight) in (None, 0):
        raise AssertionError("shared expert was silent on a batch that still runs it")
    print("check gradients  covered experts nonzero  unused routed experts silent  shared live")


def check_training(result, name):
    first = result["history"][0]["train_loss"]
    assert_near(f"{name} step-0 train loss", first, math.log(65), LOSS_BAND)
    expect_shared = name == "dsmoe"
    for row in result["history"]:
        for key in ("train_loss", "val_loss", "train_aux", "val_aux"):
            if not math.isfinite(row[key]):
                raise AssertionError(f"{name} step {row['step']} {key} is not finite")
        for layer, fractions in enumerate(row["expert_fractions"]):
            assert_near(f"{name} layer {layer} fraction sum", sum(fractions), 1.0, 1e-4)
        if expect_shared:
            if len(row["shared_fractions"]) != 4:
                raise AssertionError(f"{name} shared fractions {row['shared_fractions']}")
            for layer, value in enumerate(row["shared_fractions"]):
                assert_near(f"{name} layer {layer} shared fraction", value, 1.0, 1e-4)
        elif row["shared_fractions"]:
            raise AssertionError("mixtral recorded a shared expert")
    print(
        f"check train {name}  steps {len(result['history'])} evals  "
        f"step 0 train {first:.4f}  "
        f"last train {result['history'][-1]['train_loss']:.4f}"
    )


def main():
    device = pick_device()
    mixtral = check_cache("mixtral", device)
    dsmoe = check_cache("dsmoe", device)
    check_counts(mixtral, dsmoe)
    check_gate_and_dispatch(device)
    check_aux_values(device)
    check_expert_gradients(device)
    for model_name in ("mixtral", "dsmoe"):
        result = train(model_name=model_name, max_iters=SMOKE_STEPS, tag=f"{model_name}-smoke")
        check_training(result, model_name)
    print("smoke ok")


if __name__ == "__main__":
    main()
