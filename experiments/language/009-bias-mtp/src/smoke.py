"""Check the bias router, the second-token head, then take 20 training steps.

    uv run python -m src.smoke

The cache check fails if prefill or decode logits differ from a full forward
by more than 1e-4. The bias check fails if the buffer is a parameter, if a
zero bias does not reproduce top-3 on the raw logits, or if one forced
assignment does not move the bias by 0.001. The training steps use the laptop
width and the Tiny Shakespeare split.
"""

import math

import torch
from torch.nn import functional as F

from src.model import DeepSeekMoE, SparseMoE, build_model, verify_cache
from src.train import FEED_FORWARD, MODES, PRESETS, pick_device, train

GAP_LIMIT = 1e-4
AUX_LIMIT = 1e-4
SMOKE_STEPS = 20
# Step 0 of a fresh head sits on the uniform-guess loss, ln(65).
LOSS_BAND = 0.15
INFERENCE_PARAMETERS = 4_480_768
ACTIVE_PARAMETERS = 1_335_040
MTP_PROJ = 32_768
MTP_BLOCK = 196_608
MTP_HEAD = 8_320
MTP_PARAMETERS = MTP_PROJ + MTP_BLOCK + MTP_HEAD
MTP_TRAINING_PARAMETERS = INFERENCE_PARAMETERS + MTP_PARAMETERS
BIAS_STEP = 0.001


def assert_gap(name, gap, limit=GAP_LIMIT):
    if gap > limit:
        raise AssertionError(f"{name} gap {gap:.3e} exceeds {limit:.0e}")


def assert_near(name, value, target, limit):
    gap = abs(value - target)
    if gap > limit:
        raise AssertionError(f"{name} {value:.6f} is {gap:.3e} from {target:.6f}")


def model_config(name):
    config = dict(PRESETS["laptop"])
    spec = dict(FEED_FORWARD)
    hidden_mult = spec.pop("hidden_mult")
    config.update(spec)
    config.update(MODES[name])
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
        f"inference {model.num_inference_parameters():,}  "
        f"active {model.num_active_parameters():,}"
    )
    return model


def check_counts(dsmoe, bias, mtp):
    """Inference counts stay on the experiment 007 model. The extra head is training-only."""
    for name, model in (("dsmoe", dsmoe), ("bias", bias), ("mtp", mtp)):
        if model.num_inference_parameters() != INFERENCE_PARAMETERS:
            raise AssertionError(
                f"{name} inference {model.num_inference_parameters()} != {INFERENCE_PARAMETERS}"
            )
        if model.num_active_parameters() != ACTIVE_PARAMETERS:
            raise AssertionError(
                f"{name} active {model.num_active_parameters()} != {ACTIVE_PARAMETERS}"
            )
        if model.kv_bytes_per_token() != 4096:
            raise AssertionError(f"{name} kv bytes {model.kv_bytes_per_token()} != 4096")
    if dsmoe.num_parameters() != INFERENCE_PARAMETERS or bias.num_parameters() != INFERENCE_PARAMETERS:
        raise AssertionError("dsmoe or bias counts the bias buffer as a parameter")
    if mtp.mtp_parameter_count() != MTP_PARAMETERS:
        raise AssertionError(f"mtp extra {mtp.mtp_parameter_count()} != {MTP_PARAMETERS}")
    if mtp.num_parameters() != MTP_TRAINING_PARAMETERS:
        raise AssertionError(f"mtp training {mtp.num_parameters()} != {MTP_TRAINING_PARAMETERS}")
    if mtp.mtp.proj.weight.numel() != MTP_PROJ or mtp.mtp_head.weight.numel() != MTP_HEAD:
        raise AssertionError("mtp projection or head width drifted")
    block = sum(p.numel() for p in mtp.mtp.attn.parameters()) + sum(p.numel() for p in mtp.mtp.mlp.parameters())
    if block != MTP_BLOCK:
        raise AssertionError(f"mtp block {block} != {MTP_BLOCK}")
    print(
        f"check counts  inference {INFERENCE_PARAMETERS:,}  active {ACTIVE_PARAMETERS:,}  "
        f"mtp training {MTP_TRAINING_PARAMETERS:,}"
    )


def check_bias(device):
    """The bias changes who is selected and does not scale the gate."""
    torch.manual_seed(0)
    moe = DeepSeekMoE(n_embd=32, n_routed=15, top_k=3, n_shared=1, hidden=8, use_bias=True).to(device)
    routed = moe.routed
    if routed.bias.numel() != 15:
        raise AssertionError(f"bias width {routed.bias.numel()} != 15")
    if routed.bias.requires_grad:
        raise AssertionError("bias buffer requires a gradient")
    if any(name.endswith("bias") for name, _param in moe.named_parameters()):
        raise AssertionError("bias is registered as a parameter")
    before = moe.routed.bias.detach().clone()
    routed.bias.add_(1)
    if sum(p.numel() for p in moe.parameters()) != sum(p.numel() for p in DeepSeekMoE(
        n_embd=32, n_routed=15, top_k=3, n_shared=1, hidden=8, use_bias=True
    ).parameters()):
        raise AssertionError("a nonzero bias changed the parameter count")
    routed.bias.copy_(before)

    flat = torch.randn(4, 32, device=device)
    logits = routed.gate(flat)
    _probs, top_idx, gates = routed.route(flat)
    _raw_val, raw_idx = torch.topk(logits, routed.top_k, dim=-1)
    if not torch.equal(top_idx, raw_idx):
        raise AssertionError("a zero bias did not reproduce top-3 on the raw logits")
    full = routed.full_gates(top_idx, gates)
    if not torch.equal((full > 0).sum(dim=-1), torch.full((flat.size(0),), 3, device=device)):
        raise AssertionError("gate does not keep exactly three experts")
    if not torch.equal((full == 0).sum(dim=-1), torch.full((flat.size(0),), 12, device=device)):
        raise AssertionError("unselected experts are not exactly zero")
    assert_gap("gate weight sum", (full.sum(dim=-1) - 1).abs().max().item(), AUX_LIMIT)
    if moe.routed.gate.out_features != 15:
        raise AssertionError("the shared expert has a router row")

    with torch.no_grad():
        routed.bias.zero_()
        routed.bias[0] = 50
    _probs, biased_idx, biased_gates = routed.route(flat)
    if not (biased_idx == 0).any(dim=-1).all():
        raise AssertionError("a large bias did not bring expert 0 into the top-3")
    chosen = logits.gather(-1, biased_idx)
    raw_gates = F.softmax(chosen, dim=-1)
    assert_gap("raw-logit gate", (biased_gates - raw_gates).abs().max().item())
    scored = F.softmax(chosen + routed.bias[biased_idx], dim=-1)
    if (biased_gates - scored).abs().max().item() < 1e-3:
        raise AssertionError("the gate absorbed the bias")
    print("check bias  zero matches raw top-3  nonzero bias leaves the gate on the raw logits")


def check_bias_update(device):
    """Every token on expert 0 lowers that bias and raises the others."""
    routed = SparseMoE(n_embd=16, n_expert=15, top_k=3, hidden=8, use_bias=True).to(device)
    frequency = torch.zeros(15, device=device)
    frequency[0] = 1
    routed.apply_bias_update(frequency, BIAS_STEP)
    assert_near("overloaded bias", routed.bias[0].item(), -BIAS_STEP, 1e-6)
    for index in range(1, 15):
        assert_near(f"starved bias {index}", routed.bias[index].item(), BIAS_STEP, 1e-6)
    print(f"check bias update  expert 0 {routed.bias[0].item():.4f}  others {routed.bias[1].item():.4f}")


def check_mtp_grad(device):
    """The second-token loss trains its head. Prefill still walks only the main stack."""
    torch.manual_seed(0)
    model = build_model(65, model_config("mtp")).to(device)
    model.train()
    x = torch.randint(0, 65, (2, 8), device=device)
    y = torch.randint(0, 65, (2, 8), device=device)
    z = torch.randint(0, 65, (2, 8), device=device)
    _logits, main_loss, _aux, mtp_loss = model(x, y, z)
    (main_loss + 0.1 * mtp_loss).backward()
    grad = model.mtp_head.weight.grad
    if grad is None or grad.abs().sum().item() == 0:
        raise AssertionError("second-token head received no gradient")
    model.eval()
    report = verify_cache(model)
    if report["decode_token_layers"] != model.n_layer:
        raise AssertionError("prefill or decode entered the second-token block")
    print(f"check mtp grad  head {grad.abs().sum().item():.3e}  decode token-layers {report['decode_token_layers']}")


def check_training(result, name):
    first = result["history"][0]["train_loss"]
    assert_near(f"{name} step-0 train loss", first, math.log(65), LOSS_BAND)
    if name == "mtp":
        mtp_first = result["history"][0]["train_mtp"]
        assert_near(f"{name} step-0 mtp loss", mtp_first, math.log(65), LOSS_BAND)
    coef = result["config"]["aux_loss_coef"]
    if name == "dsmoe" and coef != 0.01:
        raise AssertionError(f"dsmoe aux coef {coef} != 0.01")
    if name != "dsmoe" and coef != 0:
        raise AssertionError(f"{name} still adds the Switch term ({coef})")
    for row in result["history"]:
        keys = ["train_loss", "val_loss", "train_aux", "val_aux"]
        if name == "mtp":
            keys += ["train_mtp", "val_mtp"]
        for key in keys:
            if not math.isfinite(row[key]):
                raise AssertionError(f"{name} step {row['step']} {key} is not finite")
        for layer, fractions in enumerate(row["expert_fractions"]):
            assert_near(f"{name} layer {layer} fraction sum", sum(fractions), 1.0, 1e-4)
        if len(row["shared_fractions"]) != 4:
            raise AssertionError(f"{name} shared fractions {row['shared_fractions']}")
        for layer, value in enumerate(row["shared_fractions"]):
            assert_near(f"{name} layer {layer} shared fraction", value, 1.0, 1e-4)
    if name == "dsmoe":
        print(
            f"check train {name}  steps {len(result['history'])} evals  "
            f"step 0 train {first:.4f}  "
            f"last train {result['history'][-1]['train_loss']:.4f}"
        )
        return
    if result["history"][0]["bias_min"] != 0 or result["history"][0]["bias_max"] != 0:
        raise AssertionError(f"{name} step-0 bias is not zero")
    print(
        f"check train {name}  steps {len(result['history'])} evals  "
        f"step 0 train {first:.4f}  "
        f"last train {result['history'][-1]['train_loss']:.4f}"
    )


def main():
    device = pick_device()
    dsmoe = check_cache("dsmoe", device)
    bias = check_cache("bias", device)
    mtp = check_cache("mtp", device)
    check_counts(dsmoe, bias, mtp)
    check_bias(device)
    check_bias_update(device)
    check_mtp_grad(device)
    results = {}
    for model_name in ("dsmoe", "bias", "mtp"):
        result = train(model_name=model_name, max_iters=SMOKE_STEPS, tag=f"{model_name}-smoke")
        check_training(result, model_name)
        results[model_name] = result
    dsmoe_loss = results["dsmoe"]["history"][0]["train_loss"]
    bias_loss = results["bias"]["history"][0]["train_loss"]
    assert_near("bias step-0 train loss", bias_loss, dsmoe_loss, 1e-4)
    print("smoke ok")


if __name__ == "__main__":
    main()
