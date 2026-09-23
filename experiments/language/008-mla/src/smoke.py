"""Check latent attention, the KV cache, then take 20 training steps.

    uv run python -m src.smoke

The cache check fails if prefill or decode logits differ from a full forward
by more than 1e-4. The latent checks fail if the cache is not 768 bytes, if
up-projecting a cached latent disagrees with a direct projection, or if the
content key changes with position. The training steps use the laptop width
and the Tiny Shakespeare split.
"""

import math

import torch

from src.model import LatentKVCache, build_model, verify_cache
from src.train import PRESETS, pick_device, train

GAP_LIMIT = 1e-4
SMOKE_STEPS = 20
# Step 0 of a fresh head sits on the uniform-guess loss, ln(65).
LOSS_BAND = 0.15
# Dense GPT from experiment 006. Latent attention drops 10,240 parameters per
# layer: 65,536 in the four multi-head projections, 55,296 in the low-rank map.
MHA_PARAMETERS = 803_072
MLA_PARAMETERS = 762_112
MHA_KV_BYTES = 4_096
MLA_KV_BYTES = 768


def assert_gap(name, gap, limit=GAP_LIMIT):
    if gap > limit:
        raise AssertionError(f"{name} gap {gap:.3e} exceeds {limit:.0e}")


def assert_near(name, value, target, limit):
    gap = abs(value - target)
    if gap > limit:
        raise AssertionError(f"{name} {value:.6f} is {gap:.3e} from {target:.6f}")


def check_cache(mla, device):
    config = dict(PRESETS["laptop"])
    config["mla"] = mla
    # Vocabulary size only affects the embedding and the head. 65 matches Tiny Shakespeare.
    model = build_model(65, config).to(device)
    model.eval()
    report = verify_cache(model)
    name = "mla" if mla else "mha"
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
        f"parameters {model.num_parameters():,}"
    )
    return model


def check_counts(mha, mla):
    """Latent attention keeps fewer parameters and a smaller cache."""
    if mha.num_parameters() != MHA_PARAMETERS:
        raise AssertionError(f"mha parameters {mha.num_parameters()} != {MHA_PARAMETERS}")
    if mla.num_parameters() != MLA_PARAMETERS:
        raise AssertionError(f"mla parameters {mla.num_parameters()} != {MLA_PARAMETERS}")
    if mha.kv_bytes_per_token() != MHA_KV_BYTES:
        raise AssertionError(f"mha kv bytes {mha.kv_bytes_per_token()} != {MHA_KV_BYTES}")
    if mla.kv_bytes_per_token() != MLA_KV_BYTES:
        raise AssertionError(f"mla kv bytes {mla.kv_bytes_per_token()} != {MLA_KV_BYTES}")
    if not mla.mla or mha.mla:
        raise AssertionError("mode flag does not match the model")
    print(
        f"check counts  mha {mha.num_parameters():,} params  {mha.kv_bytes_per_token()} bytes  "
        f"mla {mla.num_parameters():,} params  {mla.kv_bytes_per_token()} bytes"
    )


def check_latent(model):
    """The cached latent up-projects to the same K and V as a direct projection.

    Content keys ignore position. The shared rotary key does not.
    """
    if not model.mla:
        raise AssertionError("latent check requires the mla model")
    attn = model.blocks[0].attn
    device = next(model.parameters()).device
    cos, sin = model.cos_rope, model.sin_rope

    x = torch.ones(1, 1, model.n_embd, device=device)
    _c0, k0, _v0, r0 = attn.keys_and_values(x, slice(0, 1), cos, sin)
    _c1, k1, _v1, r1 = attn.keys_and_values(x, slice(3, 4), cos, sin)
    content_gap = (k0 - k1).abs().max().item()
    rope_gap = (r0 - r1).abs().max().item()
    if content_gap != 0:
        raise AssertionError(f"content key changed with position by {content_gap:.3e}")
    if rope_gap < 1e-3:
        raise AssertionError(f"rotary key did not change with position (gap {rope_gap:.3e})")

    batch, time = 2, 5
    states = torch.randn(batch, time, model.n_embd, device=device)
    c_kv, k_content, value, k_rope = attn.keys_and_values(states, slice(0, time), cos, sin)
    cache = LatentKVCache(
        n_layer=model.n_layer,
        batch=batch,
        max_seq=model.block_size,
        kv_rank=model.kv_rank,
        rope_dim=model.rope_dim,
        device=device,
        dtype=states.dtype,
    )
    attn.write_kv(states, cache, 0, cos, sin)
    if cache.c_kv.shape[-1] != model.kv_rank or cache.k_rope.shape[-1] != model.rope_dim:
        raise AssertionError(
            f"cache widths {cache.c_kv.shape[-1]} and {cache.k_rope.shape[-1]} "
            f"!= {model.kv_rank} and {model.rope_dim}"
        )
    # One token is the latent plus the shared key. No per-head values are stored.
    stored = model.n_layer * (cache.c_kv[0, 0, 0].numel() + cache.k_rope[0, 0, 0].numel()) * 4
    if stored != MLA_KV_BYTES:
        raise AssertionError(f"stored bytes {stored} != {MLA_KV_BYTES}")
    latent_gap = (cache.c_kv[attn.layer_idx, :, :time] - c_kv).abs().max().item()
    k_read, v_read, rope_read = attn.read_kv(cache, time)
    assert_gap("latent", latent_gap)
    assert_gap("content key", (k_read - k_content).abs().max().item())
    assert_gap("value", (v_read - value).abs().max().item())
    assert_gap("rotary key", (rope_read - k_rope).abs().max().item())
    print(
        f"check latent  content position gap {content_gap:.3e}  "
        f"rope position gap {rope_gap:.3e}  "
        f"reconstruct gap {max(latent_gap, (k_read - k_content).abs().max().item()):.3e}"
    )


def check_training(result, name):
    first = result["history"][0]["train_loss"]
    assert_near(f"{name} step-0 train loss", first, math.log(65), LOSS_BAND)
    for row in result["history"]:
        for key in ("train_loss", "val_loss"):
            if not math.isfinite(row[key]):
                raise AssertionError(f"{name} step {row['step']} {key} is not finite")
    expected_bytes = MLA_KV_BYTES if name == "mla" else MHA_KV_BYTES
    expected_parameters = MLA_PARAMETERS if name == "mla" else MHA_PARAMETERS
    if result["kv_bytes_per_token"] != expected_bytes:
        raise AssertionError(f"{name} kv bytes {result['kv_bytes_per_token']} != {expected_bytes}")
    if result["num_parameters"] != expected_parameters:
        raise AssertionError(f"{name} parameters {result['num_parameters']} != {expected_parameters}")
    print(
        f"check train {name}  steps {len(result['history'])} evals  "
        f"step 0 train {first:.4f}  "
        f"last train {result['history'][-1]['train_loss']:.4f}"
    )


def main():
    device = pick_device()
    mha = check_cache(False, device)
    mla = check_cache(True, device)
    check_counts(mha, mla)
    check_latent(mla)
    for model_name in ("mha", "mla"):
        result = train(model_name=model_name, max_iters=SMOKE_STEPS, tag=f"{model_name}-smoke")
        check_training(result, model_name)
    print("smoke ok")


if __name__ == "__main__":
    main()
