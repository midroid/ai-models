"""Check the KV cache, then take 20 training steps on MPS, CUDA, or CPU.

    uv run python -m src.smoke

The cache check fails if prefill or decode logits differ from a full forward
by more than 1e-4, or if a CED prefill runs the decoder on more than the last
character. The training steps use the laptop width and the Tiny Shakespeare split.
"""

from src.model import build_model, verify_cache
from src.train import PRESETS, pick_device, train

GAP_LIMIT = 1e-4
SMOKE_STEPS = 20


def assert_gap(name, gap):
    if gap > GAP_LIMIT:
        raise AssertionError(f"{name} logit gap {gap:.3e} exceeds {GAP_LIMIT:.0e}")


def check_model(ced, device):
    config = dict(PRESETS["laptop"])
    config["ced"] = ced
    # Vocabulary size only affects the embedding and the head. 65 matches Tiny Shakespeare.
    model = build_model(65, config).to(device)
    model.eval()
    report = verify_cache(model)
    name = "ced" if ced else "gpt"
    n_prompt = report["n_prompt"]
    # GPT: every layer, every prompt character. CED: encoder on all of them,
    # decoder only on the last character (one token-layer per decoder layer).
    expected_encoder = model.n_encoder * n_prompt
    expected_decoder = model.n_decoder * 1
    if report["prefill_encoder_token_layers"] != expected_encoder:
        raise AssertionError(
            f"{name} prefill encoder token-layers "
            f"{report['prefill_encoder_token_layers']} != {expected_encoder}"
        )
    if report["prefill_decoder_token_layers"] != expected_decoder:
        raise AssertionError(
            f"{name} prefill decoder token-layers "
            f"{report['prefill_decoder_token_layers']} != {expected_decoder}"
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
        f"prefill token-layers enc {report['prefill_encoder_token_layers']} "
        f"dec {report['prefill_decoder_token_layers']}  "
        f"decode token-layers {report['decode_token_layers']}  "
        f"kv bytes/token {report['kv_bytes_per_token']}  "
        f"parameters {model.num_parameters():,}"
    )
    return model


def main():
    device = pick_device()
    gpt = check_model(False, device)
    ced = check_model(True, device)
    # The two modes own the same matrices. CED changes which tensor is multiplied
    # by the key and value weights, not how many weights there are. The cache
    # is also the same size: every layer still stores its own keys and values.
    if gpt.num_parameters() != ced.num_parameters():
        raise AssertionError(
            f"parameter counts differ: gpt {gpt.num_parameters()} ced {ced.num_parameters()}"
        )
    if gpt.kv_bytes_per_token() != ced.kv_bytes_per_token():
        raise AssertionError("kv bytes per token differ between gpt and ced")
    for model_name in ("gpt", "ced"):
        train(model_name=model_name, max_iters=SMOKE_STEPS, tag=f"{model_name}-smoke")
    print("smoke ok")


if __name__ == "__main__":
    main()
