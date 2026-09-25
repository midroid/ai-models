"""Model size, tying, shapes, causal mask, and a finite backward."""

import torch

from src.model import MODEL_10M, build_model, count_parameters, unique_parameters


def tiny_config():
    config = dict(MODEL_10M)
    config.update(
        {
            "vocab_size": 64,
            "n_embd": 32,
            "n_layer": 2,
            "n_head": 4,
            "n_kv_head": 2,
            "head_dim": 8,
            "intermediate": 64,
            "block_size": 32,
            "dropout": 0.0,
        }
    )
    return config


def test_parameter_count():
    assert count_parameters(MODEL_10M) == 10_942_720
    model = build_model(MODEL_10M)
    assert model.num_parameters() == 10_942_720
    assert model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr()


def test_attention_shapes_and_causal_mask():
    model = build_model(MODEL_10M)
    model.eval()
    inputs = torch.randint(0, MODEL_10M["vocab_size"], (2, 16))
    logits, _ = model(inputs)
    assert model.blocks[0].attn.last_q_shape == (2, 16, 8, 32)
    assert model.blocks[0].attn.last_k_shape == (2, 16, 2, 32)
    changed = inputs.clone()
    changed[:, 8:] = (changed[:, 8:] + 1) % MODEL_10M["vocab_size"]
    other, _ = model(changed)
    assert torch.allclose(logits[:, :8], other[:, :8], atol=1e-5)


def test_backward_is_finite():
    model = build_model(tiny_config())
    inputs = torch.randint(0, 64, (2, 8))
    _, loss = model(inputs, inputs.clone())
    loss.backward()
    assert torch.isfinite(loss)
    for param in unique_parameters(model):
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()
