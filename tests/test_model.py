"""Unit tests for SHUTEN-DŌJI model architecture."""

import pytest
import torch

from src.model.architecture import ShutenDojiConfig, ShutenDojiModel


@pytest.fixture
def mini_config():
    """Tiny config for unit testing (not the real model)."""
    return ShutenDojiConfig(
        hidden_size=256,
        intermediate_size=512,
        num_layers=4,
        vocab_size=1000,
        num_attention_heads=8,
        num_kv_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        num_experts=8,
        num_shared_experts=1,
        num_experts_per_token=2,
        expert_hidden_size=128,
        num_memory_slots=16,
        memory_dim=64,
        num_token_types=9,
        num_dense_layers=1,
    )


@pytest.fixture
def model(mini_config):
    return ShutenDojiModel(mini_config)


def test_model_creation(model, mini_config):
    assert model is not None
    params = model.num_parameters()
    assert params > 0
    assert params < 1e9  # should be small for test config


def test_forward_pass(model, mini_config):
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, mini_config.vocab_size, (batch_size, seq_len))

    output = model(input_ids)
    assert "logits" in output
    assert output["logits"].shape == (batch_size, seq_len, mini_config.vocab_size)


def test_forward_with_labels(model, mini_config):
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, mini_config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, mini_config.vocab_size, (batch_size, seq_len))

    output = model(input_ids, labels=labels)
    assert output["loss"] is not None
    assert output["loss"].dim() == 0  # scalar
    assert output["loss"].item() > 0


def test_forward_with_token_types(model, mini_config):
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, mini_config.vocab_size, (batch_size, seq_len))
    token_type_ids = torch.randint(0, mini_config.num_token_types, (batch_size, seq_len))

    output = model(input_ids, token_type_ids=token_type_ids)
    assert output["logits"].shape == (batch_size, seq_len, mini_config.vocab_size)


def test_aux_loss_nonzero(model, mini_config):
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, mini_config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, mini_config.vocab_size, (batch_size, seq_len))

    output = model(input_ids, labels=labels)
    assert output["aux_loss"].item() >= 0  # router loss should be non-negative


def test_config_param_estimates(mini_config):
    total = mini_config.total_params_estimate
    active = mini_config.active_params_estimate
    assert total > 0
    assert active > 0
    assert active <= total


def test_gradient_flow(model, mini_config):
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, mini_config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, mini_config.vocab_size, (batch_size, seq_len))

    output = model(input_ids, labels=labels)
    output["loss"].backward()

    # Check that at least some gradients are non-zero
    has_grad = False
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad
