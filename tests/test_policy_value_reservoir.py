import pytest
import torch
import torch.nn as nn
from models.policy_value_reservoir import PolicyValueReservoir


def _small_model():
    return PolicyValueReservoir(obs_dim=12, embed_dim=16, reservoir_size=256,
                                 n_actions=10, use_tensor_train=True, tt_rank=4,
                                 tt_n_cores=2, context_len=8, seed=0)


def test_forward_shapes_and_state_threading():
    B = 3
    model = _small_model()
    mem, spk, window = model.init_state(B, device=torch.device("cpu"))
    obs = torch.randn(B, 12)
    logits, value, mem2, spk2, window2 = model(obs, mem, spk, window)
    assert logits.shape == (B, 10)
    assert value.shape == (B,)
    assert mem2.shape == (B, 256)
    assert spk2.shape == (B, 256)
    assert window2.shape[0] == B and window2.shape[2] == 256
    assert window2.shape[1] <= model.context_len


def test_window_grows_then_caps_at_context_len():
    B = 1
    model = _small_model()
    mem, spk, window = model.init_state(B, device=torch.device("cpu"))
    obs = torch.randn(B, 12)
    for step in range(model.context_len + 5):
        logits, value, mem, spk, window = model(obs, mem, spk, window)
        assert window.shape[1] == min(step + 1, model.context_len)


def test_reservoir_stays_frozen_across_a_training_step():
    model = _small_model()
    w_in_before = model.reservoir.W_in.clone()
    opt = torch.optim.Adam(model.trainable_parameters(), lr=1e-2)
    mem, spk, window = model.init_state(2, device=torch.device("cpu"))
    obs = torch.randn(2, 12)
    logits, value, mem, spk, window = model(obs, mem, spk, window)
    loss = logits.sum() + value.sum()
    opt.zero_grad()
    loss.backward()
    # The frozen-reservoir assertions below cannot, on their own, tell "the embedding
    # is training normally" apart from "the embedding is silently dead": the ~30
    # readout tensors receive gradient straight from logits.sum() in either case, so
    # both scenarios would pass identically. Pin the surrogate-gradient path THROUGH
    # the frozen reservoir explicitly. If a future detach/no_grad for PPO rollout
    # collection (Task 8/11) ever lands inside PolicyValueReservoir.forward -- on
    # `feat` or `spk_next` -- instead of in the trainer, the trainable embedding
    # becomes dead weight; this makes that fail loudly instead of passing silently.
    assert model.embedding.weight.grad is not None, (
        "no gradient reached the trainable embedding -- the path through the frozen "
        "reservoir is broken (detach/no_grad inside forward?)"
    )
    assert model.embedding.weight.grad.abs().sum() > 0, (
        "embedding gradient is identically zero -- the embedding is dead weight"
    )
    opt.step()
    model.assert_reservoir_frozen()
    assert torch.equal(model.reservoir.W_in, w_in_before), "reservoir W_in must never change"


def test_frozen_reservoir_tripwire_actually_catches_a_violation():
    """Mutation test for the guard above. A tripwire that cannot fail proves nothing,
    so promote the frozen W_in buffer to a trainable nn.Parameter and confirm BOTH
    halves of the invariant fire: assert_reservoir_frozen() must raise, and W_in must
    visibly drift across a training step."""
    model = _small_model()
    w_in = model.reservoir.W_in.clone()
    del model.reservoir.W_in  # drop the buffer registration
    model.reservoir.W_in = nn.Parameter(w_in.clone())  # ...and re-add it as trainable
    w_in_before = model.reservoir.W_in.detach().clone()

    opt = torch.optim.Adam(model.trainable_parameters(), lr=1e-2)
    mem, spk, window = model.init_state(2, device=torch.device("cpu"))
    obs = torch.randn(2, 12)
    logits, value, mem, spk, window = model(obs, mem, spk, window)
    opt.zero_grad()
    (logits.sum() + value.sum()).backward()
    opt.step()

    # Half 1: the zero-nn.Parameter invariant must reject the promoted weight.
    with pytest.raises(AssertionError, match="zero nn.Parameters"):
        model.assert_reservoir_frozen()
    # Half 2: the bit-identity check must independently catch the numeric drift.
    assert not torch.equal(model.reservoir.W_in.detach(), w_in_before), (
        "W_in did not change even as a trainable Parameter -- the bit-identity half "
        "of the tripwire would not catch a real violation"
    )
