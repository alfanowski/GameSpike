import torch
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
    opt.step()
    model.assert_reservoir_frozen()
    assert torch.equal(model.reservoir.W_in, w_in_before), "reservoir W_in must never change"
