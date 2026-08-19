import torch
from models.policy_value_gru import PolicyValueGRU


def test_forward_shapes():
    B = 4
    model = PolicyValueGRU(obs_dim=12, embed_dim=32, hidden_dim=192, n_actions=10)
    h = model.init_hidden(B, device=torch.device("cpu"))
    obs = torch.randn(B, 12)
    logits, value, h_next = model(obs, h)
    assert logits.shape == (B, 10)
    assert value.shape == (B,)
    assert h_next.shape == (1, B, 192)


def test_hidden_state_persists_across_steps():
    model = PolicyValueGRU(obs_dim=12, embed_dim=32, hidden_dim=192, n_actions=10)
    h = model.init_hidden(1, device=torch.device("cpu"))
    obs = torch.randn(1, 12)
    _, _, h1 = model(obs, h)
    _, _, h2 = model(obs, h1)
    assert not torch.allclose(h1, h2), "hidden state should evolve across steps given a recurrent model"


def test_all_parameters_are_trainable():
    model = PolicyValueGRU(obs_dim=12, embed_dim=32, hidden_dim=192, n_actions=10)
    assert model.trainable_parameter_count() == sum(p.numel() for p in model.parameters())
