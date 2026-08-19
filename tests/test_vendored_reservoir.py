import torch
from models.spiking_reservoir import SpikingReservoir
from models.baseline_transformer import Block


def test_reservoir_has_zero_trainable_parameters():
    res = SpikingReservoir(reservoir_size=256, input_dim=16, use_tensor_train=True,
                            tt_rank=4, tt_n_cores=2, seed=0)
    trainable = [p for p in res.parameters() if p.requires_grad]
    assert trainable == [], "vendored reservoir must have zero trainable parameters"


def test_reservoir_step_shapes_and_dtype():
    B, input_dim, N = 4, 16, 256
    res = SpikingReservoir(reservoir_size=N, input_dim=input_dim, use_tensor_train=True,
                            tt_rank=4, tt_n_cores=2, seed=0)
    mem = torch.zeros(B, N)
    spk = torch.zeros(B, N)
    x_t = torch.randn(B, input_dim)
    spk_next, mem_next = res.step(x_t, mem, spk)
    assert spk_next.shape == (B, N)
    assert mem_next.shape == (B, N)
    feat = res.readout_feature(spk_next, mem_next)
    assert feat.shape == (B, N)


def test_block_forward_shape():
    B, T, dim = 2, 8, 32
    block = Block(dim=dim, n_heads=4, context_len=16)
    x = torch.randn(B, T, dim)
    out = block(x)
    assert out.shape == (B, T, dim)
