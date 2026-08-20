import torch
from training.novelty_gate import NoveltyGate


def test_first_state_is_maximally_novel_with_empty_buffer():
    gate = NoveltyGate(dim=4, capacity=8, k=2)
    score = gate.score(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert score > 0.0  # no neighbors yet -> defined as maximal novelty, not zero/NaN


def test_repeated_state_has_lower_novelty_than_a_fresh_direction():
    gate = NoveltyGate(dim=4, capacity=8, k=2)
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    for _ in range(5):
        gate.push(v)
    repeated_score = gate.score(v)
    fresh_score = gate.score(torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert repeated_score < fresh_score


def test_buffer_respects_capacity():
    gate = NoveltyGate(dim=2, capacity=4, k=1)
    for i in range(10):
        gate.push(torch.tensor([float(i), 0.0]))
    assert len(gate.buffer) == 4
    assert gate.buffer[0][0].item() == 6.0  # oldest 6 entries evicted, FIFO
