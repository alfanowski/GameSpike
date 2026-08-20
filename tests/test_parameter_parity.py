"""Spec §5's binding constraint: the two arms' TRAINABLE parameter counts match.

The models under test are the ones `training.train.build_model` actually builds,
not locally-constructed copies with the same hyperparameters written out a second
time. That distinction is the whole point of this file: an earlier version
instantiated its own `PolicyValueGRU(hidden_dim=192)` / `PolicyValueReservoir(...)`
here, so changing `build_model`'s `hidden_dim` or `d_model` would have taken the
real training arms out of parity while this test stayed green -- it was checking
that two constants in this file matched each other, not that the experiment was
controlled.
"""
from training.train import build_model

TOLERANCE = 0.10  # spec §5: matched trainable-parameter budget, within 10%


def _trainable_count(arm: str) -> int:
    """The arm exactly as the training loop builds it. The optimizer is discarded:
    it is `build_model`'s other return value, not part of what is being measured."""
    model, _optimizer = build_model(arm)
    return model.trainable_parameter_count()


def test_baseline_and_reservoir_arms_have_matched_trainable_parameter_counts():
    gru_count = _trainable_count("baseline")
    res_count = _trainable_count("reservoir")
    ratio = res_count / gru_count
    assert (1 - TOLERANCE) <= ratio <= (1 + TOLERANCE), (
        f"trainable parameter counts diverge beyond {TOLERANCE:.0%}: "
        f"GRU={gru_count}, reservoir-arm={res_count}, ratio={ratio:.3f}. "
        "Adjust d_model/n_layers in ActorCriticReadout or hidden_dim in PolicyValueGRU "
        "to rebalance -- this is a hard requirement (spec §5), not a nice-to-have."
    )


def test_parity_holds_at_every_training_seed():
    """Seeding varies the reservoir arm's frozen weights (and both arms' trainable
    init), so parity has to be a property of the architecture rather than of seed 0
    -- otherwise a multi-seed §5 comparison could silently run unmatched arms."""
    counts = {
        seed: (build_model("baseline", seed=seed)[0].trainable_parameter_count(),
               build_model("reservoir", seed=seed)[0].trainable_parameter_count())
        for seed in (0, 1, 7)
    }
    assert len(set(counts.values())) == 1, (
        f"trainable parameter counts depend on the seed: {counts}"
    )
