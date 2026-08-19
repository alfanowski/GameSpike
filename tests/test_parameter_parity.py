from models.policy_value_gru import PolicyValueGRU
from models.policy_value_reservoir import PolicyValueReservoir

TOLERANCE = 0.10  # spec §5: matched trainable-parameter budget, within 10%


def test_baseline_and_reservoir_arms_have_matched_trainable_parameter_counts():
    gru = PolicyValueGRU(obs_dim=12, embed_dim=32, hidden_dim=192, n_actions=10)
    reservoir_model = PolicyValueReservoir(obs_dim=12, embed_dim=32, reservoir_size=8192,
                                            n_actions=10, use_tensor_train=True, tt_rank=8,
                                            tt_n_cores=4, context_len=64, seed=0)
    gru_count = gru.trainable_parameter_count()
    res_count = reservoir_model.trainable_parameter_count()
    ratio = res_count / gru_count
    assert (1 - TOLERANCE) <= ratio <= (1 + TOLERANCE), (
        f"trainable parameter counts diverge beyond {TOLERANCE:.0%}: "
        f"GRU={gru_count}, reservoir-arm={res_count}, ratio={ratio:.3f}. "
        "Adjust d_model/n_layers in ActorCriticReadout or hidden_dim in PolicyValueGRU "
        "to rebalance -- this is a hard requirement (spec §5), not a nice-to-have."
    )
