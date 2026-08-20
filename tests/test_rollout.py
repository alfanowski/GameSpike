"""Rollout mechanics.

Two halves, with two different reasons to exist:

  * The multi-process random-policy collector, against a real ROM. Random-policy
    only, deliberately: this pins down the multiprocessing/shape contract in
    isolation from any model.
  * The model-driven collector's episode-boundary bookkeeping, against stub
    env/model objects. These deliberately need NO ROM: terminated-vs-truncated
    handling and the truncation bootstrap are decided by branch logic, and a real
    emulator cannot be made to truncate on cue without running 3000 steps, so
    stubs test the contract far more sharply than the real thing could.
"""
import os

import numpy as np
import pytest
import torch

from envs.mario_land_env import OBS_DIM
from training.novelty_gate import NoveltyGate
from training.ppo import compute_gae
from training.rollout import collect_rollout_random_policy, collect_rollout_with_model

ROM_PATH = os.environ.get("MARIO_LAND_ROM_PATH")
requires_rom = pytest.mark.skipif(
    not ROM_PATH or not os.path.exists(ROM_PATH),
    reason="MARIO_LAND_ROM_PATH not set or file missing; set it to your own legally-dumped ROM",
)

N_ACTIONS = 10
STUB_VALUE = 7.0  # every critic call returns this, so a bootstrap is identifiable


class _StubEnv:
    """Env whose episode ends on cue, so termination and truncation can be told
    apart. Emits a distinct observation per step (step index in slot 0) so a test
    can prove WHICH observation a bootstrap or a novelty push was taken from."""

    def __init__(self, terminate_at=None, truncate_at=None):
        self.terminate_at = terminate_at
        self.truncate_at = truncate_at
        self.t = 0
        self.reset_calls = 0
        self.closed = False

    def _obs(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0] = float(self.t)
        return obs

    def reset(self):
        self.reset_calls += 1
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        terminated = self.t == self.terminate_at
        truncated = self.t == self.truncate_at
        return self._obs(), 1.0, terminated, truncated, {}

    def close(self):
        self.closed = True


def _stub_state_fns():
    """(init_state_fn, step_fn) matching the real models' contract: state is a
    tuple, step_fn returns (logits, value, *next_state)."""
    def init_state_fn(batch_size, device):
        return (torch.zeros(batch_size, 1, device=device),)

    def step_fn(model, obs, state):
        logits = torch.zeros(obs.shape[0], N_ACTIONS)
        value = torch.full((obs.shape[0],), STUB_VALUE)
        return logits, value, state[0]

    return init_state_fn, step_fn


def _collect(env, n_steps, novelty_gate=None, novelty_coef=0.0):
    obs, _ = env.reset()
    return collect_rollout_with_model(
        env=env, obs=obs, model=None, model_state_fns=_stub_state_fns(),
        n_steps=n_steps, novelty_gate=novelty_gate or NoveltyGate(dim=OBS_DIM, capacity=8, k=2),
        novelty_coef=novelty_coef,
    )


@requires_rom
def test_collect_rollout_shapes():
    data = collect_rollout_random_policy(ROM_PATH, n_envs=2, n_steps=16)
    assert data["obs"].shape == (2, 16, 12)
    assert data["actions"].shape == (2, 16)
    assert data["rewards"].shape == (2, 16)
    assert data["dones"].shape == (2, 16)
    assert data["actions"].max() < 10 and data["actions"].min() >= 0


# --------------------------------------------------------------------------- #
# Model-driven collector: env lifecycle, and terminated vs truncated.
# --------------------------------------------------------------------------- #

def test_collector_does_not_own_the_env_lifecycle():
    """It must neither reset an already-running env at the start nor close it.

    The caller owns the env now; an earlier version constructed and closed one
    per call, which restarted the level at every rollout boundary.
    """
    env = _StubEnv()
    obs, _ = env.reset()
    assert env.reset_calls == 1  # the caller's initial reset
    rollout = collect_rollout_with_model(
        env=env, obs=obs, model=None, model_state_fns=_stub_state_fns(),
        n_steps=4, novelty_gate=NoveltyGate(dim=OBS_DIM, capacity=8, k=2), novelty_coef=0.0,
    )
    assert env.reset_calls == 1, "collector reset an env whose episode never ended"
    assert not env.closed, "collector closed an env it does not own"
    assert env.t == 4
    # A second call continues from where the first stopped, rather than restarting.
    collect_rollout_with_model(
        env=env, obs=rollout["final_obs"], model=None, model_state_fns=_stub_state_fns(),
        n_steps=4, novelty_gate=NoveltyGate(dim=OBS_DIM, capacity=8, k=2), novelty_coef=0.0,
    )
    assert env.t == 8, "second rollout restarted the env instead of continuing it"


def test_running_out_of_steps_mid_episode_bootstraps_a_real_value():
    rollout = _collect(_StubEnv(), n_steps=4)
    assert rollout["dones"] == [0.0, 0.0, 0.0, 0.0]
    assert rollout["last_value"] == pytest.approx(STUB_VALUE)
    assert rollout["truncation_values"] == [0.0] * 4


def test_terminated_episode_bootstraps_zero():
    rollout = _collect(_StubEnv(terminate_at=4), n_steps=4)
    assert rollout["terminateds"] == [0.0, 0.0, 0.0, 1.0]
    assert rollout["truncateds"] == [0.0] * 4
    assert rollout["dones"] == [0.0, 0.0, 0.0, 1.0]
    # No future to bootstrap: neither the end-of-rollout value nor a truncation value.
    assert rollout["last_value"] == 0.0
    assert rollout["truncation_values"] == [0.0] * 4


def test_truncated_episode_bootstraps_a_real_critic_value():
    """The bug this pins down: collapsing truncation into `done` and bootstrapping
    0.0 would zero the value target of an episode that was still alive, which only
    becomes reachable once the env persists long enough to hit its step limit."""
    rollout = _collect(_StubEnv(truncate_at=4), n_steps=4)
    assert rollout["truncateds"] == [0.0, 0.0, 0.0, 1.0]
    assert rollout["terminateds"] == [0.0] * 4
    assert rollout["dones"] == [0.0, 0.0, 0.0, 1.0]  # still cuts the GAE recursion
    assert rollout["truncation_values"] == pytest.approx([0.0, 0.0, 0.0, STUB_VALUE])
    assert rollout["truncation_values"] != rollout["terminateds"]


def test_episode_end_resets_the_env_and_continues_collecting():
    rollout = _collect(_StubEnv(terminate_at=2), n_steps=4)
    # Slot 0 carries the env's step index: 1, 2, then a fresh episode's 0, 1.
    starts = [float(o[0]) for o in rollout["obs"]]
    assert starts == [0.0, 1.0, 0.0, 1.0], f"env was not reset on the episode boundary: {starts}"


def test_truncation_reward_fold_equals_a_real_bootstrap():
    """The identity run_training relies on: with dones[t]=1 cutting the recursion,
    folding gamma*V(s_T) into that step's reward reproduces exactly the GAE a real
    bootstrap would have produced."""
    gamma, lam = 0.99, 0.95
    rewards = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.5, 0.5])
    # (a) truncated at the last step: recursion cut, bootstrap folded into reward.
    folded = rewards + gamma * torch.tensor([0.0, STUB_VALUE])
    adv_fold, ret_fold = compute_gae(folded, torch.cat([values, torch.tensor([0.0])]),
                                     torch.tensor([0.0, 1.0]), gamma=gamma, lam=lam)
    # (b) reference: the same trajectory bootstrapped through the values array.
    adv_ref, ret_ref = compute_gae(rewards, torch.cat([values, torch.tensor([STUB_VALUE])]),
                                   torch.tensor([0.0, 0.0]), gamma=gamma, lam=lam)
    assert torch.allclose(adv_fold, adv_ref)
    assert torch.allclose(ret_fold, ret_ref)
    # And the fold is not a no-op: dropping it (0.0 bootstrap) gives a different target.
    adv_zero, _ = compute_gae(rewards, torch.cat([values, torch.tensor([0.0])]),
                              torch.tensor([0.0, 1.0]), gamma=gamma, lam=lam)
    assert not torch.allclose(adv_zero, adv_ref)


def test_novelty_is_scored_on_observations_not_logits():
    """Curiosity must react to the game state reached, not to the policy's action
    distribution (10-dim logits): the latter is not a state signal at all, and its
    buffer goes stale on every policy update."""
    pushed = []

    class _SpyGate(NoveltyGate):
        def push(self, state_vec):
            pushed.append(state_vec.detach().clone())
            super().push(state_vec)

    env = _StubEnv()
    rollout = _collect(env, n_steps=4, novelty_gate=_SpyGate(dim=OBS_DIM, capacity=8, k=2))
    assert len(pushed) == 4
    for vec in pushed:
        assert vec.shape == (OBS_DIM,), f"novelty vector is {tuple(vec.shape)}, not the observation"
        assert vec.shape != (N_ACTIONS,)
    # And it is the state REACHED (next_obs: env steps 1..4), not the one acted from.
    assert [float(v[0]) for v in pushed] == [1.0, 2.0, 3.0, 4.0]
    assert rollout["rewards"] == rollout["extrinsic_rewards"]  # novelty_coef=0.0 here
