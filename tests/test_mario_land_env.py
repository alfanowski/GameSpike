"""Observation / reward / termination behaviour of MarioLandEnv, against a real ROM.

The reward is the load-bearing part of this file. Two specific ways of getting it
wrong were found the expensive way during RAM-address confirmation and are pinned
here so they cannot come back:

  * building the dense term on Mario's screen-relative X, which saturates at the
    camera lock and stops rewarding real progress
    (`test_reward_keeps_paying_after_screen_x_saturates`);
  * diffing the level-progress counter across a death, which reloads the level and
    so looks like one enormous negative delta
    (`test_a_life_loss_pays_the_death_penalty_and_rebaselines`).

Both of those tests assert the *magnitude* they are guarding against as well as
the value they expect, so neither can pass vacuously.
"""
import os
import random

import numpy as np
import pytest

from envs import ram_map
from envs.mario_land_env import (
    DEATH_PENALTY,
    LEVEL_COMPLETE_BONUS,
    OBS_DIM,
    PROGRESS_REWARD_PER_PIXEL,
    MarioLandEnv,
)

ROM_PATH = os.environ.get("MARIO_LAND_ROM_PATH")

pytestmark = pytest.mark.skipif(
    not ROM_PATH or not os.path.exists(ROM_PATH),
    reason="MARIO_LAND_ROM_PATH not set or file missing; set it to your own legally-dumped ROM",
)

# One 16px camera block of progress is the largest reward a single step can earn,
# so any "ordinary" step's reward must sit inside +/- this.
MAX_ORDINARY_STEP_REWARD = 1.0


def _env(**kwargs):
    """An env with the boot's control probe skipped -- it is exercised by its own
    tests in tests/test_ram_map_invariants.py, and paying 80 frames for it in every
    test here would only slow the suite down."""
    kwargs.setdefault("verify_control", False)
    return MarioLandEnv(rom_path=ROM_PATH, **kwargs)


def _rhythmic_action(env, step_index, period=10, hold=2):
    """Run right, jumping on `hold` steps out of every `period`.

    Holding right alone stalls against the first pipe after ~30 steps, which makes
    any "does progress keep paying?" assertion meaningless. This is the same jump
    rhythm the RAM-map invariants use, expressed in env steps; measured, it carries
    Mario from progress 242 to 817 before his first death at step 176.
    """
    return env.action_index("right_run_jump" if step_index % period < hold else "right_run")


# ------------------------------------------------------------ the brief's two tests
def test_reward_is_positive_for_sustained_rightward_progress():
    env = _env()
    env.reset()
    total_reward = 0.0
    right_action = env.action_index("right")
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(right_action)
        total_reward += reward
        if terminated or truncated:
            break
    env.close()
    assert total_reward > 0.0, "moving right should accumulate positive reward"


def test_episode_truncates_at_max_steps():
    env = _env(max_episode_steps=10)
    env.reset()
    noop = env.action_index("noop")
    truncated = False
    for _ in range(11):
        obs, reward, terminated, truncated, info = env.step(noop)
        if terminated or truncated:
            break
    env.close()
    assert truncated, "episode should truncate once max_episode_steps is exceeded"


# ------------------------------------------------------------------- observation
def test_observation_shape_bounds_and_reserved_slots():
    env = _env()
    obs, info = env.reset()
    seen = [obs]
    for action in range(env.action_space.n):
        obs, reward, *_ = env.step(action)
        seen.append(obs)
        assert isinstance(reward, float), "PPO code downstream expects a plain float reward"
    env.close()
    for obs in seen:
        assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
        assert env.observation_space.contains(obs), f"observation escaped its declared box: {obs}"
        assert list(obs[9:]) == [0.0, 0.0, 0.0], "slots 9-11 are reserved and must stay zero-filled"


def test_observation_carries_the_real_game_state():
    """The vector must be wired to the game, not to plausible-looking constants."""
    env = _env()
    obs, _ = env.reset()
    assert obs[5] == pytest.approx(1.0, abs=0.02), "the level timer starts at 400/400"
    assert obs[6] > 0.0, "Mario starts world 1-1 with lives in hand"
    assert obs[7] == 0.0, "Mario starts 1-1 small, i.e. powerup state 0"
    assert obs[8] == 0.0, "no points have been scored on the very first frame"

    y_start = obs[1]
    progress_slots = []
    for step in range(20):
        obs, *_ = env.step(_rhythmic_action(env, step))
        progress_slots.append(obs[0])
    env.close()
    assert max(progress_slots) > 0.0, "slot 0 must show the in-level progress made while running right"
    assert 0.0 < y_start < 1.0, f"Mario's normalised Y should sit inside the screen, got {y_start}"


def test_on_ground_flag_clears_in_mid_air_and_returns_on_landing():
    """No ground-contact bit was confirmed in Task 3, so this flag is inferred from
    Mario's Y holding still. It has to survive the jump apex, where Y is constant
    for 5 measured frames -- that is the whole reason the window is 8 frames."""
    env = _env()
    obs, _ = env.reset()
    assert obs[4] == 1.0, "Mario is standing on the ground at the level start"

    obs, *_ = env.step(env.action_index("jump"))
    airborne = [obs[4]]
    noop = env.action_index("noop")
    for _ in range(6):  # the whole arc, apex included
        obs, *_ = env.step(noop)
        airborne.append(obs[4])
    assert all(flag == 0.0 for flag in airborne), (
        f"on-ground flag stayed set during a jump: {airborne} -- the apex's constant-Y "
        "hang is most likely leaking through the stillness window"
    )

    for _ in range(10):  # land and settle
        obs, *_ = env.step(noop)
    env.close()
    assert obs[4] == 1.0, "the on-ground flag never came back after landing"


# ------------------------------------------------------------------------ reward
def test_reward_keeps_paying_after_screen_x_saturates():
    """Guards the bug the reward design exists to avoid.

    `read_mario_x` pins at the camera-lock position, so a reward built on it would
    go quiet for almost the whole level. The reward must instead track
    `read_level_progress`, exactly, at PROGRESS_REWARD_PER_PIXEL per pixel.
    """
    env = _env()
    env.reset()
    start_progress = ram_map.read_level_progress(env.pyboy)
    screen_x = [ram_map.read_mario_x(env.pyboy)]
    total, late_total = 0.0, 0.0
    for step in range(60):
        obs, reward, terminated, truncated, info = env.step(_rhythmic_action(env, step))
        assert not (terminated or truncated), "this run was not supposed to end an episode"
        assert not info["died"], "this run was not supposed to cost a life"
        screen_x.append(ram_map.read_mario_x(env.pyboy))
        total += reward
        if step >= 30:
            late_total += reward
    end_progress = ram_map.read_level_progress(env.pyboy)
    env.close()

    x_travel = max(screen_x) - screen_x[0]
    progress_travel = end_progress - start_progress
    assert progress_travel > 3 * x_travel, (
        f"screen X moved {x_travel} and level progress {progress_travel}; the camera never "
        "locked on, so this run does not prove anything about the saturation"
    )
    assert total == pytest.approx(progress_travel * PROGRESS_REWARD_PER_PIXEL), (
        f"reward summed to {total} but the level progress made was {progress_travel}px; "
        "the dense term is not a plain level-progress delta"
    )
    assert late_total > 0.0, (
        "the second half of the run earned nothing -- the reward is following screen X, "
        "which saturates at camera lock, instead of level progress"
    )


def test_a_life_loss_pays_the_death_penalty_and_rebaselines():
    """The one genuinely novel piece of logic in this env, tested directly.

    Running right into 1-1 kills Mario at a known spot, ~575px in. On that step the
    reward must be the flat death penalty -- NOT the ~-36 that diffing the
    level-progress counter across the level reload would produce -- and the very
    next step must be back to an ordinary progress delta, measured from the
    respawned position.
    """
    env = _env()
    env.reset()
    rewards, infos = [], []
    for step in range(250):
        obs, reward, terminated, truncated, info = env.step(_rhythmic_action(env, step))
        rewards.append(reward)
        infos.append(info)
        if info["died"] or terminated or truncated:
            break
    next_step = step + 1
    death = len(rewards) - 1
    assert infos[death]["died"], "Mario never died, so nothing about the death path was tested"
    assert not terminated, "losing the first of several lives must not end the episode"
    assert infos[death]["lives"] == infos[death - 1]["lives"] - 1

    # Non-vacuity: the level really did reload underneath this step.
    progress_before = infos[death - 1]["level_progress"]
    progress_after = infos[death]["level_progress"]
    naive_reward = (progress_after - progress_before) * PROGRESS_REWARD_PER_PIXEL
    assert progress_before - progress_after > 400, (
        f"level progress only moved {progress_before} -> {progress_after} across the death; "
        "the reload this test exists to survive did not happen"
    )
    assert naive_reward < 2 * DEATH_PENALTY, (
        f"the naive delta would only have been {naive_reward}, not meaningfully worse than "
        f"{DEATH_PENALTY}, so this test would pass even with the death handling removed"
    )

    assert rewards[death] == DEATH_PENALTY, (
        f"the death step paid {rewards[death]}, expected the flat death penalty "
        f"{DEATH_PENALTY} (a naive progress delta would have paid {naive_reward})"
    )

    # And the step after the death must be an ordinary one, measured from the
    # respawn -- not still comparing against Mario's pre-death position.
    obs, reward, terminated, truncated, info = env.step(_rhythmic_action(env, next_step))
    env.close()
    assert not info["died"], "Mario died twice in a row; re-baselining could not be observed"
    # Deliberately demanding a *positive* reward, not merely a non-catastrophic one:
    # without the re-baseline the delta would be ~-575px, which the tripwire would
    # quietly turn into 0.0 -- and a test that accepted 0.0 would not notice.
    assert 0.0 < reward <= MAX_ORDINARY_STEP_REWARD, (
        f"the step after the respawn paid {reward}; it should be an ordinary rightward "
        "progress delta, so the progress tracker was not re-baselined on the respawn"
    )
    assert min(rewards) == DEATH_PENALTY, (
        f"some step paid {min(rewards)}, which is worse than the death penalty -- a level "
        "reload leaked into the dense term somewhere"
    )


def test_no_step_ever_pays_worse_than_the_death_penalty():
    """Same guarantee as above, but over a long run with many deaths.

    Lives are poked high so one episode can absorb dozens of deaths; the point is
    that *every* reload is caught, not just the first one.
    """
    env = _env(max_episode_steps=1200)
    env.reset()
    env.pyboy.memory[ram_map.ADDR_LIVES] = 0x99
    rng = random.Random(7)
    run = env.action_index("right_run")
    run_jump = env.action_index("right_run_jump")
    rewards, deaths, tripwire_hits = [], 0, 0
    for _ in range(1000):
        obs, reward, terminated, truncated, info = env.step(
            run_jump if rng.random() < 0.25 else run)
        rewards.append(reward)
        deaths += bool(info["died"])
        tripwire_hits += bool(info["progress_reset_ignored"])
        if terminated or truncated:
            break
    env.close()
    assert deaths >= 3, f"only {deaths} deaths in 1000 steps; this run proves little"
    assert min(rewards) == DEATH_PENALTY, (
        f"worst step paid {min(rewards)} across {deaths} deaths, expected no worse than "
        f"{DEATH_PENALTY}"
    )
    assert max(rewards) <= MAX_ORDINARY_STEP_REWARD, (
        f"a step paid {max(rewards)}, above the one-block-per-step ceiling"
    )
    assert tripwire_hits == 0, (
        f"the tripwire fired on {tripwire_hits} of {deaths} deaths -- the death path is "
        "supposed to catch every level reload on its own (the tripwire is only a "
        "backstop). Most likely RESPAWN_SETTLE_FRAMES is too short, so a reload that "
        "straddles a step boundary is still pending when the tracker is re-baselined."
    )


def test_an_unattributed_progress_collapse_is_never_paid_as_a_delta(monkeypatch):
    """The backstop for a reload the lives counter does not announce.

    The death path is supposed to catch every reload, so this tripwire should be
    unreachable in normal play -- which is exactly why it needs a test of its own.
    """
    env = _env()
    env.reset()
    right = env.action_index("right")
    env.step(right)

    real = ram_map.read_level_progress
    monkeypatch.setattr(ram_map, "read_level_progress", lambda p: real(p) - 1000)
    obs, reward, terminated, truncated, info = env.step(right)
    env.close()
    assert reward == 0.0, (
        f"a 1000px progress collapse was paid as {reward}; it is physically impossible as "
        "movement and must never reach the dense term"
    )
    assert obs[0] == 0.0, "the observation's progress-delta slot must be neutralised too"
    assert info["progress_reset_ignored"], "a tripwire that fires silently cannot be audited"


# ------------------------------------------------------------------ termination
def test_episode_terminates_when_the_lives_counter_runs_out():
    env = _env(max_episode_steps=2000)
    env.reset()
    right = env.action_index("right")
    terminated = False
    for _ in range(2000):
        obs, reward, terminated, truncated, info = env.step(right)
        if terminated or truncated:
            break
    lives = info["lives"]
    env.close()
    assert terminated, "walking right forever must eventually end the episode"
    assert lives == 0, f"the episode ended with {lives} lives left, expected a game over"


def test_completing_a_level_pays_the_bonus_and_terminates():
    """A real, played-through completion of world 1-1 -- not a mocked signal.

    The seed is one found by search; the emulator is deterministic, so it replays
    the same completion every time. Lives are poked high because a randomised
    policy needs many attempts to get through the level.
    """
    env = _env(max_episode_steps=4000)
    env.reset()
    env.pyboy.memory[ram_map.ADDR_LIVES] = 0x99
    rng = random.Random(34)
    run = env.action_index("right_run")
    run_jump = env.action_index("right_run_jump")
    info, reward, terminated, truncated = None, 0.0, False, False
    for _ in range(3200):
        obs, reward, terminated, truncated, info = env.step(
            run_jump if rng.random() < 0.25 else run)
        if terminated or truncated:
            break
    progress = info["level_progress"]
    world_level = info["world_level"]
    complete = info["level_complete"]
    env.close()
    assert complete and terminated, (
        f"the seeded run did not finish world 1-1 (reached {progress}, now in {world_level}). "
        "This seed only replays the same completion while the frame timing is unchanged, so "
        "any change to the ROM, the boot sequence, frame_skip or RESPAWN_SETTLE_FRAMES "
        "invalidates it -- re-run a seed search and pick a new one"
    )
    assert world_level == (1, 2), f"expected to have advanced into world 1-2, got {world_level}"
    assert reward == LEVEL_COMPLETE_BONUS, (
        f"the completing step paid {reward}, expected the flat completion bonus "
        f"{LEVEL_COMPLETE_BONUS}"
    )


# ------------------------------------------------------------- mutation checks
# Same discipline as tests/test_ram_map_invariants.py: an assertion that cannot
# fail is not a test. Each case re-introduces one of the two reward bugs this file
# exists to prevent -- by corrupting the RAM reader the env builds the reward on --
# and demands that the named guard above then fails.

_MUTATIONS = [
    # The dense term built on screen-relative X instead of level progress.
    ("read_level_progress", lambda p: ram_map.read_mario_x(p),
     test_reward_keeps_paying_after_screen_x_saturates),
    # Deaths made invisible, so the env falls through to a raw delta across the reload.
    ("read_lives", lambda p: 2,
     test_a_life_loss_pays_the_death_penalty_and_rebaselines),
]


@pytest.mark.parametrize(
    "attribute,wrong_value,guard",
    _MUTATIONS,
    ids=[f"{name}-caught-by-{guard.__name__}" for name, _, guard in _MUTATIONS],
)
def test_a_corrupted_reward_signal_is_caught_by_these_tests(attribute, wrong_value, guard, monkeypatch):
    monkeypatch.setattr(ram_map, attribute, wrong_value)
    raised = None
    try:
        guard()
    except BaseException as exc:  # noqa: BLE001 -- pytest.fail raises a BaseException subclass
        raised = exc
    assert raised is not None, (
        f"corrupting ram_map.{attribute} did not make {guard.__name__} fail; "
        "that guard is not actually pinning the reward's behaviour"
    )


# ------------------------------------------------------------------------- misc
def test_step_before_reset_is_a_loud_error():
    env = _env()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(env.action_index("noop"))
    env.close()


def test_reset_returns_to_the_level_start_after_a_death():
    """reset() must genuinely re-boot, not resume wherever the last episode ended."""
    env = _env()
    env.reset()
    start_progress = ram_map.read_level_progress(env.pyboy)
    right = env.action_index("right")
    for _ in range(60):
        env.step(right)
    moved_progress = ram_map.read_level_progress(env.pyboy)
    obs, info = env.reset()
    after_reset = ram_map.read_level_progress(env.pyboy)
    env.close()
    assert moved_progress > start_progress, "the run before the reset made no progress"
    assert after_reset == start_progress, (
        f"reset() left the game at progress {after_reset}, not back at the level start "
        f"{start_progress}"
    )
    assert info["step"] == 0 and info["lives"] > 0
