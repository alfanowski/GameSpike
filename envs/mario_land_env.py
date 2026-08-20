"""Gymnasium-style wrapper around PyBoy running Super Mario Land.

The reward is the part of this file that has to be right, so the reasoning behind
it is recorded here rather than in a report nobody will read again:

  * The dense term is a delta of `ram_map.read_level_progress()`, NOT of
    `ram_map.read_mario_x()`. Mario's X is screen-relative and pins at 81 once the
    camera locks on (measured: +31 over a life in which Mario travelled ~575px), so
    a reward built on it is near-zero for almost the whole level and swings hugely
    negative on every respawn. That bug was found and fixed during the RAM-address
    confirmation work; see the warning on `ram_map.ADDR_MARIO_X`.
  * `read_level_progress()` collapses back to the level's start value when Mario
    dies, because a death reloads the level. Diffing across that reset would hand
    the agent one enormous, entirely fictional negative reward. A life loss is
    therefore treated as a *segment boundary*: the step is paid a fixed death
    penalty, the emulator is run past the reload, and the progress tracker is
    re-baselined on the respawned position so the next step's delta is measured
    from where Mario actually is.
  * A second, independent guard (`_progress_drop_tripwire`) rejects any progress
    drop too large to be movement, whatever caused it. The death path above should
    already have caught every reload, so this should never fire -- it exists so
    that an unforeseen reload path degrades to a zero reward rather than silently
    corrupting the training signal.

Episodes end when the game runs out of lives (see `_GAME_OVER_LIVES`), when the
level is completed (the world/level byte advances -- confirmed by driving a real
completion of 1-1), or on a step-count timeout.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs import boot, ram_map

OBS_DIM = 12

# Per-dimension MEAN of the real observation vector, in the same slot order as
# `_build_observation` below.
#
# WHY IT LIVES HERE, next to OBS_DIM and `_build_observation` rather than in
# `models/`: it is a property of the OBSERVATION CONSTRUCTION, not of any model.
# The models take `obs_dim` as an argument and are deliberately game-agnostic
# (importing this module into `models/` would drag gymnasium + PyBoy into every
# model test), so `training/train.py` -- the layer that already knows both the env
# and the models -- reads this constant and passes it down. Keeping it in this file
# means whoever edits `_build_observation` sees it in the same screenful.
#
# PROVENANCE: measured over 6,000 real rollout steps on Super Mario Land --
# 3,000 under a trained policy and 3,000 under a uniform-random policy, pooled.
# It is NOT a guess and NOT derived from the [-1, 1] observation_space bounds:
# real observations are strongly non-negative and DC-dominated
# (||E[obs]||^2 = 1.331336 against E||obs||^2 = 1.713384, i.e. 77.70% of the
# observation energy is the mean).
#
# RE-MEASURE THIS IF `_build_observation` CHANGES. Adding a slot, rescaling a
# normaliser, changing `frame_skip`, or wiring the reserved slots 9-11 all move
# these numbers, and a stale mean silently degrades `embed_init_mode="centered"`
# from an exact correction into an approximate one. Slots 9-11 are exactly 0.0
# because they are the documented reserved-zero slots.
OBS_MEAN = (
    0.006229, 0.831424, 0.001833, -0.000167, 0.136833, 0.755903,
    0.193796, 0.111167, 0.000400, 0.000000, 0.000000, 0.000000,
)
assert len(OBS_MEAN) == OBS_DIM, "OBS_MEAN must carry one entry per observation slot"

# --- reward -------------------------------------------------------------------
# read_level_progress() is in pixels but only accurate to one 16px camera block, so
# one block of real level progress is worth exactly +1.0. Measured at frame_skip=4,
# sustained running yields +3.7px per step on average (+0.23 reward) and never more
# than 16px (+1.0), which keeps per-step rewards comfortably O(1) for PPO.
PROGRESS_REWARD_PER_PIXEL = 1.0 / 16.0

# A death re-baselines the progress tracker, so the dense term does NOT implicitly
# punish it -- this constant is the entire punishment, and it is deliberately not
# huge. Ten blocks of progress (~43 average steps of running) is a clear local
# signal without drowning the dense term, and the real cost of dying is implicit
# anyway: the game grants 2-3 lives, so each death spends a third of the episode.
DEATH_PENALTY = -10.0

# Finishing world 1-1 is worth ~147 in dense reward (2592 - 242 px of progress), so
# +50 is a strong incentive to actually reach the exit rather than farm progress,
# without dominating the dense signal.
LEVEL_COMPLETE_BONUS = 50.0

# Measured: the lives counter drops on essentially the same frame the game reloads
# the level (progress collapsed 817 -> 242 exactly one frame after lives went
# 2 -> 1). A step spans frame_skip frames, so the drop can land on a step's last
# frame and leave the reload for the *next* step. These extra frames guarantee the
# reload has landed before the progress tracker is re-baselined.
RESPAWN_SETTLE_FRAMES = 8

# Tripwire, not a tuning knob: Mario's top speed is ~2px/frame, so a drop of 16px
# per frame is physically impossible as movement and can only be a level reload.
MAX_BACKWARD_PIXELS_PER_FRAME = 16.0

# Measured empirically: lives read 0 while Mario is still playing his last life --
# the game-over only happens on the death *after* that. Ending the episode here is
# deliberate: it costs one life's worth of experience but keeps the agent well away
# from the game-over -> title -> attract-demo sequence, where the emulator keeps
# producing plausible-looking state that the player does not control at all (the
# exact failure mode envs/boot.py exists to prevent).
#
# DO NOT relax this to "terminate only on the death after lives hit 0" without
# revisiting the level-completion check below. The two rules are coupled: the
# completion check reads a world/level *change* as "level finished", which is only
# safe because the episode can never reach the game over. Measured: after a game
# over the ROM returns to the title and its attract demo then plays world 1-2, so
# the world/level byte changes with no level ever having been completed.
_GAME_OVER_LIVES = 0

# --- observation normalisation ------------------------------------------------
SCREEN_HEIGHT = 144.0          # Game Boy screen height; Mario's Y is a screen coordinate
MAX_TIMER = 400.0              # read_timer() returns the HUD value; 1-1 starts at 400
MAX_LIVES = 9.0                # practical ceiling; no hard game maximum was confirmed
MAX_POWERUP_STATE = 4.0        # 0=small 1=growing 2=super 3=shrinking 4=post-hit
MAX_SCORE_DELTA = 500.0        # one enemy stomp is +100; 5 leaves headroom for a combo
MAX_X_PIXELS_PER_FRAME = 2.0   # measured: <= 4px of screen X per 4-frame step
MAX_Y_PIXELS_PER_FRAME = 4.0   # measured: up to 8px in a single frame while falling
MAX_PROGRESS_PIXELS_PER_FRAME = 4.0  # measured: <= 16px of level progress per 4-frame step

# On-ground heuristic. No ground-contact flag was confirmed in Task 3, so this is
# inferred from Mario's Y holding still. A single frame of stillness is not enough:
# the jump apex holds a constant Y for 5 frames (measured), so the window must be
# longer than that. The cost is a ~8-frame lag after landing, which is the honest
# price of not having a confirmed ground-contact bit.
ON_GROUND_STILL_FRAMES = 8


class MarioLandEnv(gym.Env):
    """Gymnasium-style wrapper around PyBoy running Super Mario Land.

    The observation is a 12-dim float32 vector, all components in [-1, 1]:

        0  progress_delta_norm  in-level horizontal displacement this step
        1  mario_y_norm         Mario's screen Y (larger = lower down)
        2  vel_x_norm           screen-relative X velocity
        3  vel_y_norm           screen-relative Y velocity
        4  on_ground_flag       1.0 when Mario's Y has been still long enough
        5  timer_norm           level timer / 400
        6  lives_norm           lives / 9
        7  powerup_norm         powerup state / 4
        8  score_delta_norm     score gained this step / 500
        9  reserved             always 0.0
        10 reserved             always 0.0
        11 reserved             always 0.0

    Slots 9-11 are reserved for enemy-relative features and are wired in a later
    plan, once enemy RAM slots are located; they are zero-filled rather than
    omitted so the observation's shape never changes under a downstream model.

    Slot 0 is named `mario_x_delta_norm` in the task brief's contract. With a fixed
    frame_skip that would be numerically identical to slot 2's `vel_x_norm`, i.e. a
    wasted dimension -- and, worse, screen X stops tracking Mario's real movement
    the moment the camera locks on. Slot 0 therefore carries the same quantity the
    reward is built on (true in-level displacement) and slot 2 keeps the
    screen-relative velocity, which is still informative: it is how the agent sees
    itself being pushed around inside the camera's frame.
    """

    ACTIONS = [
        "noop", "left", "right", "left_run", "right_run",
        "jump", "left_jump", "right_jump", "left_run_jump", "right_run_jump",
    ]
    _ACTION_BUTTONS = {
        "noop": [],
        "left": ["left"],
        "right": ["right"],
        "left_run": ["left", "b"],
        "right_run": ["right", "b"],
        "jump": ["a"],
        "left_jump": ["left", "a"],
        "right_jump": ["right", "a"],
        "left_run_jump": ["left", "b", "a"],
        "right_run_jump": ["right", "b", "a"],
    }

    def __init__(self, rom_path: str, headless: bool = True, frame_skip: int = 4,
                 max_episode_steps: int = 3000, verify_control: bool = True):
        """`verify_control` gates envs/boot.py's behavioural gameplay probe.

        The emulator is deliberately NOT constructed here: reset() boots a fresh one
        through envs/boot.py, and building a throwaway PyBoy in the constructor
        would just pay for a ROM load nobody uses.
        """
        super().__init__()
        self.rom_path = rom_path
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.verify_control = verify_control
        self.window = "null" if headless else "SDL2"
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        # A drop this big cannot be Mario walking left; it is a level reload.
        self._progress_drop_tripwire = MAX_BACKWARD_PIXELS_PER_FRAME * frame_skip

        self.pyboy = None
        self._control_verified = False
        self._step_count = 0
        self._prev_progress = 0
        self._prev_x = 0
        self._prev_y = 0
        self._prev_score = 0
        self._prev_lives = 0
        self._start_world_level = None
        self._last_frame_y = 0
        self._still_frames = 0

    # ---------------------------------------------------------------- actions
    @classmethod
    def action_index_static(cls, name: str) -> int:
        return cls.ACTIONS.index(name)

    def action_index(self, name: str) -> int:
        return self.ACTIONS.index(name)

    def _press_action(self, action: int):
        for button in self._ACTION_BUTTONS[self.ACTIONS[action]]:
            self.pyboy.button_press(button)

    def _release_action(self, action: int):
        for button in self._ACTION_BUTTONS[self.ACTIONS[action]]:
            self.pyboy.button_release(button)

    # ------------------------------------------------------------------ state
    def _tick(self):
        """Advance one frame, keeping the on-ground stillness counter up to date."""
        self.pyboy.tick()
        y = ram_map.read_mario_y(self.pyboy)
        if y == self._last_frame_y:
            self._still_frames += 1
        else:
            self._last_frame_y = y
            self._still_frames = 0

    def _read_state(self):
        return dict(
            progress=ram_map.read_level_progress(self.pyboy),
            x=ram_map.read_mario_x(self.pyboy),
            y=ram_map.read_mario_y(self.pyboy),
            lives=ram_map.read_lives(self.pyboy),
            timer=ram_map.read_timer(self.pyboy),
            powerup=ram_map.read_powerup_state(self.pyboy),
            score=ram_map.read_score(self.pyboy),
            world_level=ram_map.read_world_level(self.pyboy),
        )

    def _build_observation(self, state, progress_delta, vel_x, vel_y, score_delta):
        return np.array([
            np.clip(progress_delta / (MAX_PROGRESS_PIXELS_PER_FRAME * self.frame_skip), -1.0, 1.0),
            np.clip(state["y"] / SCREEN_HEIGHT, 0.0, 1.0),
            np.clip(vel_x / (MAX_X_PIXELS_PER_FRAME * self.frame_skip), -1.0, 1.0),
            np.clip(vel_y / (MAX_Y_PIXELS_PER_FRAME * self.frame_skip), -1.0, 1.0),
            1.0 if self._still_frames >= ON_GROUND_STILL_FRAMES else 0.0,
            np.clip(state["timer"] / MAX_TIMER, 0.0, 1.0),
            np.clip(state["lives"] / MAX_LIVES, 0.0, 1.0),
            np.clip(state["powerup"] / MAX_POWERUP_STATE, 0.0, 1.0),
            np.clip(score_delta / MAX_SCORE_DELTA, -1.0, 1.0),
            0.0, 0.0, 0.0,  # reserved: enemy-relative features, wired in a later plan
        ], dtype=np.float32)

    def _info(self, state, died=False, level_complete=False, progress_reset_ignored=False):
        """`progress_reset_ignored` reports that the tripwire fired.

        It is exposed rather than kept private so that "the death path already
        catches every reload, so the tripwire is unreachable in normal play" is a
        claim tests can actually check, instead of a comment nobody can verify.
        """
        return {
            "level_progress": state["progress"],
            "lives": state["lives"],
            "world_level": state["world_level"],
            "died": died,
            "level_complete": level_complete,
            "progress_reset_ignored": progress_reset_ignored,
            "step": self._step_count,
        }

    # ------------------------------------------------------------ gym surface
    def reset(self, *, seed=None, options=None):
        """Boot a fresh emulator and leave it at the start of world 1-1.

        `seed` is accepted for the Gymnasium API but changes nothing: the boot is
        frame-deterministic and the game has no exposed RNG seed, so every episode
        starts from a bit-identical state.
        """
        super().reset(seed=seed)
        self.close()

        # The control probe costs ~0.14s and 80 frames per boot. Because the boot is
        # frame-deterministic (fresh emulator, fixed tick counts, fixed inputs), once
        # it has landed in real player-controlled gameplay for this ROM every later
        # boot lands in exactly the same place -- so it is run once per env and then
        # skipped, rather than paid for on every episode.
        verify = self.verify_control and not self._control_verified
        self.pyboy = boot.boot_to_level_start(self.rom_path, window=self.window,
                                              verify_control=verify)
        if verify:
            self._control_verified = True

        state = self._read_state()
        self._prev_progress = state["progress"]
        self._prev_x, self._prev_y = state["x"], state["y"]
        self._prev_score, self._prev_lives = state["score"], state["lives"]
        self._start_world_level = state["world_level"]
        self._last_frame_y = state["y"]
        self._still_frames = ON_GROUND_STILL_FRAMES  # Mario starts standing on the ground
        self._step_count = 0
        obs = self._build_observation(state, progress_delta=0, vel_x=0, vel_y=0, score_delta=0)
        return obs, self._info(state)

    def step(self, action: int):
        if self.pyboy is None:
            raise RuntimeError("reset() must be called before step()")

        self._press_action(action)
        try:
            for _ in range(self.frame_skip):
                self._tick()
        finally:
            self._release_action(action)

        lives = ram_map.read_lives(self.pyboy)
        died = lives < self._prev_lives
        game_over = lives <= _GAME_OVER_LIVES
        # The world/level byte only advances once the end-of-level sequence has run
        # its timer-to-points bonus, so this fires up to a few hundred frames after
        # Mario actually reaches the exit (measured on 1-1: he stood at the exit from
        # frame ~9800 while the clock drained, and the byte flipped at 9902). Those
        # steps earn nothing and the bonus lands late; that is the price of the only
        # completion signal that was empirically confirmed.
        level_complete = ram_map.read_world_level(self.pyboy) != self._start_world_level

        if died and not game_over:
            # Run past the level reload before reading anything, so the state below
            # is the respawned Mario and not a half-reloaded frame; see
            # RESPAWN_SETTLE_FRAMES. Note the respawn is NOT always the level start:
            # 1-1 has a mid-level restart point (measured: a death at progress 929
            # respawned at 882, not at 242), which is exactly why the tracker is
            # re-baselined on whatever position the game chose rather than on a
            # hardcoded level-start value.
            for _ in range(RESPAWN_SETTLE_FRAMES):
                self._tick()

        state = self._read_state()
        # A death or a level change resets the level's progress counter, so the step
        # spans a discontinuity: no delta measured across it means anything.
        boundary = died or level_complete
        progress_delta = 0 if boundary else state["progress"] - self._prev_progress

        reset_ignored = False
        if level_complete:
            reward = LEVEL_COMPLETE_BONUS
        elif died:
            reward = DEATH_PENALTY
        elif progress_delta < -self._progress_drop_tripwire:
            # Unattributed level reload -- never feed it to the reward. See the
            # module docstring; this should be unreachable in normal play.
            progress_delta = 0
            reset_ignored = True
            reward = 0.0
        else:
            reward = progress_delta * PROGRESS_REWARD_PER_PIXEL

        obs = self._build_observation(
            state,
            progress_delta=progress_delta,
            vel_x=0 if boundary else state["x"] - self._prev_x,
            vel_y=0 if boundary else state["y"] - self._prev_y,
            score_delta=0 if boundary else state["score"] - self._prev_score,
        )

        # Re-baseline on the post-reload state, so the *next* step's delta is
        # measured from where Mario actually is now.
        self._prev_progress = state["progress"]
        self._prev_x, self._prev_y = state["x"], state["y"]
        self._prev_score, self._prev_lives = state["score"], state["lives"]

        self._step_count += 1
        terminated = bool(game_over or level_complete)
        truncated = bool(not terminated and self._step_count >= self.max_episode_steps)
        return (obs, float(reward), terminated, truncated,
                self._info(state, died, level_complete, reset_ignored))

    def close(self):
        if self.pyboy is not None:
            self.pyboy.stop(save=False)
            self.pyboy = None
