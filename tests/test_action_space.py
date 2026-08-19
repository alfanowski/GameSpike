from envs.mario_land_env import MarioLandEnv

EXPECTED_ACTIONS = [
    "noop", "left", "right", "left_run", "right_run",
    "jump", "left_jump", "right_jump", "left_run_jump", "right_run_jump",
]


def test_action_list_matches_action_space_size():
    assert MarioLandEnv.ACTIONS == EXPECTED_ACTIONS
    assert len(MarioLandEnv.ACTIONS) == 10


def test_action_index_roundtrip():
    for i, name in enumerate(EXPECTED_ACTIONS):
        assert MarioLandEnv.action_index_static(name) == i
