"""Smoke tests for the SO-100 Cartesian pick-and-place env."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np


def test_tcp_site_exists():
    # Load the composite scene (not so_arm100.xml directly) so we also exercise the
    # <include> chain that the env uses at runtime.
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene" / "pick_place_scene.xml"))
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    assert sid >= 0, "tcp site not found in scene"
    print(f"  tcp site id={sid}")
    fixed_jaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
    assert model.site_bodyid[sid] == fixed_jaw_id, "tcp site not attached to Fixed_Jaw"
    print("  tcp site correctly attached to Fixed_Jaw body")


def test_env_imports_and_constructs():
    from env import SO100PickPlaceEnv
    env = SO100PickPlaceEnv()
    assert env is not None
    env.close()
    print("  env constructed and closed")


def test_action_space_is_4d():
    from env import SO100PickPlaceEnv
    env = SO100PickPlaceEnv()
    try:
        assert env.action_space.shape == (4,), f"got {env.action_space.shape}"
        low = env.action_space.low
        high = env.action_space.high
        assert np.all(low == -1.0) and np.all(high == 1.0)
        print(f"  action_space={env.action_space}")
    finally:
        env.close()


def test_observation_space_is_22d():
    from env import SO100PickPlaceEnv
    env = SO100PickPlaceEnv()
    try:
        assert env.observation_space.shape == (22,), f"got {env.observation_space.shape}"
        print(f"  observation_space={env.observation_space}")
    finally:
        env.close()


def test_reset_returns_valid_obs():
    from env import SO100PickPlaceEnv
    env = SO100PickPlaceEnv()
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (22,), f"got {obs.shape}"
        assert obs.dtype == np.float32
        assert isinstance(info, dict)
        print(f"  obs[:6]={obs[:6]}")
    finally:
        env.close()


TESTS = [
    test_tcp_site_exists,
    test_env_imports_and_constructs,
    test_action_space_is_4d,
    test_observation_space_is_22d,
    test_reset_returns_valid_obs,
]


def main():
    failures = 0
    for t in TESTS:
        name = t.__name__
        try:
            print(f"[RUN]  {name}")
            t()
            print(f"[PASS] {name}\n")
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}\n")
            failures += 1
        except Exception as e:
            print(f"[ERR]  {name}: {type(e).__name__}: {e}\n")
            failures += 1
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
