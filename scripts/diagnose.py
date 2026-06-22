"""Per-step diagnostic of a trained SAC policy on our Cartesian env.

Usage: python scripts/diagnose.py runs/<run-name>/eval/best_model.zip
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import SAC

from env import SO100PickPlaceEnv


def main() -> None:
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    if model_path is None:
        print("usage: diagnose.py <model.zip>")
        sys.exit(1)

    env = SO100PickPlaceEnv()
    model = SAC.load(model_path, device="cpu")

    obs, _ = env.reset(seed=42)
    jaw_min, jaw_max = float("inf"), float("-inf")
    max_z, max_contact_z = 0.0, 0.0
    contact_steps = 0
    snapshot_steps = {0, 25, 50, 75, 100, 125, 150, 199}

    print(f"start: ee={env._tcp_pos().round(3)} cube={env._cube_pos().round(3)} target={env._target_pos().round(3)}")
    print()

    for i in range(env.max_episode_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        jaw_angle = float(env.data.qpos[env.qpos_addrs[5]])
        jaw_min = min(jaw_min, jaw_angle)
        jaw_max = max(jaw_max, jaw_angle)
        max_z = max(max_z, info["cube_z"])
        if info["in_contact"] > 0.5:
            contact_steps += 1
            max_contact_z = max(max_contact_z, info["cube_z"])

        if i in snapshot_steps:
            ee = env._tcp_pos()
            cube = env._cube_pos()
            print(f"step {i:3d}: ee={ee.round(3).tolist()} cube={cube.round(3).tolist()} "
                  f"jaw={jaw_angle:+.3f} action_xyz=[{action[0]:+.2f},{action[1]:+.2f},{action[2]:+.2f}] "
                  f"action_jaw={action[3]:+.2f} contact={int(info['in_contact'])} "
                  f"r_reach={info.get('reward_reach', 0.0):.3f}")

        if term or trunc:
            break

    print()
    print(f"jaw range:                  [{jaw_min:+.3f}, {jaw_max:+.3f}] rad (full closed = {env.ctrlranges[5, 0]:.3f})")
    print(f"max cube_z:                 {max_z:.4f} m (table at {0.01:.4f})")
    print(f"contact steps:              {contact_steps}/200")
    print(f"max cube_z while in contact:{max_contact_z:.4f} m")


if __name__ == "__main__":
    main()
