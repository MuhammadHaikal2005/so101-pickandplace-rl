import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import PPO

from env import SO100PickPlaceEnv


def main() -> None:
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    if model_path is None:
        print("usage: diagnose.py <model.zip>")
        sys.exit(1)

    env = SO100PickPlaceEnv()
    model = PPO.load(model_path, device="cpu")
    obs, _ = env.reset(seed=42)

    jaw_min, jaw_max = float("inf"), float("-inf")
    max_z, max_contact = 0.0, 0.0
    contact_steps = 0
    snapshots = []

    for i in range(env.max_episode_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        jaw_angle = float(env.data.qpos[env.qpos_addrs[5]])
        jaw_min = min(jaw_min, jaw_angle)
        jaw_max = max(jaw_max, jaw_angle)
        max_z = max(max_z, info["cube_z"])
        if info["in_contact"] > 0.5:
            contact_steps += 1
            max_contact = max(max_contact, info["cube_z"])

        if i in {0, 25, 50, 75, 100, 150, 200, 249}:
            snapshots.append(
                f"step {i:3d}: jaw={jaw_angle:+.3f} ee_cube={info['ee_cube_dist']:.3f} "
                f"cube_z={info['cube_z']:.3f} contact={int(info['in_contact'])} "
                f"action_jaw={float(action[5]):+.3f}"
            )

        if term or trunc:
            break

    for s in snapshots:
        print(s)
    print()
    print(f"jaw range over episode: [{jaw_min:+.3f}, {jaw_max:+.3f}] rad")
    print(f"jaw actuator range:      [{env.ctrlranges[5, 0]:+.3f}, {env.ctrlranges[5, 1]:+.3f}] rad")
    print(f"max cube_z during episode: {max_z:.4f} m  (table is {0.015:.4f} m)")
    print(f"contact steps: {contact_steps}/{env.max_episode_steps}")
    print(f"max cube_z while in contact: {max_contact:.4f} m")


if __name__ == "__main__":
    main()
