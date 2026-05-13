import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from env import SO100PickPlaceEnv


def main() -> None:
    env = SO100PickPlaceEnv()
    obs, _ = env.reset(seed=0)
    print("obs_dim:", obs.shape)
    print("action_dim:", env.action_space.shape)
    print("home obs[joint_pos]:", obs[:6])
    print("home obs[cube_pos]:", obs[12:15])
    print("home obs[ee_pos]:", obs[19:22])
    print("ee->cube vec:", obs[22:25])
    print("cube->target vec:", obs[25:28])

    total_reward = 0.0
    for i in range(20):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if i < 3 or i == 19:
            print(f"step {i:3d} reward={reward:+.4f} ee_cube={info['ee_cube_dist']:.3f} "
                  f"cube_target={info['cube_target_dist']:.3f} cube_z={info['cube_z']:.3f}")

    print(f"total reward over 20 zero-action steps: {total_reward:+.3f}")
    env.close()


if __name__ == "__main__":
    main()
