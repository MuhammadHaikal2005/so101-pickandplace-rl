"""Roll out a trained SAC policy on our env and render an MP4."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio.v2 as imageio
import mujoco
import numpy as np
from stable_baselines3 import SAC

from env import SO100PickPlaceEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out", type=str, default="rollout.mp4")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--camera", type=str, default="iso", choices=["front", "iso"])
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    env = SO100PickPlaceEnv()
    model = SAC.load(args.model, device="cpu")

    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    frames: list[np.ndarray] = []
    successes = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=42 + ep)
        ep_reward = 0.0
        last_info: dict = {}
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            last_info = info
            renderer.update_scene(env.data, camera=args.camera)
            frames.append(renderer.render())
            if term or trunc:
                break

        ok = bool(last_info.get("is_success", 0.0) >= 1.0)
        successes += int(ok)
        print(
            f"episode {ep}: reward={ep_reward:+.2f} success={ok} "
            f"cube_target={last_info.get('cube_target_xy', float('nan')):.3f} "
            f"cube_z={last_info.get('cube_z', float('nan')):.3f}"
        )

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=args.fps, codec="libx264", quality=8)
    print(f"Wrote {len(frames)} frames to {out_path}")
    print(f"Success rate: {successes}/{args.episodes}")

    renderer.close()
    env.close()


if __name__ == "__main__":
    main()
