import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import gym_lowcostrobot  # noqa: F401
import imageio.v2 as imageio
from stable_baselines3 import SAC


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--env", type=str, default="LiftCube-v0",
                        choices=["LiftCube-v0", "PickPlaceCube-v0"])
    parser.add_argument("--out", type=str, default="rollout.mp4")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fixed-start", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    kwargs = dict(
        observation_mode="state",
        action_mode="ee",
        reward_type="dense",
        render_mode="rgb_array",
    )
    if args.fixed_start:
        kwargs["cube_xy_range"] = 0.01
        if args.env == "PickPlaceCube-v0":
            kwargs["target_xy_range"] = 0.01
            kwargs["goal_z_range"] = 0.0

    env = gym.make(args.env, **kwargs)
    env = gym.wrappers.FlattenObservation(env)
    model = SAC.load(args.model, device="cpu")

    frames = []
    successes = 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=42 + ep)
        ep_reward = 0.0
        ok = False
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            if info.get("is_success", False):
                ok = True
            if term or trunc:
                break
        successes += int(ok)
        print(f"episode {ep}: reward={ep_reward:+.3f} success={ok}")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=args.fps, codec="libx264", quality=8)
    print(f"Wrote {len(frames)} frames to {out_path}")
    print(f"Success rate: {successes}/{args.episodes}")
    env.close()


if __name__ == "__main__":
    main()
