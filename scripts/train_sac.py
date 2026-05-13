import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import gym_lowcostrobot  # noqa: F401
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


def make_env(env_id: str, seed: int, fixed_start: bool):
    def _init():
        kwargs = dict(
            observation_mode="state",
            action_mode="ee",
            reward_type="dense",
        )
        if fixed_start:
            kwargs["cube_xy_range"] = 0.01
            if env_id == "PickPlaceCube-v0":
                kwargs["target_xy_range"] = 0.01
                kwargs["goal_z_range"] = 0.0
        env = gym.make(env_id, **kwargs)
        env = gym.wrappers.FlattenObservation(env)
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LiftCube-v0",
                        choices=["LiftCube-v0", "PickPlaceCube-v0"])
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--run-name", type=str, default="sac_lift_v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-start", action="store_true",
                        help="Constrain cube/target to near-fixed positions")
    args = parser.parse_args()

    run_dir = Path(__file__).resolve().parent.parent / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    tb_dir = run_dir / "tb"
    eval_dir = run_dir / "eval"

    train_env = DummyVecEnv([make_env(args.env, args.seed, args.fixed_start)])
    eval_env = DummyVecEnv([make_env(args.env, args.seed + 1000, args.fixed_start)])

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        learning_starts=10_000,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=str(tb_dir),
        seed=args.seed,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(ckpt_dir),
        name_prefix="sac",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(eval_dir),
        log_path=str(eval_dir),
        eval_freq=25_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    final_path = run_dir / "final.zip"
    model.save(str(final_path))
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
