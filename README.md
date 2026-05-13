# SO-100 Pick-and-Place Proof of Concept (MuJoCo + PPO)

Throwaway proof of concept: train a PPO policy to pick up a 3 cm cube from a fixed start zone and place it in a fixed target zone, using the SO-100 arm from MuJoCo Menagerie. Built to validate that an RL policy can solve the task before porting to Isaac Lab on a more powerful machine.

## Setup

```bash
conda activate so101-rl
```

Packages already installed: `mujoco`, `gymnasium`, `stable-baselines3[extra]`, `numpy`, `tensorboard`, `tqdm`, `imageio`, `imageio-ffmpeg`.

## Sanity check

```bash
python scripts/test_env.py
```

## Train

```bash
python scripts/train.py --timesteps 1000000 --n-envs 8 --run-name ppo_v1
```

Checkpoints land in `runs/<run-name>/checkpoints/`, the best evaluation model in `runs/<run-name>/eval/best_model.zip`, the final model in `runs/<run-name>/final.zip`, and TensorBoard logs in `runs/<run-name>/tb/`.

To inspect training:

```bash
tensorboard --logdir runs/
```

## Visualise a trained policy

```bash
python scripts/eval_visualize.py --model runs/ppo_v1/eval/best_model.zip --out runs/ppo_v1/rollout.mp4 --episodes 3
```

Produces an MP4 from the `iso` camera (pass `--camera front` for a front view).

## Task definition

- Arm: SO-100 from MuJoCo Menagerie (6 actuated joints, same names as SO-101 in the workshop repo).
- Cube: 3 cm red cube, fixed start at `(0.06, -0.18)`, +/- 5 mm noise.
- Target: 7 cm green disc, fixed at `(-0.06, -0.18)`.
- Observation: joint positions (6), joint velocities (6), cube position (3), cube quaternion (4), gripper-center position (3), gripper-to-cube vector (3), cube-to-target vector (3). Total 28 dims, state-only.
- Action: 6-dim continuous in `[-1, 1]`, rescaled to actuator control ranges.
- Reward (dense): reach shaping + close-to-cube bonus + lift bonus + place shaping + success bonus, with a small action L2 penalty.
- Success: cube within 4 cm of target site and resting on the table, held for 5 consecutive frames.
- Episode length: 200 steps (5 substeps per env step at simulator dt).
