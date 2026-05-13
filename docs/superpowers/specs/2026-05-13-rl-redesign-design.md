# SO-101 Pick-and-Place RL Redesign — Design Specification

> Date: 2026-05-13
> Status: Awaiting user review
> Project: SO-101 sim-to-real RL proof of concept (`so101-rl-poc`)

## Context

Eight PPO training runs with joint-space actions in our custom MuJoCo env failed to produce a policy that completes pick-and-place. v6 was the first run where the policy reliably closed the gripper around the cube and attempted to lift; subsequent runs (v7, v8 prototypes) could not consistently lift past 2 cm. An attempted pivot to `perezjln/gym-lowcostrobot`'s `LiftCube-v0` with SAC and Cartesian actions failed for unrelated reasons (a sign bug in that library's reward function and a missed flag making the cube spawn randomly across 30 cm).

Web research surfaced a directly relevant reference: ggando's blog and `ggand0/pick-101` repository, which solved exactly this task (SO-100/SO-101 grasp and lift in MuJoCo) at 100 percent success rate in 500k SAC steps using Cartesian end-effector actions and a smooth shaped reward function.

This document specifies a redesigned environment, reward, and training pipeline that adopts the architectural lessons from ggando while keeping our scene and our task definition (pick-and-place, not just lift).

## Goals

- Produce a deterministic policy that picks up a 2 cm cube from a fixed position and places it in a fixed target zone, by the end of this development week.
- Movement must be visually realistic (no whipping motions) and physically correct (no pushing or sliding the cube into place).
- Output: trained policy checkpoint plus a rendered MP4 of multiple successful episodes.
- Reward design and observation design must port cleanly to Isaac Lab on the incoming hardware.

## Non-Goals

- Image-based observations. State-only this week.
- Randomised cube or target positions. Both fixed.
- Real-robot deployment. Sim only.
- Generalisation beyond the single pick-and-place task.

## Architecture Overview

```
+------------------+         +------------------+         +------------------+
|  SAC policy      |  ---->  |  Cartesian env   |  ---->  |  MuJoCo sim      |
|  (sb3, MlpPolicy)|         |  (our wrapper)   |         |  (our scene)     |
+------------------+         +------------------+         +------------------+
        ^                            |                            |
        |                            |  DLS IK                    |
        |  obs (22 dim)              v                            |
        |                    +------------------+                 |
        +-----  reward  <----|  Reward + obs    |<----------------+
                             +------------------+
```

The policy outputs a 4-D Cartesian command `[dx, dy, dz, gripper_delta]`. Our env wrapper converts the EE displacement into joint targets via damped least-squares inverse kinematics (DLS IK), applies the gripper delta directly to the Jaw actuator, steps the MuJoCo simulator for 10 substeps, then computes the reward and the 22-dim state observation.

## Action Space (Section 1)

| Index | Meaning | Action range | Scaled to |
| --- | --- | --- | --- |
| 0 | EE delta X | -1 to +1 | up to ±0.04 m per env step |
| 1 | EE delta Y | -1 to +1 | up to ±0.04 m per env step |
| 2 | EE delta Z | -1 to +1 | up to ±0.04 m per env step |
| 3 | Gripper delta | -1 to +1 | up to ±0.15 rad per env step |

Per-step EE motion cap: 4 cm × 50 Hz = 2 m/s peak end-effector velocity. Gripper closes through its full range (≈1.9 rad) in ~0.25 sec.

### IK Controller (Damped Least Squares)

Each env step:

1. Compute target EE position: `target = current_tcp_xpos + action[0:3] * 0.04`.
2. Clamp `target` to a conservative reachable bounding box: `x ∈ [-0.20, 0.20]`, `y ∈ [-0.30, 0.10]`, `z ∈ [0.005, 0.30]` (world coordinates, units of metres). The bounds match the SO-100's measured reach from its (0, 0, 0) base; the lower z bound keeps the EE above the floor.
3. Position Jacobian via `mujoco.mj_jacSite(model, data, jacp, None, tcp_site_id)`.
4. Solve `qdot = J^T @ inv(J @ J^T + λ²I) @ (target - tcp_xpos)` with damping `λ = 0.15`.
5. Add nullspace term biasing unused DOFs toward home pose.
6. Integrate: `q_target[0:5] = q_current[0:5] + step_size * qdot`, clipped to joint limits.
7. Repeat up to 10 iterations or until position error < 1 cm.
8. Gripper: `q_target[5] = clip(q_current[5] + action[3] * 0.15, jaw_min, jaw_max)`.
9. Write `q_target` to `data.ctrl`; step the simulator `n_substeps=10` times.

Implementation pattern follows `gym_lowcostrobot/envs/lift_cube_env.py:146-216`, adapted to our scene and joint names.

### Scene Change

Add a TCP site to `scene/so_arm100.xml` inside the `Fixed_Jaw` body:

```xml
<site name="tcp" pos="0.013 -0.10 0" size="0.005" rgba="0 1 1 0.5"/>
```

Position chosen at the geometric center of the four fixed-jaw finger pads.

## Reward Function (Section 2)

### Progress Rewards (per step, smooth gradients)

| Component | Formula | Range | Active when |
| --- | --- | --- | --- |
| Reach | `1 - tanh(10 * ee_to_cube_dist)` | 0 to 1 | always |
| Grasp | `+0.25` | 0 or 0.25 | both jaw pads contact cube |
| Lift | `2 * tanh(20 * max(0, cube_z - TABLE_Z))` | 0 to 2 | grasped |
| Transport | `1 - tanh(10 * cube_to_target_xy_dist)` | 0 to 1 | grasped AND cube_z > 0.05 |

### One-Time Milestone Bonuses

- `+1.0` first time `cube_z > TABLE_Z + 0.04` (cube clears 4 cm above table)
- `+1.0` first time `cube_xy_dist_to_target < 0.05` while lifted
- `+5.0` first time cube settles within target zone on table (`cube_xy_dist < 0.04` AND `cube_z < TABLE_Z + 0.005`)

### Discouragement System (Failure Penalties)

| Penalty | Magnitude | Trigger condition |
| --- | --- | --- |
| Drop while lifted | `-2.0` (once per drop event) | grasp lost while cube was above 4 cm |
| Pushing without grasp | `-1.0` (once per episode) | cube xy displaced > 1 cm and policy has never grasped this episode |
| Cube falls off table | `-5.0` (once) | `cube_z < TABLE_Z - 0.02` |
| Jerky motion | `-0.01 * ||action - prev_action||²` (per step) | always |

### Success and Terminal Reward

`+1000` and `terminated=True` when **all** of the following hold for 5 consecutive frames:

- `cube_xy_dist_to_target < 0.04`
- `cube_z < TABLE_Z + 0.005` (cube on table, not held)
- Not in contact (gripper released)

The large success bonus ensures release-and-place strictly dominates hold-over-target: a 200-step hold accumulates ~800 reward; a successful place at step 100 accumulates ~250 dense + ~7 milestones + 1000 success = ~1257.

## Termination and Episode Timing (Section 3)

| Property | Value |
| --- | --- |
| Simulator dt | 0.002 sec |
| Substeps per env step | 10 |
| Env step duration | 20 ms sim time (~50 Hz control) |
| Max env steps per episode | 200 |
| Episode max duration | 4 sec sim time |
| Early termination | success (5 consecutive success frames) |
| Failure termination | none (penalties instead) |

## Observation Space (22 dim, state-only)

| Component | Dim | Source |
| --- | --- | --- |
| Joint positions | 6 | `data.qpos[joint_addrs]` |
| End-effector position | 3 | `data.site_xpos[tcp_site_id]` |
| Cube position | 3 | `data.qpos[cube_free_addr:cube_free_addr+3]` |
| Target position | 3 | `data.site_xpos[target_site_id]` |
| EE-to-cube vector | 3 | derived |
| Cube-to-target vector | 3 | derived |
| Grasp flag | 1 | 1.0 if both jaw pad sets contact cube, else 0.0 |

Includes the redundant vectors `ee_to_cube` and `cube_to_target` because they shorten the path the network has to learn — small networks (`[256, 256]`) struggle to compute these from raw positions in early training.

## Algorithm

```python
SAC(
    "MlpPolicy",
    env,
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
)
```

Configuration mirrors ggando's published recipe. Training budget for first run: **500k steps** with checkpoint saved every 50k steps and eval every 25k steps over 5 episodes. Single training env (SAC parallelism in stable-baselines3 is limited).

Expected wall time on the current CPU: 60 to 90 minutes.

## Scene Configuration

| Property | Value |
| --- | --- |
| Cube size | 2 cm cube (half-extent 0.01) |
| Cube mass | 0.02 kg |
| Cube friction | 2.0 |
| Cube start position | `(0.06, -0.18, 0.01)` (fully fixed) |
| Target position | `(-0.06, -0.18, 0.001)` (fully fixed) |
| TABLE_Z | 0.01 (cube center height when on table) |
| TCP site | `Fixed_Jaw` body, local `pos="0.013 -0.10 0"` |

All other scene parameters (lighting, cameras, floor plane, target zone visual marker) carry over from `scene/pick_place_scene.xml` unchanged.

## Testing Plan

Two independent gates before declaring training success:

### Gate 1 — IK Validation

Script: `scripts/test_ik.py`. Drives the EE through five hand-picked waypoints with no policy in the loop:

1. Home pose
2. Above cube start (`0.06, -0.18, 0.15`)
3. On cube (`0.06, -0.18, 0.02`)
4. Lifted (`0.06, -0.18, 0.15`)
5. Above target (`-0.06, -0.18, 0.15`)

Renders an MP4 from the iso camera. The TCP must visually track each waypoint smoothly without overshoot or singularity chatter. Only then proceed to Gate 2.

### Gate 2 — Reward Sanity Check

Script: `scripts/test_reward.py`. Runs five random-action episodes, logging each reward component per step to a CSV. After the run:

- `reach` reward should vary between roughly 0.1 and 0.9 depending on EE-cube distance.
- `grasp` should fire intermittently (random actions occasionally produce contact).
- `lift`, `transport`, milestone bonuses should mostly be zero (random policy is unlikely to grasp and lift).
- Penalties: `jerky motion` should be small but nonzero. Others rarely fire.

If a component is stuck at zero across all five episodes (suggesting the trigger condition is unreachable from current dynamics), investigate before training.

### Gate 3 — Training and Evaluation

After Gates 1 and 2 pass, start the 500k SAC run. Eval every 25k steps. At completion:

- Render five evaluation episodes to MP4 using the deterministic policy.
- Each episode must visually show: reach → close gripper → lift → transport → release at target.
- Pushing the cube is a failure even if metrics look OK; reject and iterate the reward.

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| IK fails to converge near singularities | EE tracks poorly, policy gets confused | Gate 1 catches this before training; if needed, increase damping `λ` or add joint-velocity smoothing |
| TCP site position misaligned with actual gripping center | Policy approaches cube but never grips | Visual inspection during Gate 1; adjust the `pos` attribute in the MJCF |
| Cube physics still too slippery despite friction tuning | Same lift bottleneck as v6/v7 | Already mitigated by friction=2.0 in scene; if persists, raise to 3.0 or roughen cube faces with extra collision pads |
| SAC discovers a reward-hack we did not anticipate | Apparent success that fails Gate 3 visual check | Render mandatory; the visual gate is the ground truth, metrics are advisory |
| 500k steps insufficient | No success; reward plateau | ggando achieved 100 percent at 500k on a similar setup; if we plateau, extend to 1M before redesigning reward |

## Project Layout After Implementation

```
so101-rl-poc/
├── env/
│   └── so100_pick_place_env.py     (rewritten: Cartesian action, DLS IK, new reward)
├── scene/
│   ├── pick_place_scene.xml         (cube unchanged from v7, target unchanged)
│   └── so_arm100.xml                (one-line addition: <site name="tcp"/>)
├── scripts/
│   ├── test_ik.py                   (NEW: Gate 1)
│   ├── test_reward.py               (NEW: Gate 2)
│   ├── train_sac.py                 (rewritten to use our env, not gym-lowcostrobot)
│   ├── eval_visualize.py            (existing, may need minor tweak for new env)
│   └── diagnose.py                  (existing diagnostic)
└── docs/
    └── superpowers/specs/
        └── 2026-05-13-rl-redesign-design.md  (this file)
```

## References

- [ggando blog: SO-101 RL grasp and lift](https://ggando.com/blog/so101-rl-lift/)
- [ggand0/pick-101 GitHub repository](https://github.com/ggand0/pick-101)
- [perezjln/gym-lowcostrobot LiftCubeEnv (reference IK implementation, buggy reward)](https://github.com/perezjln/gym-lowcostrobot/blob/main/gym_lowcostrobot/envs/lift_cube_env.py)
- [hlfshell: PPO Pick and Place practical lessons](https://hlfshell.ai/posts/ppo-pick-and-place/)
- Internal progress notes:
  - `docs/2026-05-13/2026-05-13_pt01_isaac-debugging-and-mujoco-rl-poc.md`
  - `docs/2026-05-13/2026-05-13_pt02_ppo-iterations-and-cartesian-pivot.md`
