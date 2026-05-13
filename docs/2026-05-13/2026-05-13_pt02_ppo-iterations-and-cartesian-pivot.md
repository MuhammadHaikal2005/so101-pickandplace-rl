# PPO Iterations and Cartesian Pivot

> Date: 2026-05-13
> Project / Stage: SO-101 sim-to-real for GR00T N1 fine-tuning — proof-of-concept iteration
> Topic: Eight PPO training runs hit successive failure modes; web research revealed the architectural fix (SAC plus Cartesian end-effector control)

## What we accomplished

- Ran six additional PPO training cycles (v3 through v8 reward designs, v7 also adjusted physics) totalling roughly 27 million simulated steps and 100+ minutes of training.
- v3 produced the first policy with 100 percent eval success rate — but visual inspection revealed it was pushing the cube along the table rather than grasping, lifting, and placing.
- v4 to v6 progressively addressed the pushing exploit, the abrupt motion, and the gripper-never-closes failure mode by adjusting action representation, success criteria, and reward shaping.
- v6 was the first run in which the policy genuinely closed the gripper around the cube and attempted to lift, confirmed by diagnostic instrumentation.
- v7 reduced the cube to 2 cm with a lighter mass to give the gripper a fair chance to hold it; lift improved marginally but never crossed the 6 cm success gate.
- At the user's prompting, performed web research and identified that the entire architectural choice (PPO plus joint-space actions) is the wrong fit for fine manipulation. Multiple practitioner reports (ggando, hlfshell) and the canonical low-cost robot library (gym-lowcostrobot) all use SAC plus Cartesian end-effector control with proven success in 500k steps.
- Wrote `scripts/diagnose.py` to print per-step jaw angle, contact state, ee-cube distance, and cube z so that we can read the failure mode from a trained policy without relying on visual inspection alone.
- Added v3 demo `runs/ppo_v3/demo.mp4`, plus v4, v5, v6, v7 demos for visual comparison.

## Walkthrough — what we did and why

### v3: first apparent success, then visual disqualification

Coming out of pt01 we had v3 nearly written (transport-progression reward, lift-history gate on success, etc.). v3 trained in 14 minutes and the metrics looked excellent — eval success rate 100 percent, mean reward 1038, mean episode length 31 steps. The demo MP4 looked plausible at a glance. The user watched it and immediately flagged two problems:

- The arm was moving too fast.
- More importantly, the policy was **pushing** the cube along the table to the target zone rather than grasping and lifting.

This was a textbook example of why metric-driven validation is insufficient. The transport-progression reward (`5 * (1 - distance / start_distance)` when `cube_z > 0.04`) had a loophole: the cube can satisfy `z > 0.04` briefly while being pushed and tipped, qualifying for transport reward without an actual lift.

### v4: delta actions plus lift-history gate

Two coordinated changes to address both problems at once:

- **Delta-based action interpretation** to slow motion. Each env step the policy could command at most `MAX_ACTION_DELTA = 0.04 rad` of joint change. At 50 Hz this corresponds to roughly 115 deg/s, close to the real SO-100 servo limit.
- **Lift-history gate on success**: track `max(cube_z)` across the episode; require it to exceed `MIN_LIFT_FOR_SUCCESS = 0.08 m` before success can trigger. Pushing keeps the cube at z ≈ 0.015 and cannot satisfy this gate.

Result: eval reward dropped to 208 ± 0.41 and success rate fell to 0 percent. The policy converged on "approach cube, idle near it" and never closed the gripper. Diagnostic (later written) confirmed the gripper jaw stayed open the entire episode.

### v5: continuous lift reward to provide gradient

Hypothesis: the binary `+2` lift reward at `cube_z > 0.05` provides no gradient until that threshold is achieved. Under delta actions, the policy cannot "jump" to a lifted configuration; it needs a smooth signal pulling it upward. Added `reward_lift_continuous = 30 * max(0, cube_z - TABLE_Z) if in_contact`, reduced velocity penalty from 0.01 to 0.001, and extended training to 5 million steps.

Result: marginal improvement. Eval reward 244 ± 0.41 (versus v4's 208). Still 0 percent success. The diagnostic showed jaw range stayed entirely positive (gripper never closed below `+0.10 rad`, far from the fully-closed `-0.174 rad`). The policy learned to graze the cube with fingertips for the contact bonus but never actually grasped.

### User intervention: go back to v2's approach

The user paused execution to point out that v2 (absolute joint-position actions, simpler reward) had visually been doing the right thing — the gripper was positioned correctly above the cube and was closing. The problem with v2 was that after grasping it did not continue to lift. They suggested iterating from v2 rather than continuing down the delta-action path.

### v6: v2-derived with rebalanced rewards and longer episodes

Reverted to absolute joint-position actions. Adjusted the reward balance to make holding strictly less attractive than lifting:

- `reward_contact = 1.0` (down from v2's 2.0)
- `reward_lift = 50 * max(0, cube_z - TABLE_Z)` (up from v2's 30)
- Kept the `max_cube_z >= 0.08` gate on success to keep pushing ruled out
- Extended `max_episode_steps` from 250 to 500 (10 simulated seconds)

Result and diagnostic: **first run where the gripper actually closed**. Jaw range hit `-0.175 rad` (fully closed), `action_jaw = -1.000` for nearly the entire episode, contact for 220/500 steps. Eval reward 538 ± 70.7 — high variance because the policy now genuinely engages with the cube but does so inconsistently. However, **max cube z only reached 0.025 m** (1 cm above the table). The cube was being gripped but never properly lifted; the policy was lifting tentatively, the cube was slipping, the policy was clamping again.

### v7: shrink the cube to make grasping easier

The 3 cm cube was probably too large for the SO-100 gripper to hold securely through a lift motion. Changed the scene XML and `TABLE_Z` constant to use a 2 cm cube at 0.02 kg with friction increased from 1.5 to 2.0. Adjusted `MIN_LIFT_FOR_SUCCESS` from 0.08 to 0.06 (still clearly lifted, but easier to reach given the harder physics task we had been failing on).

Result: eval reward 720 ± 262 — the variance jumped to a striking 262, and episode rewards ranged from 455 to 984. The policy was on the edge of succeeding. But `max_cube_z` across the diagnostic episode stayed at 0.020 — only 1 cm above the new table height. The gripper was clamping the cube without enough motivation to actually raise it.

### Aborted v8: lift coefficient doubled

I started preparing v8 (lift coefficient 50 → 100, contact reward 1.0 → 0.5) on the theory that the marginal benefit of lifting was still insufficient. The user paused execution again and proposed a more principled move: search the web for someone who has already solved this.

### Web research

Searched for SO-100 / SO-101 RL pick-and-place, MuJoCo low-cost robot RL, and PPO pick-place practical lessons. Hit four directly relevant sources:

- **ggando.com/blog/so101-rl-lift** — someone trained exactly the SO-101 to grasp and lift in MuJoCo, achieving 100 percent success in **500k steps**.
- **github.com/ggand0/pick-101** — their code repository.
- **github.com/perezjln/gym-lowcostrobot** — the canonical Gymnasium library for low-cost arms including SO-100. Already includes `LiftCube-v0` and `PickPlaceCube-v0` environments. Supports Cartesian end-effector control natively.
- **hlfshell.ai/posts/ppo-pick-and-place** — practitioner blog on PPO pick-and-place.

The three sources converge on the same conclusions:

- **Use SAC, not PPO.** ggando explicitly states "SAC outperformed PPO on continuous control". hlfshell needed 20+ million PPO steps for a single-object pick-and-place.
- **Use Cartesian end-effector delta actions, not joint actions.** ggando quote: "Cartesian action space >> joint space for manipulation RL." Their action space is just 4 dimensions: `[dx, dy, dz, gripper]`. Random actions in this space naturally explore 3D space, whereas random joint actions produce wild flailing.
- **Smooth reach reward.** `1.0 - tanh(10 * distance)` rather than raw `-distance`. Bounded, smooth, and saturating at the goal.
- **Explicit drop penalty.** `-2.0` when the grasp is lost after being achieved. Without this, the policy treats dropping as merely "less optimal" rather than as a punished failure mode.
- **Multiple lift bonuses at graduated heights.** Continuous lift progress, plus binary +1 at z=0.02, binary +1 at z=0.08, plus +10 for sustained success.
- **Fingertip primitive collision geoms** for stable multi-point contact. (We already have these via the menagerie's `finger_collision` class.)

The realisation is that we have been training in the genuinely harder configuration the entire session. Eight reward redesigns inside the wrong architecture were never going to converge well. The path forward is to adopt the proven architecture rather than continue tuning rewards.

## Problems hit and how we fixed them

### v3 reward hacking via pushing

- **What happened:** 100 percent eval success rate, but the demo showed the policy pushing the cube along the table rather than picking it up.
- **Why it happened:** The transport-progression term was gated only by `cube_z > 0.04`, which a tipped cube can satisfy transiently. The success criterion did not require any lift history.
- **How we fixed it:** Tracked `max_cube_z` per episode and gated success on `max_cube_z >= MIN_LIFT_FOR_SUCCESS`. Also raised the transport-reward `z` threshold from 0.04 to 0.07.
- **Lesson learned:** Success criteria that examine only terminal state can be exploited by short-lived transients. Encode path requirements when the path matters.

### v4 stalled "idle near cube"

- **What happened:** Reward dropped, success rate at 0, eval std ±0.4 (very deterministic). Policy converged on stopping near the cube and doing nothing.
- **Why it happened:** Delta actions removed the policy's ability to jump to good configurations. The binary lift reward gave no gradient until achieved. There was no signal pulling the policy past the contact bonus.
- **How we fixed it:** Added a continuous lift reward proportional to height, gated on contact. This created a gradient from "barely lifted" to "well lifted". (v5)
- **Lesson learned:** Binary reward terms need to be paired with continuous shaping that bridges the zero-reward region, especially under constrained action spaces.

### v5 still no gripper closure

- **What happened:** Slight reward improvement, no success, jaw never closed.
- **Why it happened (real cause, found later):** Joint-space delta actions are simply too low-information for the policy to coordinate gripper closing with arm descent inside 5 million PPO steps. The continuous lift reward did not address the fundamental exploration problem.
- **How we fixed it:** Reverted to absolute joint-position actions (v6), which restored the policy's ability to express intent compactly. Eventually concluded the entire joint-space architecture should be replaced with Cartesian.
- **Lesson learned:** When the policy cannot execute the obvious right action under a constraint, the constraint is wrong; rebalancing rewards will not fix it.

### v6 cube being pinched but not lifted

- **What happened:** Gripper closing correctly, contact for almost half the episode, but `max_cube_z` only reached 0.025 — the cube was being held against the table, not lifted.
- **Why it happened:** Two compounding issues. First, the 3 cm cube was at the edge of what the gripper could hold securely during a lift. Second, even with the lift reward, the cost of "policy moves arm up and drops cube" exceeded the benefit unless the lift was very tall.
- **How we fixed it:** Shrunk the cube to 2 cm, reduced mass, increased friction (v7). Marginal improvement.
- **Lesson learned:** Physical limits of the gripped object matter. Verify the geometry can actually be grasped before blaming the policy.

### v7 still on the lifting edge

- **What happened:** Eval reward 720 ± 262 — high variance, some episodes nearly succeeded, others stalled. `max_cube_z` still ~0.020.
- **Why it happened:** The marginal benefit of lifting to z=0.06 (success gate) was still not decisively bigger than the risk of dropping. The policy was on a fence between "hold safely" and "lift and possibly drop".
- **How we fixed it:** Aborted v8 (which would have doubled the lift coefficient) and pivoted to the architectural change identified via web research.
- **Lesson learned:** When you have hit five reward designs and still cannot cross a behaviour gap, the algorithm or action space is probably wrong. Stop tuning rewards and reassess.

### Misconception: "v2's episode was too short"

- **What happened:** The user hypothesised that v2 (and v3) were moving abruptly because episodes were short, and lifting was failing because there was not enough time.
- **Why this was partially right and partially wrong:** Short episodes do create implicit time pressure via discounting, but the abrupt motion in v2/v3 was actually because absolute joint commands let the policy whip through poses, not because of episode length. v6 (also 500-step) was smooth because action filtering naturally happens when commanded targets are reachable. The lift problem turned out to be about gripper-cube physics and the lift-reward gradient, not episode length.
- **Lesson learned:** Multiple hypotheses can be partially right at the same time; resolve them one at a time and verify with diagnostics.

## Concepts clarified

### Why joint-space actions are bad for fine manipulation

A six-dimensional joint-space action means the policy must learn the inverse kinematics of the arm as part of solving the task. Random actions in joint space produce essentially random end-effector trajectories — the gripper points everywhere and nowhere. In contrast, a 4-D Cartesian action space `[dx, dy, dz, gripper]` lets random actions naturally explore 3D space around the current end-effector pose. The kinematics are solved analytically inside the env via inverse kinematics; the policy never has to discover them. This is why the same task converges in 500k SAC steps with Cartesian actions and fails to converge in 5 million PPO steps with joint actions.

### Why SAC fits better than PPO for manipulation

SAC is off-policy and uses a replay buffer, so it can repeatedly mine successful transitions. PPO is on-policy and discards old experience, so rare successful trajectories disappear before they can fully shape the policy. For pick-and-place — where success is rare during exploration — SAC's sample efficiency is decisive. Multiple sources independently report 5x to 50x sample efficiency improvements going from PPO to SAC on similar tasks.

### Reward design patterns observed in the literature

Three patterns appear across all the successful pick-and-place RL recipes:

- **Smooth shaping at saturation.** `tanh(c * distance)` rather than raw `-distance`. Bounded reward, smooth gradient, naturally saturates at the goal.
- **Explicit failure penalties.** Drop penalty `-2`, push-down penalty when the cube goes below the table. Without these, failure modes are merely "less rewarding" instead of "punished", and the policy never learns to actively avoid them.
- **Graduated milestones.** Multiple binary bonuses at successive heights (e.g., +1 at z=0.02, +1 at z=0.08, +10 at sustained target) rather than one big success bonus. The policy gets reinforcement signals throughout the task progression, not only at completion.

### What "reward hacking" means in practice

Across this session we saw three distinct reward-hack patterns:

- **State-occupancy hack** (v1): the policy parks in a state that triggers a dense bonus indefinitely.
- **Status-quo hack** (v2): the policy maintains a state (grasping) without progressing, because the maintain-reward exceeds the marginal progress-reward.
- **Terminal-condition hack** (v3): the policy reaches the terminal state via an unintended path (pushing) that the success criterion does not exclude.

These map to three reward-design countermeasures: (1) reward progress not occupancy, (2) make later phases pay strictly more than earlier ones, (3) encode path requirements in the success criterion. The literature implicitly applies all three.

## Where things stand now

```
C:\Users\Haikal\Desktop\Tech Learning\so101-rl-poc\
├── env\so100_pick_place_env.py        (v8-state: 2 cm cube, joint actions, v6 reward)
├── scene\pick_place_scene.xml         (2 cm cube)
├── scripts\
│   ├── train.py                       (PPO trainer)
│   ├── eval_visualize.py              (MP4 renderer)
│   ├── diagnose.py                    (per-step jaw/contact/lift diagnostic)
│   └── test_env.py
└── runs\
    ├── ppo_v1\ ... ppo_v7\            (all training runs preserved with demos and logs)
    └── ppo_v3\demo.mp4                (the "pushes the cube" 100% success demo)
```

Conda env `so101-rl` (Python 3.11) with `mujoco`, `gymnasium`, `stable-baselines3[extra]`, `imageio`, `tensorboard`.

About to pivot to **option A**: adopt the `perezjln/gym-lowcostrobot` library directly. It provides a `LiftCube-v0` / `PickPlaceCube-v0` environment for the SO-100 with Cartesian end-effector control built in. We will install it into our existing `so101-rl` conda env, write a small SAC training script using stable-baselines3, and start a run.

## What's next

Immediate (this session):

- Install `gym-lowcostrobot` into the `so101-rl` conda env.
- Inspect their env to confirm it loads cleanly and understand its observation / action space.
- Write `scripts/train_sac.py` using SAC with stable-baselines3.
- Kick off a training run on `LiftCube-v0` or `PickPlaceCube-v0`. Expected duration to first success: roughly 500k steps based on ggando's experience.
- Render a demo for the user once the run completes.

Notes for future-self:

- Our env, reward function, and scene work is not wasted — it remains the specification of what "correct" behaviour looks like (proper grasp, lift, transport, place with no pushing). Use it to validate the policy from gym-lowcostrobot's env afterwards.
- The diagnostic script (`scripts/diagnose.py`) generalises: it prints jaw angle, contact, ee-cube distance, cube z, and action commands over a rollout. Re-use it to debug whatever the SAC policy ends up doing.
- The `runs/` folder contains all seven failed training artefacts and demos — useful as before/after evidence when presenting the eventual successful policy to the team.
