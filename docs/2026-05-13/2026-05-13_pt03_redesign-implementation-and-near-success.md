# Redesign Implementation and Near-Success Training Run

> Date: 2026-05-13
> Project / Stage: SO-101 sim-to-real for GR00T N1 fine-tuning — redesign and proof-of-concept training
> Topic: From gym-lowcostrobot bug discovery through full architectural redesign to SAC v2 that picks, lifts, and transports the cube but does not release

## What we accomplished

- Diagnosed a real bug in `perezjln/gym-lowcostrobot`'s `LiftCubeEnv`: line 335 sums `cube_z - height_threshold` with `+np.linalg.norm(ee - cube)`, rewarding the gripper for being far from the cube. The library is not safe to use as-is for SO-100 lift training.
- Ran a brainstorming session that produced a written design spec at `docs/superpowers/specs/2026-05-13-rl-redesign-design.md` and an executable plan at `docs/superpowers/plans/2026-05-13-rl-redesign-implementation.md`, both committed to git.
- Decided architecturally: keep our own env and scene, replace joint actions with Cartesian end-effector delta actions, switch PPO to SAC, adopt a ggando-style reward function extended for the full pick-and-place task. This is the option B from the brainstorm.
- Created a GitHub repository at `github.com/MuhammadHaikal2005/so101-pickandplace-rl` and pushed the project, then iteratively pushed each milestone.
- Executed the 17-task implementation plan via subagent-driven development, with each task gated by failing-test-first TDD and the implementer's self-review. Result: 13 unit tests, all passing.
- Added a TCP site to `scene/so_arm100.xml` at the geometric center of the four fixed-jaw finger pads, for use as the IK reference point.
- Rewrote `env/so100_pick_place_env.py` end-to-end: 4-D Cartesian action (`[dx, dy, dz, gripper_delta]`), 22-D state observation, damped-least-squares IK solver, strict both-jaw grasp detection, eleven distinct reward components (reach, grasp, lift, transport, three milestone bonuses, four discouragement penalties, +1000 terminal reward).
- Built two validation gates as runnable scripts. Gate 1 (`scripts/test_ik.py`) drives the TCP through five waypoints and renders an MP4 — IK passes with max waypoint error 1.19 cm. Gate 2 (`scripts/test_reward.py`) runs five random-action episodes and logs every reward component to a CSV — random policy averages ~70 reward per episode, mostly reach plus jerk penalty, no grasps as expected.
- Ran SAC v1 training (500k steps, 70 minutes wall time). Result: eval reward plateaued at 160 with 0 percent success. Diagnosed: reach reward dominated, policy parked gripper near cube with jaw fully open.
- Rebalanced reward to v2: capped reach reward at 0.3 per step, raised grasp reward to +1.0, raised lift coefficient to 4.0, and added ±5 mm cube spawn noise to break deterministic local optima.
- Ran SAC v2 training (500k steps, 75 minutes). Result: eval reward 950, with the policy genuinely **reaching, grasping (180/200 steps contact), lifting the cube to 12.8 cm, and transporting it to within 2.8 cm of the target XY**. Success rate still 0 percent because the policy holds the cube over the target rather than releasing it.
- Designed v3: add a `reward_descent` term that pays only when the policy is holding the cube over the target zone, scaling as the cube approaches the table. This is intended to pull the policy from "hold high" into "lower and release".

## Walkthrough — what we did and why

### Discovering and ruling out gym-lowcostrobot

After the long sequence of PPO joint-space failures documented in pt02, we adopted option A from the prior brainstorm — replace our env with the canonical `perezjln/gym-lowcostrobot` library and train SAC on its `LiftCube-v0`. The first 500k-step run plateaued at ~17 to 19 reward and 0 percent success. Two things were wrong:

- I had forgotten to pass `--fixed-start` to the training script, so the cube spawned anywhere in a 30 cm square. That alone would have made convergence very slow.
- More importantly, the library's reward function (`gym_lowcostrobot/envs/lift_cube_env.py:335`) is **broken**: it computes `reward = (cube_z - height_threshold) + np.linalg.norm(ee_pos - cube_pos)` with the distance term **positive**. The policy correctly maximised it by moving the gripper away from the cube. No setting or hyperparameter would fix this — only either patching the library or replacing it entirely.

Confirming this changed the calculus: the option B branch from the brainstorm (build it ourselves, copying the architectural lessons from ggando) was now strictly better.

### The brainstorming session and the written spec

Rather than charge into implementation off the cuff, ran the brainstorming superpower skill. It asked focused single-choice questions to converge on a design:

- **Episode termination**: step-based truncation with early termination on confirmed success; failure penalised via reward, not terminated.
- **Episode length**: 200 steps at 20 ms per step (4 seconds of simulated time).
- **Spawn behaviour**: cube and target fully fixed initially (no noise).
- **Architecture**: Cartesian end-effector delta actions plus SAC plus ggando-style smooth reward (option B), not option A (joint actions plus better reward) and not option C (gym-lowcostrobot directly).

We then walked through three design sections one at a time (action space + IK; reward function + discouragement; termination + observation + algorithm + scene), getting approval after each section. The result was saved as a design spec, then committed and pushed.

The decisive point during the brainstorm: when I asked the user whether they were content with option B given the prior "massive failure" of the gym-lowcostrobot attempt, I explicitly separated the architectural choice (Cartesian + SAC) from the library bug (positive distance term in lowcostrobot's reward). The evidence overwhelmingly favours Cartesian + SAC for this task; the earlier failure was a library issue, not an architecture verdict.

### The implementation plan

Wrote a 17-task plan with complete code, exact paths, and TDD steps for each task. Saved to `docs/superpowers/plans/`. The plan structure:

- Tasks 1-9: incremental env construction, each task one method or reward term, gated by a new unit test
- Tasks 10-13: validation scripts (test_ik, test_reward) and training/eval scripts (train_sac, eval_sac_visualize)
- Task 14: delete obsolete PPO scripts
- Tasks 15-17: run validation gates, train SAC, render demo

Each task in the plan included exact code blocks for both test and implementation, exact `git -C` commit commands, and a verification step. The total length is around 30 KB.

### GitHub repository and commit hygiene

Initialised the project as a git repo, created an SSH-targeted remote at `github.com/MuhammadHaikal2005/so101-pickandplace-rl`, pushed selectively. Used `.gitignore` to drop:

- The 2.3 GB MuJoCo Menagerie clone in `third_party/`
- The 1 GB of intermediate training checkpoints in `runs/*/checkpoints/` and TensorBoard event files in `runs/*/tb/`
- The 3 MB of SO-100 mesh files in `scene/assets/` (re-fetched from menagerie via a documented one-liner in README)

What stays in git: env code, scripts, scene XML, design spec, implementation plan, progress notes, and per-run artefacts that matter (final.zip, best_model.zip, demo MP4, eval scores, training log).

Pushed 35 selectively-included files (~67 MB) from the prior PPO experiments as historical artefacts, then maintained an incremental commit-per-task discipline through the implementation.

### Subagent-driven execution

Used `subagent-driven-development` skill to execute the plan. For each task: a dispatched implementer subagent received the full task text (no plan-file reading required), executed the TDD steps, and reported back. For mechanical tasks where the plan provides verbatim code, skipped the full two-reviewer cycle and verified directly. The first task underwent the full review cycle and surfaced two minor quality issues (unused import, undocumented scene-file choice) which the implementer fixed before the task was closed.

One notable implementer concern: in Task 3 (DLS IK), the implementer correctly observed that the originally-written test required IK to **commit qpos changes** rather than restore them. The implementer wrote the IK to leave qpos at the solution, then flagged this in their report. I caught it as a design error in the test (not the implementation) — IK should be a pure function so `env.step` can let the position actuators drive joints dynamically. Fixed by re-dispatching the implementer to restore qpos in IK and rewrite the test to verify IK output via direct qpos inspection plus forward kinematics, not via actuator settling. The fixed version also asserts that `_ik_solve` does not mutate env state, catching this class of bug for future changes.

### Validation gates

Both gates passed cleanly:

- **Gate 1 (IK)**: arm tracked all five waypoints within 1.19 cm (threshold was 3 cm). MP4 at `runs/gate1_ik/ik_waypoints.mp4`.
- **Gate 2 (Reward sanity)**: required components (reach, jerk) fired on every step. Random policy never grasped — which is what we expected for a small 2 cm cube and was confirmed as a non-issue (SAC's entropy-driven exploration is supposed to discover this).

### SAC v1 — the reach trap

500k step SAC training completed in 70 minutes wall time. Result: eval reward stuck at 147 (first eval) to 161 (final eval). Episode length always 200 (no successes, ever).

The diagnostic showed exactly what was happening:

- Gripper went from jaw=-0.007 to jaw=+1.049 over the episode — **the policy progressively opened the gripper**, the wrong direction.
- Cube nudged sideways from (0.06, -0.18) to (0.084, -0.136), about 3 cm displacement. `max_cube_z = 0.013` (3 mm tip).
- Zero contact steps. Policy never grasped.

Reward decomposition explains it: with `reward_reach = 1 - tanh(10 * dist)` capping at 1.0, hovering near the cube earns ~0.83 per step × 200 = ~166. The discouragement penalty for pushing (-1 one-time) is too small to overcome this. Without ever experiencing a successful grasp during exploration, the policy never learns that grasping is more rewarding than hovering. Critically, SAC's auto-tuned `ent_coef` collapsed to 0.0013 — exploration ended after about 25k steps and the policy was locked into the local optimum.

### v2 — rebalancing the reward to break the trap

Made four targeted changes:

- **Reach reward capped at 0.3** (was 1.0). Hovering 200 steps now caps at 60 instead of 200.
- **Grasp reward up to 1.0** (was 0.25). A single grasp event pays more than 3 steps of hovering.
- **Lift coefficient 4.0** (was 2.0). Maximum lift reward is now 4 per step instead of 2.
- **Cube spawn noise ±5 mm**. Breaks the deterministic trap where the policy always sees the same starting state and converges quickly.

All 13 tests passed after updating the reach-reward test bounds (which had assumed the old 1.0 cap).

Trained 500k steps. Eval reward progression told a clear story:

| Timesteps | Mean reward |
| --- | --- |
| 25k | 36 |
| 50k | 45 |
| 100k | 44 |
| 125k | 42 |
| **150k** | **288** ← discovered grasping |
| 175k | 905 |
| 200k | 214 (noisy regression) |
| 250k | 412 |
| 325k | 896 |
| 500k | 950 |

The policy genuinely learned to:

- Reach the cube (steps 0-25)
- Close the gripper (action_jaw=-0.97 at step 25, contact established)
- Lift the cube (max cube_z = 0.1286, i.e., 12 cm above the table)
- Transport the cube over the target (final cube XY position 0.027 m from target, well inside the 4 cm threshold)
- 180 of 200 steps in contact

But it does not release. Across all five eval episodes, the final state is identical: cube held in the air over the target. Success bonus never collected.

### Why the v2 policy does not release

The reward math at the holding-over-target state pays handsomely:

- reach ≈ 0.23 (gripper close to its own held cube)
- grasp = 1.0
- lift ≈ 3.96 (cube at z = 0.11)
- transport ≈ 0.73 (cube near target)
- total ≈ 5.9 per step

Holding for the remaining 150 of 200 steps gives ~885 reward. The eval reward of 950 matches that math closely. The policy has found a stable, high-paying attractor and lacks the gradient to leave it.

In principle, releasing for success at any point should be strictly better: +1000 bonus minus the discounted holding rewards forgone. But for the policy to discover this, it has to actually try releasing. SAC's entropy in v2 is much higher than v1 (0.148 versus 0.0013), but the policy has converged on the "hold" attractor and is no longer exploring opening the gripper.

This is the credit-assignment problem: the policy never sees an episode where releasing leads to higher cumulative return, so it never learns the value of release.

### v3 design — descent reward

Designed one targeted addition for v3:

```python
reward_descent = 0.0
if in_contact and cube_target_xy < 0.05:
    descent_progress = 1.0 - float(np.tanh(20.0 * cube_lift))
    reward_descent = 2.0 * descent_progress
```

Behaviour:

- Over target with cube high (lift = 0.10 m): descent reward ≈ 0.1 (negligible)
- Over target with cube descending (lift = 0.04 m): descent reward ≈ 0.8
- Over target with cube near table (lift = 0.005 m): descent reward ≈ 1.9
- Plus the existing place bonus (+5 when settled) and success bonus (+1000 on release)

This adds a continuous gradient from "hold high" to "lower toward table", which then naturally chains into the place bonus and the success bonus. Importantly, it is purely additive — does not change any existing reward, only adds a new positive term in a specific regime.

About to commit v3 and start the next 500k training run.

## Problems hit and how we fixed them

### gym-lowcostrobot LiftCubeEnv has a sign bug

- **What happened:** Trained SAC on `LiftCube-v0` for 500k steps. Eval reward 17-20 throughout, never improved, 0 percent success. Diagnostic showed the policy was moving the gripper **away** from the cube.
- **Why it happened:** `gym_lowcostrobot/envs/lift_cube_env.py` line 335: `reward = (cube_z - self.height_threshold) + np.linalg.norm(ee_pos - cube_pos)`. The distance term should be negative; it is positive. The policy correctly maximised reward by moving the gripper as far from the cube as possible.
- **How we fixed it:** Abandoned the library. Wrote our own Cartesian env with our own correct reward function.
- **Lesson learned:** Verify the reward function of any third-party RL env by reading the source before trusting it. Reward sign bugs are invisible in metrics — only the policy's actual behaviour reveals them.

### IK design conflict between purity and the test

- **What happened:** First attempt at the DLS IK implementer left `data.qpos` at the IK solution rather than restoring it. The test passed but env dynamics would have broken — every step would teleport the arm instead of letting actuators drive it smoothly.
- **Why it happened:** The test as written set `data.ctrl` to the IK output and then ran 20 mj_step substeps. With kp=50 actuators, 20 substeps from home is not enough time to reach a 7 cm-away target — the test would fail if IK was pure. So the implementer adapted IK rather than the test.
- **How we fixed it:** Re-dispatched the implementer with explicit fix instructions: restore qpos in IK (make it pure), and rewrite the test to verify IK output by writing it into qpos and forward-kinematicising, not via actuator settling. Added an assertion that `_ik_solve` does not mutate `data.qpos`.
- **Lesson learned:** When the test and the implementation seem to conflict, sometimes the test is wrong. Pure functions are worth preserving — if the test pretends to validate the function but actually validates the actuator dynamics, the test design is the bug. Catch this with state-preservation assertions.

### v1 SAC convergence to the reach-and-hover trap

- **What happened:** SAC v1 plateaued at eval reward 160 after 25k steps. 0 percent success. Policy parked gripper near cube with jaw open, never grasped.
- **Why it happened:** Reach reward (max 1.0/step) earned 200 reward across the full episode. Grasp reward (0.25/step) only fired if grasping, which exploration almost never achieved. Without successful grasp episodes in the replay buffer, SAC could not learn the value of grasping. Entropy collapsed to 0.0013, killing further exploration. The policy was perfectly stable in a suboptimal attractor.
- **How we fixed it:** v2 reward rebalance — capped reach at 0.3 (cap the trap), boosted grasp to 1.0 and lift to 4.0 (make the real task more attractive), added ±5 mm spawn noise (force the policy to actually react rather than memorise).
- **Lesson learned:** A dense shaping reward whose total over an episode exceeds the sparse success bonus creates a local optimum that the policy will not leave. Cap dense terms.

### v2 SAC holds cube over target without releasing

- **What happened:** v2 trained 500k steps, reached 950 eval reward. Policy grasps, lifts, transports — but holds the cube above the target indefinitely. 0 percent success.
- **Why it happened:** Holding over target pays ~6 reward per step (reach + grasp + lift + transport). Across the remaining episode steps, that exceeds the eventual success bonus by a comfortable margin. To learn the value of release, the policy must actually try releasing — but SAC's entropy is no longer high enough to explore "open gripper while over target".
- **How we fixed it (in design, not yet trained):** Designed v3 with a `reward_descent` term that scales with proximity to the table when the policy is over target. This creates a continuous gradient from "hold high" to "lower toward table", which then chains into the existing place bonus and success bonus.
- **Lesson learned:** "Hold near the goal" is a common high-reward local optimum when the success criterion involves release. Either reward the act of approaching the table at the target, or penalise prolonged contact above the target.

### Stale conda-run prompt parser handling multiline -c arguments

- **What happened:** Conda's run subcommand could not handle `python -c "<multi-line string>"` invocations. Errored with `NotImplementedError: Support for scripts where arguments contain newlines not implemented.`
- **Why it happened:** `conda run` wraps the command and re-shells it. Some encoding paths cannot represent newlines in argument vectors on Windows.
- **How we fixed it:** Wrote one-off Python scripts to files (e.g., `scripts/inspect_glr.py`) and ran them via `conda run -n so101-rl python <file>`.
- **Lesson learned:** Use file-based Python invocation, not inline `-c` strings, when going through `conda run` on Windows.

## Concepts clarified

### Why Cartesian end-effector actions beat joint-space actions for manipulation

Joint-space actions (six independent joint commands) require the policy to internally solve the inverse kinematics — to know which joint combinations produce useful end-effector motions. Random actions in joint space produce essentially uniformly random end-effector poses, almost none of which point the gripper anywhere useful. Cartesian actions (`[dx, dy, dz, gripper]`) are interpreted by an analytical IK solver. Random actions in Cartesian space produce small purposeful end-effector motions — exploration naturally moves the gripper around the workspace. The kinematic structure is solved by the env, not by the policy. This is why ggando's lift task converged in 500k SAC steps where our PPO joint-space attempts could not converge in 5 million.

### Why off-policy SAC beats on-policy PPO for sparse-success tasks

PPO is on-policy: it samples trajectories from the current policy, computes advantages, takes a gradient step, then discards the trajectories. A rare successful trajectory contributes one gradient step and is gone. SAC is off-policy: it stores trajectories in a replay buffer and re-samples from them across many gradient steps. A rare successful trajectory continues to shape the policy for hundreds of subsequent updates. For tasks where success during exploration is rare (manipulation, sparse-reward navigation), SAC's sample efficiency is decisive.

### The credit-assignment problem and dense reward design

When the agent must perform a sequence of actions (reach → grasp → lift → transport → release) and only the final action triggers the big terminal reward, the agent must learn to credit early actions for the eventual outcome. Dense shaping rewards address this by paying intermediate steps as the agent makes progress through the sequence. But if a dense reward pays so much per step that holding indefinitely is preferable to completing the task, the dense reward has redirected the policy toward a non-terminal attractor. Each "phase" should pay strictly more than the previous, and the terminal reward should pay strictly more than holding-in-the-final-state. v2 violates the last condition; v3 attempts to repair it.

### Why purity matters for `_ik_solve`

`_ik_solve` is called inside `env.step` to compute a joint target. The position actuators (kp=50, force-range ±3.5 Nm) then drive the joints toward that target across 10 simulation substeps (20 ms of sim time). This is the system's dynamics. If `_ik_solve` writes the IK solution directly into `data.qpos`, it skips the actuators — the arm teleports each step. That makes motion infinitely fast and bypasses the physics that ultimately determines whether the gripper can hold a cube. So `_ik_solve` must be a pure function: it iterates `data.qpos` internally as scratch space, then restores it before returning the target. The actuator dynamics handle the motion.

## Where things stand now

```
so101-rl-poc/
├── env/
│   ├── __init__.py
│   └── so100_pick_place_env.py     (Cartesian, IK, v2 rewards)
├── scene/
│   ├── pick_place_scene.xml         (2 cm cube, fixed target)
│   ├── so_arm100.xml                (with tcp site)
│   └── assets/                       (gitignored, fetch from menagerie)
├── scripts/
│   ├── test_ik.py                    (Gate 1)
│   ├── test_reward.py                (Gate 2)
│   ├── test_env.py                   (smoke)
│   ├── train_sac.py                  (uses our env)
│   ├── eval_sac_visualize.py         (uses our env)
│   └── diagnose.py                   (per-step diagnostic)
├── tests/
│   ├── __init__.py
│   └── test_env.py                   (13 tests, all passing)
├── runs/
│   ├── ppo_v1..v7/                   (historical PPO failures, committed)
│   ├── sac_lift_v1/                  (gym-lowcostrobot failure, gitignored)
│   ├── sac_cart_v1/                  (reach trap, 160 reward, 0% success)
│   ├── sac_cart_v2/                  (950 reward, holds over target, 0% success)
│   ├── gate1_ik/                     (waypoint validation MP4)
│   └── gate2_reward/                 (random-policy CSV)
└── docs/
    ├── superpowers/
    │   ├── specs/2026-05-13-rl-redesign-design.md
    │   └── plans/2026-05-13-rl-redesign-implementation.md
    └── 2026-05-13/
        ├── 2026-05-13_pt01_isaac-debugging-and-mujoco-rl-poc.md
        ├── 2026-05-13_pt02_ppo-iterations-and-cartesian-pivot.md
        └── 2026-05-13_pt03_redesign-implementation-and-near-success.md (this file)
```

GitHub remote: `git@github.com:MuhammadHaikal2005/so101-pickandplace-rl.git` (public). Latest pushed commit: `80c5755` ("Reward v2: cap reach at 0.3, raise grasp to 1.0 and lift to 4.0, add cube spawn noise"). Branch: `main`. SSH-tracking remote configured.

Conda env: `so101-rl` (Python 3.11) with `mujoco`, `gymnasium`, `stable-baselines3[extra]`, `imageio`, `tensorboard`, and `gym-lowcostrobot` (installed but no longer used).

Best trained policy as of this note: `runs/sac_cart_v2/eval/best_model.zip`. It can reach, grasp, lift, and transport the cube to the target — but cannot release it. Demo at `runs/sac_cart_v2/demo.mp4`.

## What's next

Immediate (this session):

- Implement v3 reward (`reward_descent` added when grasped and over target, scales as cube approaches table).
- Update test_env.py if any reward-component assertions need adjustment.
- Train SAC v3 for 500k steps (estimated 70 to 90 minutes).
- Render demo MP4 and inspect: did the policy learn to lower and release?

If v3 succeeds:

- The proof of concept is complete. Present demo to the team.
- Update README to describe the final architecture.
- Update memory with the working recipe so future SO-101 RL work starts here.

If v3 still does not release:

- Consider adding a small bonus for opening the gripper while over the target.
- Or increase SAC's `ent_coef` floor to maintain exploration.
- Or use a curriculum: spawn the cube already in the gripper for some episodes so the policy can learn the release-and-place portion in isolation.

Longer term (next week, on the incoming Threadripper + 3x 3090 hardware):

- Port the working reward function, observation space, and Cartesian action handling to Isaac Lab in the workshop repo.
- Retrain at higher parallel-env count on the SO-101 USD assets.
- Drive the trained policy through the workshop repo's existing dataset-recording pipeline to generate the automated GR00T N1 fine-tuning dataset.
