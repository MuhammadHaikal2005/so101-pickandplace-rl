# Isaac Sim Debugging and MuJoCo RL Pick-and-Place Proof of Concept

> Date: 2026-05-13
> Project / Stage: SO-101 sim-to-real for GR00T N1 fine-tuning — environment setup and proof of concept
> Topic: Failed Isaac Sim launch on Windows + Blackwell, pivoted to a MuJoCo PPO pick-and-place proof of concept

## What we accomplished

- Identified the Isaac Sim launch failure as a deterministic access violation inside `rtx.scenedb.plugin.dll`, captured the exact fault signature, and ruled out cache, driver-update, headless, and crash-reporter mitigations.
- Cleared all three Omniverse caches (~1.4 GB total: shaders, texture cache, Kit 107.3 cache).
- Updated the NVIDIA driver from 596.21 to 596.49 and verified that Isaac Sim's compatibility checker reports the driver as supported.
- Confirmed that the failure persists across initialisation-time, post-`app ready`, headless `SimulationApp` initialisation, and crash-reporter-disabled launches.
- Inspected the `simulation-training-so-101` workshop repo and confirmed that it is an Isaac Lab teleop/imitation pipeline, not an RL training pipeline, tested only on Linux + Blackwell/Ada GPUs in Docker.
- Stood up a separate, throwaway MuJoCo proof-of-concept project at `C:\Users\Haikal\Desktop\Tech Learning\so101-rl-poc\`, including a Gymnasium environment, PPO training script, and an MP4 evaluation script.
- Created an isolated `so101-rl` conda environment (Python 3.11) with `mujoco`, `gymnasium`, `stable-baselines3[extra]`, `imageio`, and `tensorboard`.
- Cloned MuJoCo Menagerie, identified that its `trs_so_arm100` actually has the same six-DOF joint structure as the SO-101 in the workshop repo, and copied the MJCF and mesh assets into the project's scene directory so paths resolve cleanly.
- Iterated through three PPO training runs (v1, v2, v3) over a total of 9 million simulated steps. v3 reached a 100 percent eval success rate and produced a five-episode demo MP4 at `runs/ppo_v3/demo.mp4`.
- Identified that v3 is reward-hacking by pushing the cube rather than performing a proper grasp, and began the v4 reward redesign to enforce a true pick-lift-transport-place sequence.

## Walkthrough — what we did and why

### Isaac Sim launch failure

The user reported that the `isaacsim` PowerShell alias produced the NVIDIA crash reporter dialog, and pressing Cancel closed the application. The user's hypothesis was that the crash reporter was appearing prematurely, before the application had finished loading, and that the same behaviour on Ubuntu had been benign (dismissing the dialog allowed loading to continue).

We approached this systematically rather than symptomatically. Reading the Omniverse logs at `C:\Users\Haikal\.nvidia-omniverse\logs\` revealed that the latest two sessions had ended with explicit `event:"crash"` entries in `omni.processlifetime.log`, with Windows exception code `0xc0000005` (access violation). The per-session Kit log at `Kit/Isaac-Sim Full/5.1/kit_20260513_011441.log` contained a Breakpad stack trace whose top frames were inside `rtx.scenedb.plugin.dll`, called from `carb.scenerenderer-rtx.plugin.dll`, called from `omni.hydra.rtx.plugin.dll`, called from `omni::usd::UsdManager::getFoundationPlugins`. This is the RTX renderer's scene-database plugin being initialised during Hydra engine creation.

The user's mental model needed correcting because on Windows the kit process is genuinely terminating, not just blocked behind the dialog. The `... waiting for PID 2128 to exit ...` lines in the log confirmed that the Kit process is paused inside the Breakpad handler waiting for the dialog to close, and that once the dialog closes the handler proceeds to terminate the process — not resume application loading.

We tried four interventions in order:

1. **Clear caches.** Removed shaders, texturecache, and `Kit/107.3` to rule out stale state.
2. **Run compatibility check.** Confirmed that Driver 596.21 and the RTX 5070 are supported, but RAM at 16 GB is below the 32 GB minimum. Failure reported as RAM-only.
3. **Driver update.** User updated from 596.21 to 596.49. The compatibility check still passed everything except RAM, and `kit.exe` still launched cleanly under the compatibility-check application.
4. **Disable the crash reporter** with `--/crashreporter/enabled=false`. The dialog disappeared but the process still terminated silently around five seconds after `app ready`. Windows Error Reporting captured `kit.exe` faulting in `rtx.scenedb.plugin.dll` at fault offset `0x000000000000e533b`, exception code `0xc0000005`, WER bucket hash `81e936a480668649036a4729a2284327`.

The fault offset is identical across every attempt, including the original crashes before any intervention. This is a deterministic bug in the RTX scene-database plugin as shipped with Isaac Sim 5.1.0-rc.19 / Kit 107.3.3 when running against the RTX 5070 Blackwell on Windows 11 25H2.

We also tested whether headless mode could avoid the failing code path. A minimal Python script doing `SimulationApp({"headless": True})` and `sim_app.close()` reproduced the same crash in the same module — the headless flag suppresses only the window, not the RTX render delegate. None of the Python success markers printed, meaning `SimulationApp(...)` never returned.

We concluded that no software-level mitigation will resolve this without either rolling the driver back to a version that worked on this machine in mid-March 2026, or escalating with NVIDIA developer support using the WER bucket hash and minidumps.

### Pivot to a MuJoCo proof of concept

The user has a powerful machine arriving in one to two weeks (Threadripper 3970x, three RTX 3090s, 32 GB RAM) that will run Isaac Lab in Docker on Linux as the workshop repo requires. The goal for this week is purely to prove that a state-based RL policy can perform pick-and-place, so that next week's effort can focus on porting the validated reward and observation design to Isaac Lab and scaling up data generation.

After inspecting `simulation-training-so-101`, we confirmed that:

- The repo uses Isaac Lab (`from isaaclab.envs import ManagerBasedRLEnvCfg`), so it shares the broken Isaac Sim dependency.
- The repo contains teleop and imitation-learning scripts only (`lerobot_agent`, `lerobot_eval`, `lerobot_push_dataset`), no RL training.
- The robot description is USD only, no URDF or MJCF.
- The six joints — Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw — match the SO-100 description in MuJoCo Menagerie despite the SO-100/SO-101 naming distinction.

We chose MuJoCo with stable-baselines3 PPO because it runs comfortably on the current 16 GB Windows machine, supports the SO-100 out of the box, and the trained policy plus reward design will port directly to Isaac Lab.

### Building the proof-of-concept project

Project layout under `C:\Users\Haikal\Desktop\Tech Learning\so101-rl-poc\`:

- `scene/pick_place_scene.xml` — world description with arm, table, cube, target zone, and two cameras.
- `scene/so_arm100.xml` and `scene/assets/` — local copy of the SO-100 MJCF and meshes, copied here so MuJoCo's `meshdir` resolves correctly relative to the parent scene file.
- `env/so100_pick_place_env.py` — Gymnasium env with 29-dim observation (state-only, including a contact-based grasp flag), 6-dim continuous action.
- `scripts/test_env.py` — smoke test.
- `scripts/train.py` — PPO with checkpoints, TensorBoard logging, eval callback.
- `scripts/eval_visualize.py` — load a checkpoint and render an MP4.
- `third_party/mujoco_menagerie/` — shallow clone of the upstream repo (kept for provenance).

The conda environment is named `so101-rl`, isolated from the existing `lerobot` and `isaaclab` envs.

### Reward iteration

We trained PPO three times, each time learning something specific from the failure mode:

**v1 (close-bonus reward, three million steps).** Reward gave a `+0.5` bonus whenever the gripper was within `0.035 m` of the cube. The policy converged to parking the gripper at exactly that threshold to harvest the bonus indefinitely. End-of-episode reward climbed to `+64`, success rate stayed at zero, and the cube never moved. This is a textbook example of a dense-shaping reward producing a local optimum that beats the actual task.

**v2 (contact-based reward, two seconds longer episodes).** Replaced the proximity bonus with a `+2` per-step reward for any contact between the cube and the gripper-pad geoms, added a much stronger lift reward, and required the gripper to have released the cube as part of success. Doubled the simulation substeps from five to ten so each 250-step episode would have five seconds of simulated time. Eval reward jumped to `+476` (the policy was now actually grasping and lifting), but success rate stayed at zero. The new local optimum was "grasp the cube and hold it in the air without moving toward the target", because contact plus lift paid more than the modest place penalty.

**v3 (progress-shaped transport reward, 200-point success bonus).** Replaced the negative place penalty with a positive transport reward that grows as the cube approaches the target (`5 × (1 − distance / start_distance)`), and made the success bonus large enough (`+200`) that completing the task always beats any holding strategy. Result: eval success rate `100 percent`, mean reward `1038 ± 80`, mean episode length 31 simulation steps (the episode terminates early on a `SUCCESS_HOLD_STEPS = 5` consecutive success criterion).

The v3 demo MP4 is at `runs/ppo_v3/demo.mp4`.

### User feedback on v3 — reward hacking and motion speed

The user identified two issues by watching the demo:

1. The arm moves unrealistically fast. With absolute joint-position actions in `[-1, 1]` mapped directly to actuator ranges, the policy can command a full joint swing in a single 20 ms env step, producing whipping motions.
2. The success criterion `cube near target AND on table` does not require the cube to have been genuinely lifted, so the policy learned to push the cube along the table rather than grasping and placing it. The cube tipping during pushing was enough to occasionally cross the `0.04 m` lift threshold used in the transport-reward gate, so the policy got the transport reward via tipping rather than lifting.

This is exactly the right kind of feedback at the right time. The metrics did not reveal the hacking — we needed the user's eye.

### v4 design (in progress at the point of this note)

We started v4 edits to address both issues but had not yet retrained when the user paused for this note:

- **Delta-based action interpretation.** Each env step the policy commands a joint-position delta of at most `MAX_ACTION_DELTA = 0.04 rad`. With 20 ms env steps that caps joint speed at about 115 degrees per second, comparable to the real SO-100 servos.
- **Lift-history gate on success.** Track `max(cube_z)` over the episode and require it to exceed `MIN_LIFT_FOR_SUCCESS = 0.08 m` (4.5 cm clear of the table) before the success criterion can trigger. Pushing keeps the cube at `z = 0.015` or so, well below this gate.
- **Stricter transport reward.** Increase the `cube_z` gate from `0.04` to `0.07`, so a tipping push cannot qualify.

These three changes together force the policy to perform a true grasp-lift-transport-place sequence. Reward design otherwise unchanged from v3.

## Problems hit and how we fixed them

### MuJoCo could not find SO-100 mesh files

- **What happened:** First env load failed with `Error: Error opening file '...trs_so_arm100/Lower_Arm_Motor.stl'`. The path was correct except the `assets/` segment was missing.
- **Why it happened:** When MuJoCo encounters an `<include>`, it processes the included file's `<compiler meshdir="assets/">` directive but resolves the relative path against the **parent** scene file's directory, not the included file's directory.
- **How we fixed it:** Copied `so_arm100.xml` and the `assets/` directory into `scene/` next to the parent scene file. The relative `meshdir="assets/"` then resolves correctly.
- **Lesson learned:** MuJoCo `<include>` paths are surprising. The cleanest pattern is to keep included MJCFs and their assets in the same directory as the parent scene file.

### MuJoCo offscreen framebuffer too small

- **What happened:** First MP4 render failed with `Image width 960 > framebuffer width 640`.
- **Why it happened:** MuJoCo's default offscreen framebuffer is 640 by 480.
- **How we fixed it:** Added `offwidth="1280" offheight="960"` to the `<visual><global>` element in the scene XML.
- **Lesson learned:** Any MuJoCo offscreen rendering above the default resolution requires this XML setting.

### v1 policy hovered near cube without grasping

- **What happened:** Training reward climbed from -45 to +64, but success rate stayed at zero. Every episode ended with `ee_cube_dist ≈ 0.033` (right at the close-bonus threshold).
- **Why it happened:** The close-to-cube bonus was a step-function reward of `+0.5` per step inside `0.035 m`. Once the policy could collect this bonus indefinitely, any further motion risked losing it without a guaranteed payoff. Classic local optimum.
- **How we fixed it:** Removed the close-to-cube bonus entirely. Replaced with a contact-based grasp signal that only pays when the cube actually touches a finger pad geom.
- **Lesson learned:** Any dense shaping reward that pays for being in a state, rather than for changing state, is a candidate for this kind of trap.

### v2 policy grasped and held without transporting

- **What happened:** Training reward climbed to +462, success rate still zero. End-of-episode cube position was almost identical to start position. The policy had learned to grasp and lift, then stop.
- **Why it happened:** The reward decomposition favoured static holding: contact (+2 per step) plus lift (+1 to +2 per step) gave +3 to +4 per step, while the place reward `-1.5 × distance` only deducted -0.15 to -0.30 per step. Holding paid roughly 10x more than the marginal benefit of moving toward the target.
- **How we fixed it:** Replaced the negative place penalty with a positive transport reward that scales with progress: `5 × (1 − cube_target_dist / start_distance)`. This makes moving the cube toward the target strictly more rewarding than holding it in place. Also increased the success bonus from `+50` to `+200`.
- **Lesson learned:** Dense reward shaping needs to have strict monotonic progression: each phase of the task must pay strictly more than the previous one, with the largest payoff reserved for completion.

### v3 policy pushed instead of picking

- **What happened:** 100 percent eval success rate, but watching the demo MP4 the user noticed the policy was sliding the cube along the table to the target rather than lifting it. The success criterion did not enforce a true lift.
- **Why it happened:** The transport-reward gate was `cube_z > 0.04`, which a tipped/rolling cube can satisfy briefly. The success criterion only checked end-of-episode position, not lift history.
- **How we fixed it (v4 in progress):** Track the maximum cube z over the episode and require it to exceed `0.08 m`. Increase the transport-reward gate to `0.07 m`. Switch actions from absolute joint position to delta-based to slow motion and reduce the policy's ability to perform impulsive pushes.
- **Lesson learned:** Success criteria that only check terminal state are exploitable. Use lift history or phase tracking to enforce that the agent passed through the required phases.

## Concepts clarified

### Why Isaac Sim's headless mode did not avoid the RTX crash

`SimulationApp(headless=True)` suppresses only the GUI window. Internally the application still creates a USD context, which instantiates a Hydra render delegate, which by default is the RTX delegate (`omni.hydra.rtx`), which loads `rtx.scenedb.plugin.dll`. The crash is in the scene-database plugin's startup, not in any rendering or windowing code, so removing the window does nothing to avoid it. To genuinely avoid the failing code path you would need a non-RTX Hydra delegate (Storm, Iray), and Isaac Sim's design is tightly coupled to the RTX delegate.

### What Breakpad does on Windows

`carb.crashreporter-breakpad.plugin` is the in-process crash handler. When a fatal signal (access violation, abort, etc.) fires:

1. Breakpad catches it.
2. Writes a minidump.
3. Launches the GUI dialog process (`crashreport.gui.exe`).
4. Waits for that dialog to exit.
5. Terminates the process.

The waiting step is what made it look like Isaac Sim was "about to load if we just dismiss the dialog" — the kit process is alive but inside the crash handler, doing nothing. On Ubuntu the same plugin may be configured to permit thread crashes without terminating the process, which is why the user's prior Ubuntu experience was different.

### Why SO-100 and SO-101 are kinematically interchangeable for this proof

MuJoCo Menagerie's `trs_so_arm100` exposes six DOFs: Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, and Jaw. The workshop repo's SO-101 USD defines the same six joints under the same names. The "100 versus 101" naming refers to a hardware revision; for the kinematics relevant to RL training, the two are equivalent. The visual difference is the slightly longer wrist segment on the 101.

### Reward shaping versus reward hacking

The three failed reward designs in this session each demonstrated a different reward-hacking pattern:

- v1 paid for **being in a state** (close to cube). Agent learned to park.
- v2 paid for **maintaining a state** (grasping). Agent learned to hold.
- v3 paid for **end-state success** but did not constrain **how** that state was reached. Agent learned to push.

The general lesson is that dense shaping rewards must pay for **progress** through phases, not for **occupying** phases, and the success criterion must encode **path** requirements when the path matters (such as "the cube must have been lifted").

## Where things stand now

### Isaac Sim status

- Driver: 596.49 (clean-installed by user)
- Caches: cleared
- GPU priority: set to high-performance RTX 5070 for `kit.exe`
- Outcome: same deterministic crash in `rtx.scenedb.plugin.dll` at offset `0xe533b`, exception `0xc0000005`. Not usable on this machine. Outstanding recommendation: roll back to an earlier NVIDIA driver that worked in mid-March 2026, or open a developer support ticket with NVIDIA using the WER bucket hash and minidump files.

### MuJoCo proof of concept

```
C:\Users\Haikal\Desktop\Tech Learning\so101-rl-poc\
├── README.md
├── requirements.txt
├── env\
│   ├── __init__.py
│   └── so100_pick_place_env.py    (v4 edits in place, not yet trained)
├── scene\
│   ├── pick_place_scene.xml
│   ├── so_arm100.xml              (local copy)
│   └── assets\                    (local copy of SO-100 meshes)
├── scripts\
│   ├── test_env.py
│   ├── train.py
│   └── eval_visualize.py
├── runs\
│   ├── ppo_v1\                    (failed: 0% success, hovers near cube)
│   ├── ppo_v2\                    (failed: 0% success, grasps and holds)
│   ├── ppo_v3\                    (works: 100% success, but pushes the cube)
│   │   ├── eval\best_model.zip
│   │   └── demo.mp4               <-- the current best demo
│   ├── ppo_v1_train.log
│   ├── ppo_v2_train.log
│   └── ppo_v3_train.log
└── third_party\
    └── mujoco_menagerie\
```

Conda env: `so101-rl` (Python 3.11) with `mujoco`, `gymnasium`, `stable-baselines3[extra]`, `imageio`, `tensorboard`.

### v4 env state

- `MAX_ACTION_DELTA = 0.04` constant added.
- `MIN_LIFT_FOR_SUCCESS = 0.08` constant added.
- `TRANSPORT_LIFT_THRESH = 0.07` constant added.
- `_prev_ctrl` and `_max_cube_z` state added to env.
- `_apply_delta_action` replaces `_scale_action`.
- `reset` reinitialises both new state variables.

Still to do in v4 (this is where the note was requested):

- Update `step` to call `_apply_delta_action`, update `_max_cube_z`, gate transport reward on `cube_z > TRANSPORT_LIFT_THRESH`, gate success on `_max_cube_z >= MIN_LIFT_FOR_SUCCESS`.
- Smoke-test the updated env.
- Run a 3 million step PPO training as `ppo_v4`.
- Render a five-episode demo MP4 for the user to inspect.

## What's next

Immediate (this session):

- Finish the v4 `step` method edits.
- Smoke-test, train, render.
- Verify by watching the v4 demo that the arm is now performing a real grasp-lift-transport-place sequence at a realistic speed.

This week:

- If v4 is satisfactory, present the demo MP4 to the user's team.
- Optionally run a longer training (five million steps, different seed) to confirm robustness.

Next week or the week after, on the incoming hardware (Threadripper 3970x, 3x RTX 3090, 32 GB RAM):

- Install Isaac Lab in Docker on Linux matching the workshop repo's tested configuration.
- Port the v4 reward function and observation design from `env/so100_pick_place_env.py` into a `ManagerBasedRLEnvCfg` reward term in the workshop repo's task tree.
- Retrain on the actual SO-101 USD at much higher parallel-env counts.
- Drive the trained policy through the workshop repo's existing dataset-recording pipeline to generate the automated GR00T N1 fine-tuning dataset at scale.
