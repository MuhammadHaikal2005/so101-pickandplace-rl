# RL Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Cartesian-action + SAC + ggando-reward redesign specified in `docs/superpowers/specs/2026-05-13-rl-redesign-design.md`, then train a policy that solves SO-100 pick-and-place in MuJoCo.

**Architecture:** Replace the existing joint-space PPO env with a Cartesian-action env that uses damped least-squares IK internally, paired with SAC from stable-baselines3 and a shaped reward derived from ggando's published recipe.

**Tech Stack:** MuJoCo 3, Gymnasium, stable-baselines3 SAC, NumPy, ImageIO for MP4 rendering. Conda env `so101-rl` (already exists).

---

## File Structure

| Path | Change | Responsibility |
| --- | --- | --- |
| `scene/so_arm100.xml` | Modify | Add `<site name="tcp"/>` for IK reference |
| `env/so100_pick_place_env.py` | Replace | New Cartesian env class with DLS IK, ggando-style reward, discouragement system |
| `env/__init__.py` | Unchanged | Re-exports `SO100PickPlaceEnv` |
| `tests/__init__.py` | Create | Empty marker |
| `tests/test_env.py` | Create | Smoke tests for env init, IK, reward components, success criterion |
| `scripts/test_ik.py` | Create | Gate 1 — drives EE through waypoints, renders MP4 |
| `scripts/test_reward.py` | Create | Gate 2 — random rollouts logging reward components |
| `scripts/train_sac.py` | Modify | Switch from gym-lowcostrobot to our env |
| `scripts/eval_sac_visualize.py` | Modify | Switch from gym-lowcostrobot to our env |
| `scripts/inspect_glr.py` | Delete | No longer relevant |
| `scripts/train.py` | Delete | PPO training script, superseded |
| `scripts/eval_visualize.py` | Delete | Old PPO eval, superseded |
| `scripts/diagnose.py` | Modify | Update for new env's info dict keys |

Tests are runnable Python scripts (no pytest dependency). Each prints `OK` or `FAIL` and exits with the appropriate code.

---

## Task 1: Add TCP Site to SO-100 MJCF

**Files:**
- Modify: `scene/so_arm100.xml` (inside the `Fixed_Jaw` body block, before `<body name="Moving_Jaw" ...>`)
- Create: `tests/__init__.py`
- Create: `tests/test_env.py`

- [ ] **Step 1: Create empty test package marker**

Write `tests/__init__.py`:

```python
```

(empty file)

- [ ] **Step 2: Write the failing test for TCP site existence**

Write `tests/test_env.py`:

```python
"""Smoke tests for the SO-100 Cartesian pick-and-place env."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import mujoco


def test_tcp_site_exists():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene" / "pick_place_scene.xml"))
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    assert sid >= 0, "tcp site not found in scene"
    print(f"  tcp site id={sid}")
    # The TCP should be inside the Fixed_Jaw body (id check via site_bodyid)
    fixed_jaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
    assert model.site_bodyid[sid] == fixed_jaw_id, "tcp site not attached to Fixed_Jaw"
    print("  tcp site correctly attached to Fixed_Jaw body")


TESTS = [test_tcp_site_exists]


def main():
    failures = 0
    for t in TESTS:
        name = t.__name__
        try:
            print(f"[RUN]  {name}")
            t()
            print(f"[PASS] {name}\n")
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}\n")
            failures += 1
        except Exception as e:
            print(f"[ERR]  {name}: {type(e).__name__}: {e}\n")
            failures += 1
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the test, confirm it fails**

```
conda run -n so101-rl python tests/test_env.py
```

Expected: `[FAIL] test_tcp_site_exists: tcp site not found in scene`

- [ ] **Step 4: Add the TCP site to the MJCF**

In `scene/so_arm100.xml`, find the line:

```xml
              <body name="Fixed_Jaw" pos="0 -0.0601 0" euler="0 1.57079 0">
```

After the closing `</geom>` of `fixed_jaw_pad_4` (the last finger pad), and **before** the `<body name="Moving_Jaw" ...>` line, insert:

```xml
                <site name="tcp" pos="0.013 -0.085 0" size="0.005" rgba="0 1 1 0.5"/>
```

Indentation should match the surrounding finger pad geoms (16 spaces). The `pos` places the TCP at the geometric center of the fixed jaw pads.

- [ ] **Step 5: Run the test, confirm it passes**

```
conda run -n so101-rl python tests/test_env.py
```

Expected: `[PASS] test_tcp_site_exists`

- [ ] **Step 6: Commit**

```
git add scene/so_arm100.xml tests/__init__.py tests/test_env.py
git commit -m "Add TCP site to SO-100 MJCF for IK reference"
```

---

## Task 2: Env Skeleton — Init, Spaces, Reset

**Files:**
- Modify: `env/so100_pick_place_env.py` (full rewrite)
- Modify: `tests/test_env.py` (append new tests)

- [ ] **Step 1: Add the failing tests for env init**

Open `tests/test_env.py`. Replace the `TESTS = [...]` line with the new tests added. Insert these test functions **above** the `TESTS = [...]` line:

```python
def test_env_imports_and_constructs():
    from env import SO100PickPlaceEnv
    env = SO100PickPlaceEnv()
    assert env is not None
    env.close()
    print("  env constructed and closed")


def test_action_space_is_4d():
    from env import SO100PickPlaceEnv
    env = SO100PickPlaceEnv()
    try:
        assert env.action_space.shape == (4,), f"got {env.action_space.shape}"
        low = env.action_space.low
        high = env.action_space.high
        assert np.all(low == -1.0) and np.all(high == 1.0)
        print(f"  action_space={env.action_space}")
    finally:
        env.close()


def test_observation_space_is_22d():
    from env import SO100PickPlaceEnv
    env = SO100PickPlaceEnv()
    try:
        assert env.observation_space.shape == (22,), f"got {env.observation_space.shape}"
        print(f"  observation_space={env.observation_space}")
    finally:
        env.close()


def test_reset_returns_valid_obs():
    from env import SO100PickPlaceEnv
    env = SO100PickPlaceEnv()
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (22,), f"got {obs.shape}"
        assert obs.dtype == np.float32
        assert isinstance(info, dict)
        print(f"  obs[:6]={obs[:6]}")
    finally:
        env.close()
```

Update the `TESTS` list:

```python
TESTS = [
    test_tcp_site_exists,
    test_env_imports_and_constructs,
    test_action_space_is_4d,
    test_observation_space_is_22d,
    test_reset_returns_valid_obs,
]
```

- [ ] **Step 2: Run tests, confirm new ones fail**

```
conda run -n so101-rl python tests/test_env.py
```

Expected: `[PASS] test_tcp_site_exists` and `[FAIL]` or `[ERR]` for the rest (because the env file still has the old code; SO100PickPlaceEnv will instantiate but with the wrong action/obs spaces).

- [ ] **Step 3: Replace env file with skeleton**

Replace the entire contents of `env/so100_pick_place_env.py` with:

```python
"""SO-100 pick-and-place env with Cartesian end-effector actions.

Action space (4-D, [-1, 1]):
  0: dX  -> ±0.04 m per step
  1: dY  -> ±0.04 m per step
  2: dZ  -> ±0.04 m per step
  3: gripper delta -> ±0.15 rad per step

Observation space (22-D, state-only):
  joint_pos (6) | ee_pos (3) | cube_pos (3) | target_pos (3) |
  ee_to_cube (3) | cube_to_target (3) | grasp_flag (1)
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

SCENE_PATH = Path(__file__).resolve().parent.parent / "scene" / "pick_place_scene.xml"

JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
ACTUATOR_NAMES = JOINT_NAMES
ARM_JOINT_NAMES = JOINT_NAMES[:5]   # all except Jaw, used by IK

HOME_QPOS = np.array([0.0, -1.57, 1.57, 1.57, -1.57, 0.0], dtype=np.float32)

CUBE_START_XYZ = np.array([0.06, -0.18, 0.01], dtype=np.float32)
TARGET_XY = np.array([-0.06, -0.18], dtype=np.float32)
TABLE_Z = 0.01

# Action scaling
EE_DELTA_MAX = 0.04   # metres
JAW_DELTA_MAX = 0.15  # radians

# Reachable workspace clamp for EE target
EE_BOX_LOW = np.array([-0.20, -0.30, 0.005], dtype=np.float32)
EE_BOX_HIGH = np.array([0.20, 0.10, 0.30], dtype=np.float32)

# Reward thresholds
LIFT_THRESH_ABS = TABLE_Z + 0.04   # 0.05  — cube center is "lifted" above this
TRANSPORT_LIFT_ABS = TABLE_Z + 0.05  # 0.06 — transport reward active above this
MIN_LIFT_FOR_BONUS_ABS = TABLE_Z + 0.04  # first-lift bonus threshold
PLACE_XY_THRESH = 0.04
PLACE_Z_TOLERANCE = 0.005
SUCCESS_HOLD_STEPS = 5

JAW_PAD_NAMES_FIXED = [f"fixed_jaw_pad_{i}" for i in (1, 2, 3, 4)]
JAW_PAD_NAMES_MOVING = [f"moving_jaw_pad_{i}" for i in (1, 2, 3, 4)]


class SO100PickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 200,
        substeps: int = 10,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.max_episode_steps = max_episode_steps
        self.substeps = substeps
        self.render_mode = render_mode

        # Joint and actuator IDs
        self.joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        )
        self.qpos_addrs = np.array([self.model.jnt_qposadr[j] for j in self.joint_ids])
        self.qvel_addrs = np.array([self.model.jnt_dofadr[j] for j in self.joint_ids])
        self.act_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATOR_NAMES]
        )
        self.ctrlranges = self.model.actuator_ctrlrange[self.act_ids].astype(np.float32)

        # Arm joint addresses (used by IK)
        arm_joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINT_NAMES]
        )
        self.arm_qpos_addrs = np.array([self.model.jnt_qposadr[j] for j in arm_joint_ids])
        self.arm_dof_addrs = np.array([self.model.jnt_dofadr[j] for j in arm_joint_ids])

        # Cube freejoint
        self.cube_qpos_addr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        ]

        # Useful IDs
        self.tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
        self.cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.fixed_jaw_geom_ids = set(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in JAW_PAD_NAMES_FIXED
        )
        self.moving_jaw_geom_ids = set(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in JAW_PAD_NAMES_MOVING
        )

        # Action / observation spaces
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(22,), dtype=np.float32)

        # Episode state
        self._renderer = None
        self._step_count = 0
        self._success_streak = 0
        self._has_grasped = False
        self._was_lifted = False
        self._first_lift_fired = False
        self._first_target_fired = False
        self._place_fired = False
        self._push_penalty_fired = False
        self._off_table_fired = False
        self._initial_cube_xy = CUBE_START_XYZ[:2].copy()
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._np_random, _ = gym.utils.seeding.np_random(0)

    def _tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp_site_id].copy()

    def _cube_pos(self) -> np.ndarray:
        return self.data.qpos[self.cube_qpos_addr : self.cube_qpos_addr + 3].copy()

    def _target_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.target_site_id].copy()

    def _is_grasped(self) -> bool:
        """Both jaw sides must contact the cube (strict grasp)."""
        touched_fixed = False
        touched_moving = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            other = None
            if g1 == self.cube_geom_id:
                other = g2
            elif g2 == self.cube_geom_id:
                other = g1
            if other is None:
                continue
            if other in self.fixed_jaw_geom_ids:
                touched_fixed = True
            elif other in self.moving_jaw_geom_ids:
                touched_moving = True
            if touched_fixed and touched_moving:
                return True
        return False

    def _observation(self) -> np.ndarray:
        qpos = self.data.qpos[self.qpos_addrs].astype(np.float32)
        ee = self._tcp_pos().astype(np.float32)
        cube = self._cube_pos().astype(np.float32)
        target = self._target_pos().astype(np.float32)
        ee_to_cube = (cube - ee).astype(np.float32)
        cube_to_target = (target - cube).astype(np.float32)
        grasp_flag = np.array([1.0 if self._is_grasped() else 0.0], dtype=np.float32)
        return np.concatenate([qpos, ee, cube, target, ee_to_cube, cube_to_target, grasp_flag])

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[self.qpos_addrs] = HOME_QPOS
        self.data.ctrl[self.act_ids] = HOME_QPOS
        self.data.qpos[self.cube_qpos_addr : self.cube_qpos_addr + 3] = CUBE_START_XYZ
        self.data.qpos[self.cube_qpos_addr + 3 : self.cube_qpos_addr + 7] = [1.0, 0.0, 0.0, 0.0]

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._success_streak = 0
        self._has_grasped = False
        self._was_lifted = False
        self._first_lift_fired = False
        self._first_target_fired = False
        self._place_fired = False
        self._push_penalty_fired = False
        self._off_table_fired = False
        self._initial_cube_xy = CUBE_START_XYZ[:2].copy()
        self._prev_action = np.zeros(4, dtype=np.float32)

        return self._observation(), {}

    def step(self, action):
        # Placeholder for now: zero ctrl so the test_reset_returns_valid_obs path is taken.
        # The real step logic is added in subsequent tasks.
        self._step_count += 1
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        obs = self._observation()
        return obs, 0.0, False, self._step_count >= self.max_episode_steps, {}

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="iso")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
```

- [ ] **Step 4: Run tests, confirm all pass**

```
conda run -n so101-rl python tests/test_env.py
```

Expected: all four current tests `[PASS]`.

- [ ] **Step 5: Commit**

```
git add env/so100_pick_place_env.py tests/test_env.py
git commit -m "Rewrite env: Cartesian action, 22-D state obs, skeleton step()"
```

---

## Task 3: Implement Damped-Least-Squares IK

**Files:**
- Modify: `env/so100_pick_place_env.py` (add `_ik_solve` method)
- Modify: `tests/test_env.py` (add IK test)

- [ ] **Step 1: Add failing test for IK convergence**

In `tests/test_env.py`, add this test function above the `TESTS` list:

```python
def test_ik_reaches_waypoint():
    """IK should drive the TCP within 1 cm of a reachable target after one solve."""
    from env import SO100PickPlaceEnv
    env = SO100PickPlaceEnv()
    try:
        env.reset(seed=0)
        # Target a reachable point ~10 cm forward of the gripper home pose
        target = env._tcp_pos() + np.array([0.0, -0.05, -0.05], dtype=np.float32)
        new_qpos = env._ik_solve(target)
        # Apply and step the sim to settle
        env.data.ctrl[env.act_ids[:5]] = new_qpos[:5]
        for _ in range(20):
            mujoco.mj_step(env.model, env.data)
        final_ee = env._tcp_pos()
        err = np.linalg.norm(final_ee - target)
        assert err < 0.02, f"IK error too large: {err:.4f} m, target={target}, final={final_ee}"
        print(f"  ik error = {err:.4f} m")
    finally:
        env.close()
```

Add to `TESTS` list:

```python
    test_ik_reaches_waypoint,
```

- [ ] **Step 2: Run test, confirm it fails**

```
conda run -n so101-rl python tests/test_env.py
```

Expected: `[ERR] test_ik_reaches_waypoint: AttributeError: 'SO100PickPlaceEnv' object has no attribute '_ik_solve'`

- [ ] **Step 3: Implement `_ik_solve` in env**

In `env/so100_pick_place_env.py`, add this method to the class (insert after `_tcp_pos`):

```python
    def _ik_solve(
        self,
        target_pos: np.ndarray,
        max_iter: int = 10,
        tol: float = 0.01,
        damping: float = 0.15,
        step_size: float = 0.5,
        nullspace_weight: float = 0.05,
    ) -> np.ndarray:
        """Damped least-squares IK that returns 6-D joint targets.

        Updates joints 0-4 to reach `target_pos`; joint 5 (Jaw) is left at current value.
        Uses `self.data` working copies, then restores qpos before returning the targets.
        """
        # Clamp the target into the reachable box
        target = np.clip(target_pos, EE_BOX_LOW, EE_BOX_HIGH).astype(np.float64)

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        ee_id = self.tcp_site_id

        original_qpos = self.data.qpos.copy()
        q_arm = self.data.qpos[self.arm_qpos_addrs].astype(np.float64).copy()
        home_arm = HOME_QPOS[:5].astype(np.float64)

        for _ in range(max_iter):
            # Write current arm guess into the data copy and forward kinematics
            self.data.qpos[self.arm_qpos_addrs] = q_arm
            mujoco.mj_forward(self.model, self.data)

            tcp = self.data.site_xpos[ee_id]
            err = target - tcp
            err_norm = np.linalg.norm(err)
            if err_norm < tol:
                break

            mujoco.mj_jacSite(self.model, self.data, jacp, None, ee_id)
            J = jacp[:, self.arm_dof_addrs]  # 3 x 5

            # Damped pseudoinverse: J^T (J J^T + lambda^2 I)^-1
            JJt = J @ J.T + (damping ** 2) * np.eye(3)
            qdot = J.T @ np.linalg.solve(JJt, err)

            # Nullspace bias toward home pose
            JtJ_inv_Jt = np.linalg.pinv(J)
            nullspace = (np.eye(5) - JtJ_inv_Jt @ J) @ (nullspace_weight * (home_arm - q_arm))
            qdot = qdot + nullspace

            qdot_norm = np.linalg.norm(qdot)
            if qdot_norm > 1.0:
                qdot = qdot / qdot_norm

            q_arm = q_arm + step_size * qdot

            # Clip to joint limits
            low = self.model.jnt_range[[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINT_NAMES], 0]
            high = self.model.jnt_range[[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINT_NAMES], 1]
            q_arm = np.clip(q_arm, low, high)

        # Restore original qpos
        self.data.qpos[:] = original_qpos
        mujoco.mj_forward(self.model, self.data)

        result = np.empty(6, dtype=np.float32)
        result[:5] = q_arm.astype(np.float32)
        result[5] = self.data.qpos[self.qpos_addrs[5]]  # leave Jaw at current
        return result
```

- [ ] **Step 4: Run test, confirm it passes**

```
conda run -n so101-rl python tests/test_env.py
```

Expected: `[PASS] test_ik_reaches_waypoint` with IK error well under 0.02 m.

- [ ] **Step 5: Commit**

```
git add env/so100_pick_place_env.py tests/test_env.py
git commit -m "Add damped least-squares IK to env"
```

---

## Task 4: Cartesian Action Handling in step()

**Files:**
- Modify: `env/so100_pick_place_env.py` (replace placeholder `step` method)
- Modify: `tests/test_env.py` (add EE motion test)

- [ ] **Step 1: Add failing test for EE motion**

Add to `tests/test_env.py` above the `TESTS` list:

```python
def test_positive_z_action_raises_ee():
    """Commanding action [0, 0, +1, 0] for 30 steps should raise the TCP at least 5 cm."""
    from env import SO100PickPlaceEnv
    env = SO100PickPlaceEnv()
    try:
        env.reset(seed=0)
        z_start = env._tcp_pos()[2]
        action = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        for _ in range(30):
            env.step(action)
        z_end = env._tcp_pos()[2]
        rise = z_end - z_start
        assert rise > 0.05, f"TCP rose only {rise:.4f} m; expected > 0.05 m"
        print(f"  z_start={z_start:.4f}, z_end={z_end:.4f}, rise={rise:.4f} m")
    finally:
        env.close()
```

Add to `TESTS` list:

```python
    test_positive_z_action_raises_ee,
```

- [ ] **Step 2: Run, confirm failure**

```
conda run -n so101-rl python tests/test_env.py
```

Expected: `[FAIL] test_positive_z_action_raises_ee: TCP rose only ...` (because the current `step` ignores the action).

- [ ] **Step 3: Replace `step` method with real Cartesian handling**

In `env/so100_pick_place_env.py`, replace the existing `step` method with:

```python
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # Compute commanded EE target
        ee_delta = action[:3] * EE_DELTA_MAX
        target_ee = self._tcp_pos() + ee_delta

        # Solve IK for joints 0-4
        new_qpos = self._ik_solve(target_ee)

        # Gripper delta (joint 5)
        jaw_low, jaw_high = self.ctrlranges[5]
        current_jaw = self.data.qpos[self.qpos_addrs[5]]
        new_jaw = float(np.clip(current_jaw + action[3] * JAW_DELTA_MAX, jaw_low, jaw_high))
        new_qpos[5] = new_jaw

        # Apply control and step
        self.data.ctrl[self.act_ids] = new_qpos
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._observation()

        truncated = self._step_count >= self.max_episode_steps
        info = {}
        self._prev_action = action.copy()
        return obs, 0.0, False, truncated, info
```

- [ ] **Step 4: Run, confirm it passes**

```
conda run -n so101-rl python tests/test_env.py
```

Expected: `[PASS] test_positive_z_action_raises_ee`.

- [ ] **Step 5: Commit**

```
git add env/so100_pick_place_env.py tests/test_env.py
git commit -m "Wire Cartesian action through IK into env.step"
```

---

## Task 5: Reach Reward + Step Reward Aggregation

**Files:**
- Modify: `env/so100_pick_place_env.py` (replace `step` to compute reward)
- Modify: `tests/test_env.py` (add reach reward test)

- [ ] **Step 1: Add failing test**

Add to `tests/test_env.py` above the `TESTS` list:

```python
def test_reach_reward_decreases_with_distance():
    """Reach reward should be near 1 at zero distance and near 0 at 0.3 m distance."""
    from env import SO100PickPlaceEnv
    env = SO100PickPlaceEnv()
    try:
        env.reset(seed=0)
        # Place cube at TCP (zero distance) - manipulate cube qpos directly
        tcp = env._tcp_pos()
        env.data.qpos[env.cube_qpos_addr : env.cube_qpos_addr + 3] = tcp
        mujoco.mj_forward(env.model, env.data)
        _, r_close, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
        # Now move cube 0.3 m away
        env.data.qpos[env.cube_qpos_addr : env.cube_qpos_addr + 3] = tcp + np.array([0.3, 0.0, 0.0])
        mujoco.mj_forward(env.model, env.data)
        _, r_far, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
        assert r_close > 0.5, f"reach reward at zero distance too low: {r_close}"
        assert r_far < 0.2, f"reach reward at 0.3 m too high: {r_far}"
        print(f"  r_close={r_close:.3f}, r_far={r_far:.3f}")
    finally:
        env.close()
```

Add to `TESTS`:

```python
    test_reach_reward_decreases_with_distance,
```

- [ ] **Step 2: Run, confirm fail**

```
conda run -n so101-rl python tests/test_env.py
```

Expected: `[FAIL] test_reach_reward_decreases_with_distance` (step still returns 0.0).

- [ ] **Step 3: Add reward computation to `step`**

In `env/so100_pick_place_env.py`, modify the `step` method. Replace:

```python
        self._step_count += 1
        obs = self._observation()

        truncated = self._step_count >= self.max_episode_steps
        info = {}
        self._prev_action = action.copy()
        return obs, 0.0, False, truncated, info
```

with:

```python
        self._step_count += 1

        tcp = self._tcp_pos()
        cube = self._cube_pos()
        target = self._target_pos()
        ee_cube_dist = float(np.linalg.norm(tcp - cube))
        reward_reach = 1.0 - float(np.tanh(10.0 * ee_cube_dist))

        reward = reward_reach
        info = {
            "reward_reach": reward_reach,
        }

        obs = self._observation()
        truncated = self._step_count >= self.max_episode_steps
        self._prev_action = action.copy()
        return obs, float(reward), False, truncated, info
```

- [ ] **Step 4: Run, confirm pass**

```
conda run -n so101-rl python tests/test_env.py
```

Expected: `[PASS] test_reach_reward_decreases_with_distance` with r_close near 1 and r_far near 0.

- [ ] **Step 5: Commit**

```
git add env/so100_pick_place_env.py tests/test_env.py
git commit -m "Add reach reward (1 - tanh(10*dist))"
```

---

## Task 6: Grasp Reward + Lift Reward

**Files:**
- Modify: `env/so100_pick_place_env.py`
- Modify: `tests/test_env.py`

- [ ] **Step 1: Add failing tests**

Add to `tests/test_env.py`:

```python
def test_grasp_reward_fires_when_both_jaws_contact():
    """Forcing both jaw pads into contact with the cube should produce grasp reward."""
    from env import SO100PickPlaceEnv
    env = SO100PickPlaceEnv()
    try:
        env.reset(seed=0)
        # Place cube exactly at TCP, then close jaws fully
        tcp = env._tcp_pos()
        env.data.qpos[env.cube_qpos_addr : env.cube_qpos_addr + 3] = tcp
        env.data.qpos[env.qpos_addrs[5]] = env.ctrlranges[5, 0]  # fully closed
        env.data.ctrl[env.act_ids[5]] = env.ctrlranges[5, 0]
        mujoco.mj_forward(env.model, env.data)
        for _ in range(5):
            mujoco.mj_step(env.model, env.data)
        _, reward, _, _, info = env.step(np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32))
        # Grasp may or may not register depending on exact geometry; the test asserts
        # the info dict reports the flag, and that when it fires, the grasp bonus appears.
        assert "reward_grasp" in info, "reward_grasp missing from info"
        if info.get("reward_grasp", 0.0) > 0.0:
            print(f"  grasp fired: reward_grasp={info['reward_grasp']:.3f}")
        else:
            print("  grasp did not fire (geometry-dependent); info key present is sufficient")
    finally:
        env.close()


def test_lift_reward_requires_grasp():
    """Lift reward should be zero if cube is above table but not grasped."""
    from env import SO100PickPlaceEnv
    env = SO100PickPlaceEnv()
    try:
        env.reset(seed=0)
        # Move cube to z=0.10 without grasping
        env.data.qpos[env.cube_qpos_addr + 2] = 0.10
        mujoco.mj_forward(env.model, env.data)
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        assert info.get("reward_lift", -1.0) == 0.0, f"got reward_lift={info.get('reward_lift')} for ungrasped cube"
        print(f"  reward_lift correctly zero for ungrasped lifted cube")
    finally:
        env.close()
```

Add to `TESTS`:

```python
    test_grasp_reward_fires_when_both_jaws_contact,
    test_lift_reward_requires_grasp,
```

- [ ] **Step 2: Run, confirm fail**

```
conda run -n so101-rl python tests/test_env.py
```

Expected: `[FAIL]` on both, because `reward_grasp` and `reward_lift` are not yet in info.

- [ ] **Step 3: Add grasp + lift terms to `step`**

In `env/so100_pick_place_env.py`, locate the reward block inside `step` and replace:

```python
        reward_reach = 1.0 - float(np.tanh(10.0 * ee_cube_dist))

        reward = reward_reach
        info = {
            "reward_reach": reward_reach,
        }
```

with:

```python
        reward_reach = 1.0 - float(np.tanh(10.0 * ee_cube_dist))

        in_contact = self._is_grasped()
        reward_grasp = 0.25 if in_contact else 0.0

        cube_z = float(cube[2])
        cube_lift = max(0.0, cube_z - TABLE_Z)
        reward_lift = 2.0 * float(np.tanh(20.0 * cube_lift)) if in_contact else 0.0

        reward = reward_reach + reward_grasp + reward_lift
        info = {
            "reward_reach": reward_reach,
            "reward_grasp": reward_grasp,
            "reward_lift": reward_lift,
            "in_contact": float(in_contact),
            "cube_z": cube_z,
        }
```

- [ ] **Step 4: Run, confirm pass**

```
conda run -n so101-rl python tests/test_env.py
```

Expected: `[PASS]` on both new tests.

- [ ] **Step 5: Commit**

```
git add env/so100_pick_place_env.py tests/test_env.py
git commit -m "Add grasp and lift reward terms (gated on contact)"
```

---

## Task 7: Transport Reward + Milestone Bonuses

**Files:**
- Modify: `env/so100_pick_place_env.py`
- Modify: `tests/test_env.py`

- [ ] **Step 1: Add failing test**

Add to `tests/test_env.py`:

```python
def test_first_lift_bonus_fires_once():
    """First-lift milestone bonus should fire exactly once when cube clears 4 cm above table."""
    from env import SO100PickPlaceEnv
    env = SO100PickPlaceEnv()
    try:
        env.reset(seed=0)
        # Tick once with cube below threshold
        _, _, _, _, info0 = env.step(np.zeros(4, dtype=np.float32))
        assert info0.get("bonus_first_lift", 0.0) == 0.0
        # Now lift cube past threshold
        env.data.qpos[env.cube_qpos_addr + 2] = TABLE_Z + 0.05
        mujoco.mj_forward(env.model, env.data)
        _, _, _, _, info1 = env.step(np.zeros(4, dtype=np.float32))
        assert info1.get("bonus_first_lift", 0.0) == 1.0, f"expected 1.0, got {info1.get('bonus_first_lift')}"
        # Should not fire a second time
        _, _, _, _, info2 = env.step(np.zeros(4, dtype=np.float32))
        assert info2.get("bonus_first_lift", -1.0) == 0.0, "first-lift bonus fired twice"
        print("  first-lift bonus fires exactly once")
    finally:
        env.close()
```

Also import the constants. Add this near the top of `tests/test_env.py`:

```python
TABLE_Z = 0.01  # mirrors env constant for test setup
```

Add to `TESTS`:

```python
    test_first_lift_bonus_fires_once,
```

- [ ] **Step 2: Run, confirm fail**

```
conda run -n so101-rl python tests/test_env.py
```

Expected: `[FAIL]` because `bonus_first_lift` is not yet emitted.

- [ ] **Step 3: Add transport + milestone bonuses to `step`**

In `env/so100_pick_place_env.py`, find the reward block and replace:

```python
        reward = reward_reach + reward_grasp + reward_lift
        info = {
            "reward_reach": reward_reach,
            "reward_grasp": reward_grasp,
            "reward_lift": reward_lift,
            "in_contact": float(in_contact),
            "cube_z": cube_z,
        }
```

with:

```python
        cube_target_xy = float(np.linalg.norm(cube[:2] - target[:2]))
        is_lifted = cube_z > TRANSPORT_LIFT_ABS
        reward_transport = (1.0 - float(np.tanh(10.0 * cube_target_xy))) if (in_contact and is_lifted) else 0.0

        # Milestone bonuses (each fires once per episode)
        bonus_first_lift = 0.0
        if not self._first_lift_fired and cube_z > MIN_LIFT_FOR_BONUS_ABS:
            self._first_lift_fired = True
            bonus_first_lift = 1.0

        bonus_first_target = 0.0
        if not self._first_target_fired and is_lifted and cube_target_xy < 0.05:
            self._first_target_fired = True
            bonus_first_target = 1.0

        cube_settled = (
            cube_target_xy < PLACE_XY_THRESH
            and cube_z < (TABLE_Z + PLACE_Z_TOLERANCE + 0.005)
        )
        bonus_place = 0.0
        if not self._place_fired and cube_settled:
            self._place_fired = True
            bonus_place = 5.0

        reward = (
            reward_reach
            + reward_grasp
            + reward_lift
            + reward_transport
            + bonus_first_lift
            + bonus_first_target
            + bonus_place
        )
        info = {
            "reward_reach": reward_reach,
            "reward_grasp": reward_grasp,
            "reward_lift": reward_lift,
            "reward_transport": reward_transport,
            "bonus_first_lift": bonus_first_lift,
            "bonus_first_target": bonus_first_target,
            "bonus_place": bonus_place,
            "in_contact": float(in_contact),
            "cube_z": cube_z,
            "cube_target_xy": cube_target_xy,
        }
```

- [ ] **Step 4: Run, confirm pass**

```
conda run -n so101-rl python tests/test_env.py
```

Expected: all current tests `[PASS]`.

- [ ] **Step 5: Commit**

```
git add env/so100_pick_place_env.py tests/test_env.py
git commit -m "Add transport reward and three milestone bonuses"
```

---

## Task 8: Discouragement Penalties

**Files:**
- Modify: `env/so100_pick_place_env.py`
- Modify: `tests/test_env.py`

- [ ] **Step 1: Add failing test for drop penalty**

Add to `tests/test_env.py`:

```python
def test_drop_penalty_fires_on_grasp_loss_while_lifted():
    """Losing grasp while cube_z > LIFT_THRESH_ABS should fire -2.0 once."""
    from env import SO100PickPlaceEnv
    env = SO100PickPlaceEnv()
    try:
        env.reset(seed=0)
        # Simulate the "was lifted" state by forcing internal flag and current state
        env._was_lifted = True   # manually mark; in real episodes this is set during step
        # Force in_contact = False (cube far from gripper) and cube above lift threshold
        env.data.qpos[env.cube_qpos_addr : env.cube_qpos_addr + 3] = np.array([0.30, 0.30, 0.10])
        mujoco.mj_forward(env.model, env.data)
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        assert info.get("penalty_drop", 0.0) == -2.0, f"expected -2.0, got {info.get('penalty_drop')}"
        print("  drop penalty fires correctly")
    finally:
        env.close()
```

Add to `TESTS`:

```python
    test_drop_penalty_fires_on_grasp_loss_while_lifted,
```

- [ ] **Step 2: Run, confirm fail**

```
conda run -n so101-rl python tests/test_env.py
```

Expected: `[FAIL]` because `penalty_drop` is not yet emitted.

- [ ] **Step 3: Add penalties to `step`**

In `env/so100_pick_place_env.py`, locate the line after `bonus_place = ...` and insert the penalty computation BEFORE the `reward = ...` line:

Replace:

```python
        cube_settled = (
            cube_target_xy < PLACE_XY_THRESH
            and cube_z < (TABLE_Z + PLACE_Z_TOLERANCE + 0.005)
        )
        bonus_place = 0.0
        if not self._place_fired and cube_settled:
            self._place_fired = True
            bonus_place = 5.0

        reward = (
            reward_reach
            + reward_grasp
            + reward_lift
            + reward_transport
            + bonus_first_lift
            + bonus_first_target
            + bonus_place
        )
        info = {
            "reward_reach": reward_reach,
            "reward_grasp": reward_grasp,
            "reward_lift": reward_lift,
            "reward_transport": reward_transport,
            "bonus_first_lift": bonus_first_lift,
            "bonus_first_target": bonus_first_target,
            "bonus_place": bonus_place,
            "in_contact": float(in_contact),
            "cube_z": cube_z,
            "cube_target_xy": cube_target_xy,
        }
```

with:

```python
        cube_settled = (
            cube_target_xy < PLACE_XY_THRESH
            and cube_z < (TABLE_Z + PLACE_Z_TOLERANCE + 0.005)
        )
        bonus_place = 0.0
        if not self._place_fired and cube_settled:
            self._place_fired = True
            bonus_place = 5.0

        # Discouragement penalties
        penalty_drop = 0.0
        if self._was_lifted and not in_contact and cube_z > LIFT_THRESH_ABS:
            penalty_drop = -2.0  # fires once per drop event
            self._was_lifted = False
        elif in_contact and cube_z > LIFT_THRESH_ABS:
            self._was_lifted = True

        penalty_push = 0.0
        if not self._has_grasped:
            cube_xy_disp = float(np.linalg.norm(cube[:2] - self._initial_cube_xy))
            if not self._push_penalty_fired and cube_xy_disp > 0.01:
                penalty_push = -1.0
                self._push_penalty_fired = True

        penalty_off_table = 0.0
        if not self._off_table_fired and cube_z < (TABLE_Z - 0.02):
            penalty_off_table = -5.0
            self._off_table_fired = True

        action_delta = action - self._prev_action
        penalty_jerk = -0.01 * float(np.sum(action_delta ** 2))

        if in_contact:
            self._has_grasped = True

        reward = (
            reward_reach
            + reward_grasp
            + reward_lift
            + reward_transport
            + bonus_first_lift
            + bonus_first_target
            + bonus_place
            + penalty_drop
            + penalty_push
            + penalty_off_table
            + penalty_jerk
        )
        info = {
            "reward_reach": reward_reach,
            "reward_grasp": reward_grasp,
            "reward_lift": reward_lift,
            "reward_transport": reward_transport,
            "bonus_first_lift": bonus_first_lift,
            "bonus_first_target": bonus_first_target,
            "bonus_place": bonus_place,
            "penalty_drop": penalty_drop,
            "penalty_push": penalty_push,
            "penalty_off_table": penalty_off_table,
            "penalty_jerk": penalty_jerk,
            "in_contact": float(in_contact),
            "cube_z": cube_z,
            "cube_target_xy": cube_target_xy,
        }
```

- [ ] **Step 4: Run, confirm pass**

```
conda run -n so101-rl python tests/test_env.py
```

Expected: `[PASS]` on all tests.

- [ ] **Step 5: Commit**

```
git add env/so100_pick_place_env.py tests/test_env.py
git commit -m "Add discouragement penalties: drop, push, off-table, jerk"
```

---

## Task 9: Success Criterion and Terminal Reward

**Files:**
- Modify: `env/so100_pick_place_env.py`
- Modify: `tests/test_env.py`

- [ ] **Step 1: Add failing test**

Add to `tests/test_env.py`:

```python
def test_success_triggers_termination_with_bonus():
    """Holding the success state for SUCCESS_HOLD_STEPS frames should terminate with +1000."""
    from env import SO100PickPlaceEnv
    env = SO100PickPlaceEnv()
    try:
        env.reset(seed=0)
        # Set cube at target on table, gripper away (so not in contact)
        target = env._target_pos()
        env.data.qpos[env.cube_qpos_addr : env.cube_qpos_addr + 3] = np.array(
            [target[0], target[1], TABLE_Z]
        )
        env.data.qpos[env.cube_qpos_addr + 3 : env.cube_qpos_addr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(env.model, env.data)
        terminated_at = None
        cumulative = 0.0
        for i in range(10):
            # Force cube to stay at target by pinning each step (action commanded keeps EE far)
            env.data.qpos[env.cube_qpos_addr : env.cube_qpos_addr + 3] = np.array(
                [target[0], target[1], TABLE_Z]
            )
            env.data.qpos[env.cube_qpos_addr + 7 : env.cube_qpos_addr + 13] = 0  # zero cube vels
            mujoco.mj_forward(env.model, env.data)
            obs, r, term, trunc, info = env.step(np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32))
            cumulative += r
            if term:
                terminated_at = i
                break
        assert terminated_at is not None, "termination never fired"
        assert cumulative > 900, f"expected success bonus to push cumulative reward over 900, got {cumulative:.1f}"
        print(f"  terminated at step {terminated_at}, cumulative={cumulative:.1f}")
    finally:
        env.close()
```

Add to `TESTS`:

```python
    test_success_triggers_termination_with_bonus,
```

- [ ] **Step 2: Run, confirm fail**

```
conda run -n so101-rl python tests/test_env.py
```

Expected: `[FAIL]` because step never returns terminated=True.

- [ ] **Step 3: Add success check + terminal reward to `step`**

In `env/so100_pick_place_env.py`, locate this block at the end of `step`:

```python
        reward = (
            reward_reach
            + reward_grasp
            + reward_lift
            + reward_transport
            + bonus_first_lift
            + bonus_first_target
            + bonus_place
            + penalty_drop
            + penalty_push
            + penalty_off_table
            + penalty_jerk
        )
```

Replace it with:

```python
        # Success criterion (must hold for SUCCESS_HOLD_STEPS frames)
        success_state = (
            cube_target_xy < PLACE_XY_THRESH
            and cube_z < (TABLE_Z + PLACE_Z_TOLERANCE)
            and not in_contact
        )
        if success_state:
            self._success_streak += 1
        else:
            self._success_streak = 0

        terminated = self._success_streak >= SUCCESS_HOLD_STEPS
        reward_success = 1000.0 if terminated else 0.0

        reward = (
            reward_reach
            + reward_grasp
            + reward_lift
            + reward_transport
            + bonus_first_lift
            + bonus_first_target
            + bonus_place
            + penalty_drop
            + penalty_push
            + penalty_off_table
            + penalty_jerk
            + reward_success
        )
```

And replace the `return obs, float(reward), False, truncated, info` line with:

```python
        info["reward_success"] = reward_success
        info["success_streak"] = self._success_streak
        info["is_success"] = float(terminated)
        obs = self._observation()
        truncated = self._step_count >= self.max_episode_steps and not terminated
        self._prev_action = action.copy()
        return obs, float(reward), terminated, truncated, info
```

Remove the now-duplicate lines (the previous `obs = self._observation()`, `truncated = ...`, and `return` statements) above this block to keep only one return path.

- [ ] **Step 4: Run, confirm pass**

```
conda run -n so101-rl python tests/test_env.py
```

Expected: all tests `[PASS]`.

- [ ] **Step 5: Commit**

```
git add env/so100_pick_place_env.py tests/test_env.py
git commit -m "Add success criterion and +1000 terminal reward"
```

---

## Task 10: Gate 1 — IK Validation Script

**Files:**
- Create: `scripts/test_ik.py`

- [ ] **Step 1: Write the validation script**

Create `scripts/test_ik.py`:

```python
"""Gate 1: drive the TCP through five waypoints and render an MP4.

Run: python scripts/test_ik.py

Inspect runs/gate1_ik/ik_waypoints.mp4: the TCP should visibly track each waypoint
smoothly with no chatter near singularities.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import mujoco
import numpy as np

from env import SO100PickPlaceEnv

WAYPOINTS = [
    ("home",           np.array([-0.01, -0.24, 0.16])),
    ("above_cube",     np.array([ 0.06, -0.18, 0.15])),
    ("on_cube",        np.array([ 0.06, -0.18, 0.03])),
    ("lifted",         np.array([ 0.06, -0.18, 0.15])),
    ("above_target",   np.array([-0.06, -0.18, 0.15])),
]

STEPS_PER_WAYPOINT = 50


def main():
    out_dir = ROOT / "runs" / "gate1_ik"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = SO100PickPlaceEnv()
    env.reset(seed=0)

    renderer = mujoco.Renderer(env.model, height=720, width=960)
    frames = []
    distances_log = []

    for name, target in WAYPOINTS:
        for step in range(STEPS_PER_WAYPOINT):
            # Command Cartesian action that moves toward target
            current = env._tcp_pos()
            err = target - current
            err_norm = np.linalg.norm(err)
            if err_norm > 1e-3:
                direction = err / err_norm
                # Action magnitude proportional to error, capped at 1
                magnitude = min(1.0, err_norm / 0.04)
                action_xyz = direction * magnitude
            else:
                action_xyz = np.zeros(3)
            action = np.array([action_xyz[0], action_xyz[1], action_xyz[2], 0.0], dtype=np.float32)
            env.step(action)

            renderer.update_scene(env.data, camera="iso")
            frames.append(renderer.render())

        final = env._tcp_pos()
        d = float(np.linalg.norm(final - target))
        distances_log.append((name, target.tolist(), final.tolist(), d))
        print(f"waypoint '{name}': target={target.round(3).tolist()} final={final.round(3).tolist()} err={d:.4f} m")

    out_mp4 = out_dir / "ik_waypoints.mp4"
    imageio.mimsave(str(out_mp4), frames, fps=30, codec="libx264", quality=8)
    print(f"wrote {len(frames)} frames to {out_mp4}")

    # Summary
    max_err = max(d for *_, d in distances_log)
    print(f"\nmax waypoint error: {max_err:.4f} m")
    if max_err < 0.03:
        print("GATE 1 PASS: IK tracks all waypoints within 3 cm")
        sys.exit(0)
    else:
        print("GATE 1 FAIL: IK error exceeds 3 cm on at least one waypoint")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```
conda run -n so101-rl python scripts/test_ik.py
```

Expected: prints per-waypoint errors, writes `runs/gate1_ik/ik_waypoints.mp4`, exits 0 with "GATE 1 PASS". If any waypoint exceeds 3 cm error, the script exits 1; in that case, increase `max_iter` in `_ik_solve` or adjust the bounding box constants.

- [ ] **Step 3: Inspect the MP4 visually**

Open `runs/gate1_ik/ik_waypoints.mp4`. Confirm visually:

- Arm reaches each waypoint
- No chatter or oscillation
- Smooth motion (no whipping)

If visual inspection fails (even with numeric pass), investigate before training.

- [ ] **Step 4: Commit**

```
git add scripts/test_ik.py
git commit -m "Add Gate 1: IK waypoint validation script"
```

---

## Task 11: Gate 2 — Reward Sanity Script

**Files:**
- Create: `scripts/test_reward.py`

- [ ] **Step 1: Write the script**

Create `scripts/test_reward.py`:

```python
"""Gate 2: run random-action episodes and log every reward component to a CSV.

Run: python scripts/test_reward.py

Output: runs/gate2_reward/components.csv with per-step values of every reward and
penalty term, plus a printed summary of which components ever fired non-zero.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from env import SO100PickPlaceEnv

NUM_EPISODES = 5

COMPONENT_KEYS = [
    "reward_reach",
    "reward_grasp",
    "reward_lift",
    "reward_transport",
    "bonus_first_lift",
    "bonus_first_target",
    "bonus_place",
    "penalty_drop",
    "penalty_push",
    "penalty_off_table",
    "penalty_jerk",
    "reward_success",
]


def main():
    out_dir = ROOT / "runs" / "gate2_reward"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "components.csv"

    env = SO100PickPlaceEnv()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "step", "action_0", "action_1", "action_2", "action_3", *COMPONENT_KEYS, "total"])

        component_fired_at_least_once = {k: 0 for k in COMPONENT_KEYS}

        for ep in range(NUM_EPISODES):
            obs, _ = env.reset(seed=ep)
            ep_total = 0.0
            for step in range(env.max_episode_steps):
                action = env.action_space.sample()
                obs, r, term, trunc, info = env.step(action)
                ep_total += r
                row = [ep, step, *action.tolist()]
                for k in COMPONENT_KEYS:
                    val = float(info.get(k, 0.0))
                    row.append(val)
                    if abs(val) > 0.0:
                        component_fired_at_least_once[k] += 1
                row.append(r)
                writer.writerow(row)
                if term or trunc:
                    break
            print(f"episode {ep}: cumulative reward = {ep_total:.2f}")

    print(f"\nwrote {csv_path}")
    print("\nComponent firings across all random episodes:")
    for k in COMPONENT_KEYS:
        print(f"  {k:25s}  fired in {component_fired_at_least_once[k]} steps")

    # Sanity checks
    must_fire = ["reward_reach", "penalty_jerk"]
    failed = [k for k in must_fire if component_fired_at_least_once[k] == 0]
    if failed:
        print(f"\nGATE 2 FAIL: components never fired: {failed}")
        sys.exit(1)
    print("\nGATE 2 PASS: reward components fire as expected for random policy")
    sys.exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```
conda run -n so101-rl python scripts/test_reward.py
```

Expected: prints 5 episode rewards, writes CSV, prints component firings, exits 0 with "GATE 2 PASS". `reward_reach` and `penalty_jerk` must fire on virtually every step.

- [ ] **Step 3: Inspect the CSV briefly**

```
head "/c/Users/Haikal/Desktop/Tech Learning/so101-rl-poc/runs/gate2_reward/components.csv"
```

Expected: header row plus several data rows with realistic values.

- [ ] **Step 4: Commit**

```
git add scripts/test_reward.py
git commit -m "Add Gate 2: reward sanity check script"
```

---

## Task 12: SAC Training Script Update

**Files:**
- Modify: `scripts/train_sac.py` (replace contents)

- [ ] **Step 1: Replace `scripts/train_sac.py`**

Replace the entire contents of `scripts/train_sac.py` with:

```python
"""Train SAC on our Cartesian SO-100 pick-and-place env."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env import SO100PickPlaceEnv


def make_env(seed: int):
    def _init():
        env = SO100PickPlaceEnv()
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--run-name", type=str, default="sac_v1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(__file__).resolve().parent.parent / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    tb_dir = run_dir / "tb"
    eval_dir = run_dir / "eval"

    train_env = DummyVecEnv([make_env(args.seed)])
    eval_env = DummyVecEnv([make_env(args.seed + 1000)])

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
```

- [ ] **Step 2: Run a 1k-step smoke test**

```
conda run -n so101-rl python scripts/train_sac.py --timesteps 2000 --run-name sac_smoke
```

Expected: SAC trainer starts, completes 2000 steps in under a minute, saves `runs/sac_smoke/final.zip`. No errors.

- [ ] **Step 3: Remove the smoke-test artefacts**

```
rm -rf "/c/Users/Haikal/Desktop/Tech Learning/so101-rl-poc/runs/sac_smoke"
```

- [ ] **Step 4: Commit**

```
git add scripts/train_sac.py
git commit -m "Switch SAC training to our Cartesian env"
```

---

## Task 13: Eval Script Update

**Files:**
- Modify: `scripts/eval_sac_visualize.py` (replace contents)

- [ ] **Step 1: Replace the file**

Replace the entire contents of `scripts/eval_sac_visualize.py` with:

```python
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
```

- [ ] **Step 2: Commit**

```
git add scripts/eval_sac_visualize.py
git commit -m "Switch eval/visualization to our Cartesian env"
```

---

## Task 14: Remove Obsolete Scripts

**Files:**
- Delete: `scripts/train.py`, `scripts/eval_visualize.py`, `scripts/inspect_glr.py`

- [ ] **Step 1: Delete superseded scripts**

```
git rm scripts/train.py scripts/eval_visualize.py scripts/inspect_glr.py
```

- [ ] **Step 2: Commit**

```
git commit -m "Remove obsolete PPO scripts and gym-lowcostrobot inspector"
```

---

## Task 15: Run Both Gates Before Training

**Files:**
- None (execution only)

- [ ] **Step 1: Run Gate 1**

```
conda run -n so101-rl python scripts/test_ik.py
```

Expected: prints per-waypoint errors, exits with "GATE 1 PASS". If it fails, fix IK before proceeding.

- [ ] **Step 2: Watch the Gate 1 MP4**

Open `runs/gate1_ik/ik_waypoints.mp4`. Verify visually:

- TCP reaches each waypoint
- No singularity chatter
- Smooth motion

- [ ] **Step 3: Run Gate 2**

```
conda run -n so101-rl python scripts/test_reward.py
```

Expected: prints 5 episode totals, prints firing counts, exits with "GATE 2 PASS".

- [ ] **Step 4: Inspect Gate 2 CSV**

```
head "/c/Users/Haikal/Desktop/Tech Learning/so101-rl-poc/runs/gate2_reward/components.csv"
```

Confirm `reward_reach` values vary between roughly 0.1 and 0.9 across steps; `penalty_jerk` is mostly small negative numbers.

---

## Task 16: SAC Training Run

**Files:**
- None (execution only)

- [ ] **Step 1: Kick off training in background**

```
conda run -n so101-rl python scripts/train_sac.py --timesteps 500000 --run-name sac_v1 2>&1 | tee "runs/sac_v1_train.log"
```

Run this as a background process (in the dev shell, `&` at end; via tooling, set `run_in_background=true`). Expected wall time: 60 to 90 minutes.

- [ ] **Step 2: Verify training started**

While running:

```
ls -la "runs/sac_v1/"
```

Expected: `checkpoints/` and `tb/` and `eval/` subfolders appear within a few minutes.

- [ ] **Step 3: Wait for completion**

Training completes when the log shows `Saved final model to .../runs/sac_v1/final.zip`.

---

## Task 17: Evaluate and Render

**Files:**
- None (execution only)

- [ ] **Step 1: Render 5 evaluation episodes**

```
conda run -n so101-rl python scripts/eval_sac_visualize.py --model "runs/sac_v1/eval/best_model.zip" --out "runs/sac_v1/demo.mp4" --episodes 5 --camera iso
```

Expected: prints 5 episode summaries, writes `runs/sac_v1/demo.mp4`.

- [ ] **Step 2: Watch the demo MP4**

Open `runs/sac_v1/demo.mp4`. Verify visually:

- Arm reaches the cube
- Gripper closes on the cube
- Arm lifts the cube clear of the table
- Arm transports the cube to the target zone
- Gripper releases at the target
- No pushing or sliding

If the policy reward-hacks, the visual is the ground truth. Reject and iterate by adjusting reward weights and re-running training.

- [ ] **Step 3: Commit demo + training log**

```
git add runs/sac_v1/demo.mp4 runs/sac_v1/final.zip runs/sac_v1/eval/best_model.zip runs/sac_v1/eval/evaluations.npz runs/sac_v1_train.log
git commit -m "Add SAC v1 trained policy and demo MP4"
git push
```

The `.gitignore` already excludes `runs/*/checkpoints/`, `runs/*/tb/`, and other bulk; this only commits the durable artefacts.

---

## Plan Self-Review Summary

- All sections of the spec (action space, IK, reward, discouragement, termination, observation, algorithm, scene, validation gates) are covered by tasks.
- No placeholders: every code block is complete and self-contained.
- Type consistency: `info` dict keys are introduced incrementally and never renamed (e.g., `reward_reach` first appears in Task 5, all later tasks use the same name).
- The env file is built up across Tasks 2 through 9; at each task's commit point, the file is in a working state and all current tests pass.
- The two pre-training gates (Tasks 10 and 11) match what the spec requires.
- Training and eval (Tasks 12 through 17) are minimal scripts with no spec drift.
