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

        Pure function: iterates using qpos as scratch space, then restores the original
        env state before returning. Does NOT mutate self.data.qpos.
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

        tcp = self._tcp_pos()
        cube = self._cube_pos()
        target = self._target_pos()
        ee_cube_dist = float(np.linalg.norm(tcp - cube))
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

        obs = self._observation()
        truncated = self._step_count >= self.max_episode_steps
        self._prev_action = action.copy()
        return obs, float(reward), False, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="iso")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
