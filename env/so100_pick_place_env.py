from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

SCENE_PATH = Path(__file__).resolve().parent.parent / "scene" / "pick_place_scene.xml"

JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
ACTUATOR_NAMES = JOINT_NAMES

HOME_QPOS = np.array([0.0, -1.57, 1.57, 1.57, -1.57, 0.0], dtype=np.float32)

CUBE_START_XY = np.array([0.06, -0.18], dtype=np.float32)
CUBE_START_NOISE = 0.005
TARGET_XY = np.array([-0.06, -0.18], dtype=np.float32)
TABLE_Z = 0.01

MIN_LIFT_FOR_SUCCESS = 0.06
PLACE_DIST_THRESH = 0.04
SUCCESS_HOLD_STEPS = 5

JAW_PAD_NAMES = [
    "fixed_jaw_pad_1", "fixed_jaw_pad_2", "fixed_jaw_pad_3", "fixed_jaw_pad_4",
    "moving_jaw_pad_1", "moving_jaw_pad_2", "moving_jaw_pad_3", "moving_jaw_pad_4",
]


class SO100PickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str | None = None, max_episode_steps: int = 500, substeps: int = 10):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.max_episode_steps = max_episode_steps
        self.substeps = substeps
        self.render_mode = render_mode

        self.joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        )
        self.qpos_addrs = np.array([self.model.jnt_qposadr[j] for j in self.joint_ids])
        self.qvel_addrs = np.array([self.model.jnt_dofadr[j] for j in self.joint_ids])
        self.act_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATOR_NAMES]
        )
        self.ctrlranges = self.model.actuator_ctrlrange[self.act_ids].astype(np.float32)

        self.cube_qpos_addr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        ]
        self.fixed_jaw_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
        self.moving_jaw_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Moving_Jaw")
        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
        self.cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.jaw_pad_geom_ids = set(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in JAW_PAD_NAMES
        )

        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

        obs_dim = 6 + 6 + 3 + 4 + 3 + 3 + 3 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self._renderer = None
        self._step_count = 0
        self._success_streak = 0
        self._max_cube_z = TABLE_Z
        self._np_random, _ = gym.utils.seeding.np_random(0)

    def _ee_pos(self) -> np.ndarray:
        return 0.5 * (self.data.xpos[self.fixed_jaw_id] + self.data.xpos[self.moving_jaw_id])

    def _cube_pos(self) -> np.ndarray:
        return self.data.qpos[self.cube_qpos_addr : self.cube_qpos_addr + 3].copy()

    def _cube_quat(self) -> np.ndarray:
        return self.data.qpos[self.cube_qpos_addr + 3 : self.cube_qpos_addr + 7].copy()

    def _target_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.target_site_id].copy()

    def _cube_pad_contact(self) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 == self.cube_geom_id and g2 in self.jaw_pad_geom_ids) or (
                g2 == self.cube_geom_id and g1 in self.jaw_pad_geom_ids
            ):
                return True
        return False

    def _observation(self) -> np.ndarray:
        qpos = self.data.qpos[self.qpos_addrs].astype(np.float32)
        qvel = self.data.qvel[self.qvel_addrs].astype(np.float32)
        cube_pos = self._cube_pos().astype(np.float32)
        cube_quat = self._cube_quat().astype(np.float32)
        ee = self._ee_pos().astype(np.float32)
        ee_to_cube = (cube_pos - ee).astype(np.float32)
        cube_to_target = (self._target_pos() - cube_pos).astype(np.float32)
        grasp_flag = np.array([1.0 if self._cube_pad_contact() else 0.0], dtype=np.float32)
        return np.concatenate(
            [qpos, qvel, cube_pos, cube_quat, ee, ee_to_cube, cube_to_target, grasp_flag]
        )

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, -1.0, 1.0)
        low = self.ctrlranges[:, 0]
        high = self.ctrlranges[:, 1]
        return low + 0.5 * (action + 1.0) * (high - low)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[self.qpos_addrs] = HOME_QPOS
        self.data.ctrl[self.act_ids] = HOME_QPOS

        cube_xy = CUBE_START_XY + self._np_random.uniform(
            -CUBE_START_NOISE, CUBE_START_NOISE, size=2
        ).astype(np.float32)
        self.data.qpos[self.cube_qpos_addr : self.cube_qpos_addr + 3] = [
            cube_xy[0],
            cube_xy[1],
            TABLE_Z,
        ]
        self.data.qpos[self.cube_qpos_addr + 3 : self.cube_qpos_addr + 7] = [1.0, 0.0, 0.0, 0.0]

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._success_streak = 0
        self._max_cube_z = TABLE_Z
        return self._observation(), {}

    def step(self, action: np.ndarray):
        scaled = self._scale_action(np.asarray(action, dtype=np.float32))
        self.data.ctrl[self.act_ids] = scaled

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        ee = self._ee_pos()
        cube = self._cube_pos()
        target = self._target_pos()
        ee_cube_dist = float(np.linalg.norm(ee - cube))
        cube_target_dist = float(np.linalg.norm(cube[:2] - target[:2]))
        cube_z = float(cube[2])
        in_contact = self._cube_pad_contact()

        self._max_cube_z = max(self._max_cube_z, cube_z)
        cube_was_lifted = self._max_cube_z >= MIN_LIFT_FOR_SUCCESS

        reward_reach = -1.0 * ee_cube_dist
        reward_contact = 0.5 if in_contact else 0.0
        reward_lift = 100.0 * max(0.0, cube_z - TABLE_Z)
        reward_place = -1.5 * cube_target_dist
        success = (
            cube_target_dist < PLACE_DIST_THRESH
            and cube_z < (TABLE_Z + 0.015)
            and cube_was_lifted
        )
        reward_success = 200.0 if success else 0.0
        action_penalty = 0.002 * float(np.sum(np.square(action)))

        reward = (
            reward_reach
            + reward_contact
            + reward_lift
            + reward_place
            + reward_success
            - action_penalty
        )

        if success:
            self._success_streak += 1
        else:
            self._success_streak = 0

        terminated = self._success_streak >= SUCCESS_HOLD_STEPS
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "ee_cube_dist": ee_cube_dist,
            "cube_target_dist": cube_target_dist,
            "cube_z": cube_z,
            "max_cube_z": self._max_cube_z,
            "in_contact": float(in_contact),
            "is_success": float(terminated),
        }
        return self._observation(), float(reward), terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="front")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
