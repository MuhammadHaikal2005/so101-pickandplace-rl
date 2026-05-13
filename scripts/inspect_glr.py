import inspect

import gymnasium as gym
import gym_lowcostrobot
from gym_lowcostrobot.envs.lift_cube_env import LiftCubeEnv
from gym_lowcostrobot.envs.pick_place_cube_env import PickPlaceCubeEnv


for cls in (LiftCubeEnv, PickPlaceCubeEnv):
    print(f"=== {cls.__name__} ===")
    sig = inspect.signature(cls.__init__)
    for n, p in sig.parameters.items():
        if n == "self":
            continue
        print(f"  {n} = {p.default}")
    print()

env = gym.make(
    "LiftCube-v0",
    observation_mode="state",
    action_mode="ee",
)
print("LiftCube-v0 (state + ee) action_space:", env.action_space)
print("LiftCube-v0 (state + ee) observation_space:", env.observation_space)
obs, _ = env.reset()
if isinstance(obs, dict):
    for k, v in obs.items():
        print(f"  obs[{k}] shape={getattr(v, 'shape', '?')} sample[:4]={getattr(v, 'flat', [None])[:4] if hasattr(v, 'flat') else v}")
else:
    print(f"  obs shape={obs.shape}")
env.close()
