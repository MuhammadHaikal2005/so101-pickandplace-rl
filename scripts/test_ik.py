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
            current = env._tcp_pos()
            err = target - current
            err_norm = np.linalg.norm(err)
            if err_norm > 1e-3:
                direction = err / err_norm
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
