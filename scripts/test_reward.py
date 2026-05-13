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

    must_fire = ["reward_reach", "penalty_jerk"]
    failed = [k for k in must_fire if component_fired_at_least_once[k] == 0]
    if failed:
        print(f"\nGATE 2 FAIL: components never fired: {failed}")
        sys.exit(1)
    print("\nGATE 2 PASS: reward components fire as expected for random policy")
    sys.exit(0)


if __name__ == "__main__":
    main()
