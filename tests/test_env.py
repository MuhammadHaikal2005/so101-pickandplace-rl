"""Smoke tests for the SO-100 Cartesian pick-and-place env."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco


def test_tcp_site_exists():
    # Load the composite scene (not so_arm100.xml directly) so we also exercise the
    # <include> chain that the env uses at runtime.
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene" / "pick_place_scene.xml"))
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    assert sid >= 0, "tcp site not found in scene"
    print(f"  tcp site id={sid}")
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
