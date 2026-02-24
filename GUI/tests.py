import os
import sys
import random
from typing import List

# Ensure GUI package is importable
CUR_DIR = os.path.dirname(__file__)
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

from smoothing import compute_blended_keypoints


def make_dummy_keypoints() -> List[List[float]]:
    # 13 body keypoints with x, y, conf in pixels (Neck added)
    pts = []
    for i in range(13):
        pts.append([50.0 + i * 5.0, 100.0 + i * 3.0, 0.9])
    return pts


def test_identity_when_p_equals_one():
    base = make_dummy_keypoints()
    frame_shape = (480, 640, 3)
    # Point to any directory; function should short-circuit and not use it when p==1
    dummy_angles_dir = os.path.join(CUR_DIR, "..", "cycle", "angles_json")
    out = compute_blended_keypoints(base, frame_shape, dummy_angles_dir, p=1.0)
    assert out is not None, "Output should not be None for valid input"
    assert len(out) == len(base) == 13, "Output must contain 13 keypoints"
    for idx, (a, b) in enumerate(zip(base, out)):
        ax, ay, ac = a
        bx, by, bc = b
        assert round(ax, 1) == round(bx, 1), f"x mismatch at kp {idx}: {ax} vs {bx}"
        assert round(ay, 1) == round(by, 1), f"y mismatch at kp {idx}: {ay} vs {by}"
        assert round(ac, 1) == round(bc, 1), f"conf mismatch at kp {idx}: {ac} vs {bc}"


if __name__ == "__main__":
    test_identity_when_p_equals_one()
    print("OK: identity test passed")
