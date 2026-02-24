import os
import sys
import math
import tempfile
from typing import List, Tuple, Optional, Dict

# Robust import of nearest search and Pose/Keypoint
try:
    from cycle.nearest_search import find_nearest_anchors
except Exception:
    try:
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cycle'))
        from nearest_search import find_nearest_anchors  # type: ignore
    except Exception:
        find_nearest_anchors = None  # type: ignore

try:
    from cycle.initialization.anchorClasses import Pose, Keypoint
except Exception:
    try:
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cycle', 'initialization'))
        from anchorClasses import Pose, Keypoint  # type: ignore
    except Exception:
        Pose = None  # type: ignore
        Keypoint = None  # type: ignore

COCO_NAMES = [
    "LShoulder", "RShoulder", "LElbow", "RElbow",
    "LWrist", "RWrist", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle", "Neck"
]


def _wrapped_angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _rotate_vec(vx: float, vy: float, delta: float) -> Tuple[float, float]:
    c, s = math.cos(delta), math.sin(delta)
    return vx * c - vy * s, vx * s + vy * c


def _list_angles_jsons(angles_json_dir: str) -> List[str]:
    try:
        return [
            os.path.join(angles_json_dir, f)
            for f in os.listdir(angles_json_dir)
            if f.lower().endswith('.json')
        ] if os.path.isdir(angles_json_dir) else []
    except Exception:
        return []


def _build_temp_label_from_keypoints(keypoints: List[List[float]], frame_shape: Tuple[int, int, int]) -> Optional[str]:
    # Expecting 13 body keypoints (face removed, Neck included)
    if keypoints is None or len(keypoints) < 13:
        return None
    h, w = frame_shape[:2]
    xs = [kp[0] for kp in keypoints if kp[2] > 0]
    ys = [kp[1] for kp in keypoints if kp[2] > 0]
    if not xs or not ys:
        return None
    x_min, x_max = max(0, min(xs)), min(w - 1, max(xs))
    y_min, y_max = max(0, min(ys)), min(h - 1, max(ys))
    pad = 0.05
    bw = max(1.0, (x_max - x_min) * (1 + pad))
    bh = max(1.0, (y_max - y_min) * (1 + pad))
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    x_center = cx / w
    y_center = cy / h
    width = min(1.0, bw / w)
    height = min(1.0, bh / h)
    def vis_from_conf(c):
        return 2 if c is not None and c > 0.5 else 1
    parts = [0, x_center, y_center, width, height]
    for i in range(13):
        x, y, conf = keypoints[i]
        parts.extend([x / w, y / h, vis_from_conf(conf)])
    try:
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        tmp.write(" ".join(str(v) for v in parts) + "\n")
        tmp.flush(); tmp.close()
        return tmp.name
    except Exception:
        return None


def _compute_pred_angles(base_keypoints: List[List[float]], frame_shape: Tuple[int, int, int]) -> Dict[str, float]:
    if Pose is None:
        return {}
    h, w = frame_shape[:2]
    pose = Pose()  # type: ignore
    for i, name in enumerate(COCO_NAMES):
        x, y, conf = base_keypoints[i]
        pose.keypoints[name] = Keypoint(x=x / w, y=y / h, v=2 if conf and conf > 0.5 else 1)  # type: ignore
    pose.centralize(); pose.rotate()
    ang = pose.calculate_angles() or {}
    return {k: float(v) for k, v in ang.items()}


def _apply_blended_angles(base_keypoints: List[List[float]], blended: Dict[str, float]) -> List[List[float]]:
    """Apply angle deltas (radians) to the base skeleton in pixel space.
    blended contains deltas relative to current segment/joint orientation.
    Orientation keys rotate the entire upper segment; flexion keys rotate the lower
    segment around the joint. Segment lengths are preserved.
    """
    pts = [list(kp) for kp in base_keypoints]

    def vec(a, b):
        return b[0]-a[0], b[1]-a[1]
    def set_point(idx, x, y):
        pts[idx][0] = x; pts[idx][1] = y

    def rotate_segment(origin_idx, end_idx, delta):
        ox, oy = pts[origin_idx][0], pts[origin_idx][1]
        vx, vy = vec(pts[origin_idx], pts[end_idx])
        rx, ry = _rotate_vec(vx, vy, delta)
        set_point(end_idx, ox + rx, oy + ry)

    def rotate_child_by_delta(joint_idx, child_idx, delta):
        lx, ly = vec(pts[joint_idx], pts[child_idx])
        rx, ry = _rotate_vec(lx, ly, delta)
        set_point(child_idx, pts[joint_idx][0] + rx, pts[joint_idx][1] + ry)

    # 13-body-keypoint indices (face KPs removed; body re-indexed 0–12)
    LShoulder, RShoulder = 0, 1
    LElbow, RElbow = 2, 3
    LWrist, RWrist = 4, 5
    LHip, RHip = 6, 7
    LKnee, RKnee = 8, 9
    LAnkle, RAnkle = 10, 11
    Neck = 12

    # Orientation deltas (upper segments)
    if 'L_shoulder_to_elbow' in blended:
        rotate_segment(LShoulder, LElbow, blended['L_shoulder_to_elbow'])
    if 'R_shoulder_to_elbow' in blended:
        rotate_segment(RShoulder, RElbow, blended['R_shoulder_to_elbow'])
    if 'L_hip_flexion' in blended:
        rotate_segment(LHip, LKnee, blended['L_hip_flexion'])
    if 'R_hip_flexion' in blended:
        rotate_segment(RHip, RKnee, blended['R_hip_flexion'])

    # Flexion deltas (lower segments)
    if 'L_elbow_flexion' in blended:
        rotate_child_by_delta(LElbow, LWrist, blended['L_elbow_flexion'])
    if 'R_elbow_flexion' in blended:
        rotate_child_by_delta(RElbow, RWrist, blended['R_elbow_flexion'])
    if 'L_knee_flexion' in blended:
        rotate_child_by_delta(LKnee, LAnkle, blended['L_knee_flexion'])
    if 'R_knee_flexion' in blended:
        rotate_child_by_delta(RKnee, RAnkle, blended['R_knee_flexion'])

    return pts


def compute_blended_keypoints(base_keypoints: List[List[float]],
                               frame_shape: Tuple[int, int, int],
                               angles_json_dir: str,
                               p: float = 0.5) -> Optional[List[List[float]]]:
    """Compute smoothed keypoints by blending model angles with nearest anchor angles.
    - base_keypoints: list of 13 [x,y,conf] (body-only, Neck included)
    - frame_shape: (H, W, C)
    - angles_json_dir: directory containing one or more angles JSONs
    - p: weight on model angles (0..1). 1.0 => original, 0.0 => anchor angles
    Returns: list of 13 [x,y,conf] in pixels, or None if unavailable.
    """
    # Basic validation: we expect 13 body keypoints (Neck included)
    if base_keypoints is None or len(base_keypoints) < 13:
        return None

    # If p == 1.0 return identity early (don't require anchor search to be installed)
    try:
        p_w = float(p)
    except Exception:
        p_w = 0.5
    if p_w >= 1.0:
        return base_keypoints

    # If we're here we need the anchor search implementation
    if find_nearest_anchors is None:
        return None

    # Create temp YOLO label from current keypoints
    temp_label = _build_temp_label_from_keypoints(base_keypoints, frame_shape)
    if temp_label is None:
        return None

    # Find best anchor across all JSONs
    json_paths = _list_angles_jsons(angles_json_dir)
    best_item = None
    try:
        for jp in json_paths:
            try:
                res = find_nearest_anchors(temp_label, jp, None, topk=1)  # type: ignore
            except Exception:
                continue
            if not res or not res[0]['nearest']:
                continue
            cand = res[0]['nearest'][0]
            if best_item is None or cand['distance'] < best_item['distance']:
                best_item = cand
    finally:
        try:
            os.remove(temp_label)
        except Exception:
            pass

    if best_item is None:
        return None

    anchor_angles = best_item.get('angles', {}) or {}

    # Compute model prediction angles
    pred_angles = _compute_pred_angles(base_keypoints, frame_shape)
    if not pred_angles:
        return None

    # Compute deltas: (1-p) * wrap_diff(anchor, pred); p=1 => zero deltas
    deltas: Dict[str, float] = {}
    p_w = float(p)
    for k, pred_val in pred_angles.items():
        a_val = anchor_angles.get(k)
        if a_val is None:
            continue
        deltas[k] = (1.0 - p_w) * _wrapped_angle_diff(float(a_val), float(pred_val))

    # Apply deltas to produce new keypoints
    blended_pts = _apply_blended_angles(base_keypoints, deltas)
    return blended_pts
