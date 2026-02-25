"""
Standalone occlusion detection module for the SwimHPE pipeline.

Supported camera views: Side_water_level, Side_underwater, Side_above_water.
Front and Aerial views are excluded from this pipeline entirely.

kp_coords convention throughout this module:
    {keypoint_name: (x_px, y_img_px)}
    Pixel coordinates in image space (Y-down, origin at top-left).
    Y must already be flipped from SwimXYZ's Y-up annotation space:
        y_img_px = img_height - y_annotation_px
"""

from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path

# ---------------------------------------------------------------------------
# Camera view constants
# ---------------------------------------------------------------------------

# Maps SwimXYZ folder name → normalized view name.
# Only the three side views are supported; Front and Aerial are excluded.
KNOWN_VIEWS: dict[str, str] = {
    'Side_water_level': 'water_level',
    'Side_underwater':  'underwater',
    'Side_above_water': 'above_water',
}

SIDE_VIEWS: set[str] = {'water_level', 'underwater', 'above_water'}

# ---------------------------------------------------------------------------
# Self-occlusion thresholds (pixels)
# ---------------------------------------------------------------------------

SELF_OCC_PX     = 20   # Side_water_level / Side_underwater (general joints)
SELF_OCC_PX_AW  = 5    # Side_above_water — very tight; neck check handles water

HIP_SELF_OCC_PX    = 40  # Side_water_level / Side_underwater (hips occlude more broadly)
HIP_SELF_OCC_PX_AW = 8   # Side_above_water (hips, tighter)

# ---------------------------------------------------------------------------
# Brightness-based water-surface parameters (Side_underwater)
# ---------------------------------------------------------------------------

BRIGHTNESS_THRESHOLD  = 180   # HSV V-channel mean > this → water-surface zone
BRIGHTNESS_PATCH_SIZE = 10    # half-patch radius in pixels

# ---------------------------------------------------------------------------
# Neck-based water-surface line parameters
# ---------------------------------------------------------------------------

NECK_BUFFER_PX  = 20    # fixed fallback buffer around neck (pixels)
NECK_BUFFER_REL = 0.05  # 5% of torso height — used when larger than fallback

# ---------------------------------------------------------------------------
# Body groups for self-occlusion cross-comparison
# ---------------------------------------------------------------------------

UPPER_BODY_LEFT  = ('LShoulder', 'LElbow', 'LWrist')
UPPER_BODY_RIGHT = ('RShoulder', 'RElbow', 'RWrist')
LOWER_BODY_LEFT  = ('LHip', 'LKnee', 'LAnkle')
LOWER_BODY_RIGHT = ('RHip', 'RKnee', 'RAnkle')

BODY_GROUPS = [
    (UPPER_BODY_LEFT, UPPER_BODY_RIGHT),
    (LOWER_BODY_LEFT, LOWER_BODY_RIGHT),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_view_from_path(path: str | Path) -> str | None:
    """
    Walk the path parts looking for a known camera-view folder name.

    Returns a normalized view name (e.g., 'water_level') or None if the path
    does not contain a recognized view directory.
    """
    for part in Path(path).parts:
        if part in KNOWN_VIEWS:
            return KNOWN_VIEWS[part]
    return None


def compute_visibility(
    kp_coords: dict[str, tuple[float, float]],
    image,
    camera_view: str | None,
    img_w: int,
    img_h: int,
) -> tuple[dict[str, float], dict[str, str]]:
    """
    Compute per-keypoint visibility value and reason.

    Args:
        kp_coords:   {name: (x_px, y_img_px)} in image space (Y-down).
        image:       BGR numpy array (cv2 format), or None to skip brightness check.
        camera_view: Normalized view name from detect_view_from_path(), or None.
        img_w:       Image width in pixels.
        img_h:       Image height in pixels.

    Returns:
        visibility: {name: 2.0 (visible) | 1.0 (out-of-bounds) | 0.0 (occluded)}
        reason:     {name: 'visible' | 'bounds' | 'self' | 'water'}

    Occlusion logic per view:
        water_level  — self-occlusion only (px 20/40)
        underwater   — self-occlusion + brightness OR neck (OR combination)
        above_water  — tight self-occlusion (px 5/8) + inverted neck
    """
    visibility: dict[str, float] = {}
    reason:     dict[str, str]   = {}

    # Step 1 — bounds check
    in_bounds: set[str] = set()
    for name, (x, y) in kp_coords.items():
        if (0.0 <= x <= img_w) and (0.0 <= y <= img_h):
            visibility[name] = 2.0
            reason[name]     = 'visible'
            in_bounds.add(name)
        else:
            visibility[name] = 1.0
            reason[name]     = 'bounds'

    if camera_view not in SIDE_VIEWS:
        return visibility, reason

    # Only operate on in-bounds keypoints for occlusion checks
    ib_coords = {n: kp_coords[n] for n in in_bounds}

    # Step 2 — self-occlusion (sets v=0, reason='self')
    if camera_view == 'above_water':
        self_thresh = SELF_OCC_PX_AW
        hip_thresh  = HIP_SELF_OCC_PX_AW
    else:
        self_thresh = SELF_OCC_PX
        hip_thresh  = HIP_SELF_OCC_PX

    for name in _classify_self_occlusion(ib_coords, self_thresh, hip_thresh):
        visibility[name] = 0.0
        reason[name]     = 'self'

    # Step 3 — water-surface occlusion (overrides self)
    if camera_view == 'underwater':
        neck_occ = _classify_neck_occlusion(ib_coords, inverted=False)
        for name in list(in_bounds):
            if name == 'Neck':
                continue
            x, y = kp_coords[name]
            bright = image is not None and _classify_brightness(image, x, y)
            if bright or (name in neck_occ):
                visibility[name] = 0.0
                reason[name]     = 'water'

    elif camera_view == 'above_water':
        for name in _classify_neck_occlusion(ib_coords, inverted=True):
            if name in in_bounds:
                visibility[name] = 0.0
                reason[name]     = 'water'

    return visibility, reason


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_swimming_direction(
    kp_coords: dict[str, tuple[float, float]],
) -> str | None:
    """
    Determine swimming direction from shoulder vs ankle X positions.

    Shoulders are at the leading end; ankles trail behind.
    Returns 'left' (swimming towards lower x), 'right', or None if
    insufficient keypoints are present.
    """
    s_xs = [kp_coords[n][0] for n in ('LShoulder', 'RShoulder') if n in kp_coords]
    a_xs = [kp_coords[n][0] for n in ('LAnkle', 'RAnkle')       if n in kp_coords]
    if not s_xs or not a_xs:
        return None
    return 'left' if (sum(s_xs) / len(s_xs)) < (sum(a_xs) / len(a_xs)) else 'right'


def _classify_self_occlusion(
    kp_coords: dict[str, tuple[float, float]],
    threshold: float,
    hip_threshold: float,
) -> set[str]:
    """
    Return the set of keypoints determined to be self-occluded.

    For each body region (upper / lower), every left-side keypoint is compared
    against every right-side keypoint in that region.  A far-side keypoint is
    marked occluded when ANY opposite-side keypoint in the same region is within
    `threshold` pixels of it (or `hip_threshold` for hip joints).

    Swimming direction determines which side is the far (occluded) side:
        swimming left  → right side faces away → right-side points may be occluded
        swimming right → left side faces away  → left-side points may be occluded
        unknown        → both sides may be occluded
    """
    # Pass 1: compute minimum cross-body distance per keypoint
    min_dist: dict[str, float] = {}
    for left_group, right_group in BODY_GROUPS:
        for l_name in left_group:
            if l_name not in kp_coords:
                continue
            lx, ly = kp_coords[l_name]
            for r_name in right_group:
                if r_name not in kp_coords:
                    continue
                rx, ry = kp_coords[r_name]
                d = float(np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2))
                if l_name not in min_dist or d < min_dist[l_name]:
                    min_dist[l_name] = d
                if r_name not in min_dist or d < min_dist[r_name]:
                    min_dist[r_name] = d

    # Pass 2: mark far-side keypoints within threshold as occluded
    swim_dir = _get_swimming_direction(kp_coords)
    occluded: set[str] = set()

    for left_group, right_group in BODY_GROUPS:
        # Right side occluded when swimming left (or direction unknown)
        if swim_dir in ('left', None):
            for name in right_group:
                if name not in min_dist:
                    continue
                t = hip_threshold if name in ('RHip', 'LHip') else threshold
                if min_dist[name] < t:
                    occluded.add(name)

        # Left side occluded when swimming right (or direction unknown)
        if swim_dir in ('right', None):
            for name in left_group:
                if name not in min_dist:
                    continue
                t = hip_threshold if name in ('LHip', 'RHip') else threshold
                if min_dist[name] < t:
                    occluded.add(name)

    return occluded


def _classify_brightness(
    image: np.ndarray,
    kp_x: float,
    kp_y_img: float,
) -> bool:
    """
    Return True if the patch around (kp_x, kp_y_img) has a high HSV V-channel
    mean, indicating a water-surface zone (bright reflection band).

    Used for Side_underwater where the water surface appears as a bright
    (often white/silvery) band across the top of the frame.
    """
    h, w = image.shape[:2]
    x = int(np.clip(kp_x,     0, w - 1))
    y = int(np.clip(kp_y_img, 0, h - 1))
    ps = BRIGHTNESS_PATCH_SIZE
    patch = image[max(0, y - ps):min(h, y + ps),
                  max(0, x - ps):min(w, x + ps)]
    if patch.size == 0:
        return False
    v_mean = float(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2].mean())
    return v_mean > BRIGHTNESS_THRESHOLD


def _classify_neck_occlusion(
    kp_coords: dict[str, tuple[float, float]],
    inverted: bool = False,
) -> set[str]:
    """
    Return keypoints water-occluded relative to the Neck position.

    inverted=False (Side_underwater, camera looks up):
        Occluded when y_img < neck_y - buffer
        → joint appears above neck in the image → above water surface

    inverted=True (Side_above_water, camera looks down):
        Occluded when y_img > neck_y + buffer
        → joint appears below neck in the image → submerged

    Buffer adapts to torso height: max(NECK_BUFFER_PX, 5% of torso height).
    Returns empty set if Neck keypoint is not present.
    """
    if 'Neck' not in kp_coords:
        return set()

    neck_y_img = kp_coords['Neck'][1]

    # Adaptive buffer based on torso height
    s_ys = [kp_coords[n][1] for n in ('LShoulder', 'RShoulder') if n in kp_coords]
    h_ys = [kp_coords[n][1] for n in ('LHip', 'RHip')           if n in kp_coords]
    if s_ys and h_ys:
        torso_h   = abs(sum(s_ys) / len(s_ys) - sum(h_ys) / len(h_ys))
        buffer_px = max(NECK_BUFFER_PX, int(NECK_BUFFER_REL * torso_h))
    else:
        buffer_px = NECK_BUFFER_PX

    occluded: set[str] = set()
    for name, (x, y_img) in kp_coords.items():
        if name == 'Neck':
            continue
        if inverted:
            if y_img > neck_y_img + buffer_px:
                occluded.add(name)
        else:
            if y_img < neck_y_img - buffer_px:
                occluded.add(name)

    return occluded
