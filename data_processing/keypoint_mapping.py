"""
Canonical keypoint mapping for SwimXYZ → 13-body-keypoint → YOLO conversion.

Face keypoints (Nose, LEye, REye, LEar, REar) are excluded — they are not visible
or useful in swimming footage.  The model uses 13 body keypoints:
  LShoulder(0), RShoulder(1), LElbow(2), RElbow(3),
  LWrist(4), RWrist(5), LHip(6), RHip(7),
  LKnee(8), RKnee(9), LAnkle(10), RAnkle(11), Neck(12)

## SwimXYZ Column Shift

Raw SwimXYZ annotation files contain a 7-column cyclic shift in the lower-body
joint headers: the column *names* do not match the data actually stored in those
columns.  Additionally, the `LAnkle` column holds LEar (a face keypoint, now ignored).

Discovery: `keypoint_label_debug.ipynb` plotted raw column names as labeled dots on
actual video frames, revealing that "LAnkle" landed on the ear and "MidHip" landed
at the right hip.  A three-panel comparison (RAW vs two correction hypotheses) showed
that only the 7-column shift produced anatomically correct labels.

Verification: `check_knee_ankle_order()` in `keypoint_label_debug.ipynb` confirmed
0 swapped frames across all three side-view cameras after applying this correction.

Column shift summary:
  LAnkle col → LEar (face — ignored)
  RAnkle col → LHip
  MidHip col → RHip   ← "MidHip" is not a real joint; the column holds RHip data
  LHip   col → LKnee
  RHip   col → RKnee
  LKnee  col → LAnkle
  RKnee  col → RAnkle

`Neck` appears in the raw file and is included as a target keypoint (index 12).
"""

# 13 body keypoint names in canonical YOLO slot order (index = slot)
COCO_KP_NAMES = [
    'LShoulder',  # 0
    'RShoulder',  # 1
    'LElbow',     # 2
    'RElbow',     # 3
    'LWrist',     # 4
    'RWrist',     # 5
    'LHip',       # 6
    'RHip',       # 7
    'LKnee',      # 8
    'RKnee',      # 9
    'LAnkle',     # 10
    'RAnkle',     # 11
    'Neck',       # 12
]

# Keypoint name → YOLO slot index
COCO_KP_INDEX = {name: i for i, name in enumerate(COCO_KP_NAMES)}

# SwimXYZ raw column name → true body keypoint name.
# Includes alternative spellings found across different video batches.
# Face-related columns (Nose, LEye, REye, REar, and the shifted LAnkle→LEar) are
# omitted so SWIMXYZ_COL_TO_YOLO_IDX (derived below) silently ignores them.
SWIMXYZ_TO_COCO_NAME = {
    # Upper body — column names match their actual data
    'LShoulder':    'LShoulder',
    'L Clavicle':   'LShoulder',
    'RShoulder':    'RShoulder',
    'R Clavicle':   'RShoulder',
    'LElbow':       'LElbow',
    'L Forearm':    'LElbow',
    'RElbow':       'RElbow',
    'R Forearm':    'RElbow',
    'LWrist':       'LWrist',
    'L Hand':       'LWrist',
    'RWrist':       'RWrist',
    'R Hand':       'RWrist',
    # Neck — column name matches actual data
    'Neck':         'Neck',
    # Lower-body cyclic shift — column names do NOT match their actual data
    # LAnkle col holds LEar (face) — excluded
    'RAnkle':       'LHip',    # RAnkle column holds LHip coordinates
    'MidHip':       'RHip',    # MidHip column holds RHip coordinates
    'LHip':         'LKnee',   # LHip   column holds LKnee coordinates
    'RHip':         'RKnee',   # RHip   column holds RKnee coordinates
    'LKnee':        'LAnkle',  # LKnee  column holds LAnkle coordinates
    'RKnee':        'RAnkle',  # RKnee  column holds RAnkle coordinates
}

# Derived convenience dict: raw SwimXYZ column name → YOLO slot index.
# Used directly by format_conversion.py's convert_to_yolo().
SWIMXYZ_COL_TO_YOLO_IDX = {
    col: COCO_KP_INDEX[coco_name]
    for col, coco_name in SWIMXYZ_TO_COCO_NAME.items()
}
