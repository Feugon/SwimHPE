"""
Canonical keypoint mapping for SwimXYZ → COCO-17 → YOLO conversion.

## SwimXYZ Column Shift

Raw SwimXYZ annotation files contain a 7-column cyclic shift in the lower-body
joint headers: the column *names* do not match the data actually stored in those
columns.  Additionally, the `LAnkle` column holds LEar (a face keypoint).

Discovery: `keypoint_label_debug.ipynb` plotted raw column names as labeled dots on
actual video frames, revealing that "LAnkle" landed on the ear and "MidHip" landed
at the right hip.  A three-panel comparison (RAW vs two correction hypotheses) showed
that only the 7-column shift produced anatomically correct labels.

Verification: `check_knee_ankle_order()` in `keypoint_label_debug.ipynb` confirmed
0 swapped frames across all three side-view cameras after applying this correction.

Column shift summary:
  LAnkle col → LEar (face)
  RAnkle col → LHip
  MidHip col → RHip   ← "MidHip" is not a real joint; the column holds RHip data
  LHip   col → LKnee
  RHip   col → RKnee
  LKnee  col → LAnkle
  RKnee  col → RAnkle

`Neck` appears in the raw file but is not a COCO-17 keypoint and is correctly ignored.
"""

# COCO-17 keypoint names in canonical YOLO slot order (index = slot)
COCO_KP_NAMES = [
    'Nose',       # 0
    'LEye',       # 1
    'REye',       # 2
    'LEar',       # 3
    'REar',       # 4
    'LShoulder',  # 5
    'RShoulder',  # 6
    'LElbow',     # 7
    'RElbow',     # 8
    'LWrist',     # 9
    'RWrist',     # 10
    'LHip',       # 11
    'RHip',       # 12
    'LKnee',      # 13
    'RKnee',      # 14
    'LAnkle',     # 15
    'RAnkle',     # 16
]

# COCO keypoint name → YOLO slot index
COCO_KP_INDEX = {name: i for i, name in enumerate(COCO_KP_NAMES)}

# SwimXYZ raw column name → true COCO-17 keypoint name.
# Includes alternative spellings found across different video batches.
SWIMXYZ_TO_COCO_NAME = {
    # Upper body and face — column names match their actual data
    'Nose':         'Nose',
    'LEye':         'LEye',
    'Eye_L':        'LEye',
    'REye':         'REye',
    'Eye_R':        'REye',
    'REar':         'REar',
    'Ear_R':        'REar',
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
    # Lower-body cyclic shift — column names do NOT match their actual data
    'LAnkle':       'LEar',    # LAnkle column holds LEar coordinates (face area)
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
