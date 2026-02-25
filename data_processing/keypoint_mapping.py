"""
Canonical keypoint mapping for SwimXYZ body25 → 13-body-keypoint → YOLO conversion.

Face keypoints (Nose, LEye, REye, LEar, REar) are excluded — they are not visible
or useful in swimming footage.  The model uses 13 body keypoints:
  LShoulder(0), RShoulder(1), LElbow(2), RElbow(3),
  LWrist(4), RWrist(5), LHip(6), RHip(7),
  LKnee(8), RKnee(9), LAnkle(10), RAnkle(11), Neck(12)

## body25 vs COCO Annotation Files

SwimXYZ ships two annotation formats under each clip:
  body25/2D_cam.txt  — 25 keypoints × 3 values = 75 columns, ALL correct labels
  COCO/2D_cam.txt    — same 75-column header but only 54 data values; contains a
                        7-column cyclic shift in lower-body headers that was verified
                        by visual inspection in keypoint_label_debug.ipynb

**This pipeline uses body25 files.** Body25 column names match their data exactly —
no correction is needed. The shift documented in keypoint_label_debug.ipynb applies
only to COCO/2D_cam.txt files and is NOT relevant here.

Body25 index 8, "MidHip", is the genuine pelvis midpoint (not RHip data). It is not
part of our 13-KP model and is simply omitted from the mapping below.

`Neck` (body25 index 1) is present and included as target keypoint index 12.
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
# Face-related columns and MidHip (pelvis midpoint) are omitted so
# SWIMXYZ_COL_TO_YOLO_IDX (derived below) silently ignores them.
SWIMXYZ_TO_COCO_NAME = {
    # Upper body — column names match their actual data (same in body25 and COCO)
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
    # Neck — body25 index 1; column name matches data
    'Neck':         'Neck',
    # Lower body — body25 column names match their actual data (no shift)
    'RHip':         'RHip',
    'LHip':         'LHip',
    'RKnee':        'RKnee',
    'LKnee':        'LKnee',
    'RAnkle':       'RAnkle',
    'LAnkle':       'LAnkle',
    # MidHip (body25 pelvis midpoint) — omitted; not in our 13-KP model
    # Face keypoints — omitted; not in our 13-KP model
}

# Derived convenience dict: raw SwimXYZ column name → YOLO slot index.
# Used directly by format_conversion.py's convert_to_yolo().
SWIMXYZ_COL_TO_YOLO_IDX = {
    col: COCO_KP_INDEX[coco_name]
    for col, coco_name in SWIMXYZ_TO_COCO_NAME.items()
}
