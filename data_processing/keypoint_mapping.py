"""
Canonical keypoint mapping for SwimXYZ body25 → COCO17 → YOLO conversion.

The model uses 17 standard COCO keypoints:
  Nose(0), LEye(1), REye(2), LEar(3), REar(4),
  LShoulder(5), RShoulder(6), LElbow(7), RElbow(8),
  LWrist(9), RWrist(10), LHip(11), RHip(12),
  LKnee(13), RKnee(14), LAnkle(15), RAnkle(16)

Face keypoints (Nose, LEye, REye, LEar, REar) are absent from SwimXYZ body25
annotations and are filled in as 0.0 0.0 1.0 (labeled-not-visible) by
format_conversion.py's convert_frame().

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
part of our COCO17 model and is simply omitted from the mapping below.

`Neck` (body25 index 1) is also omitted — it is not a standard COCO17 keypoint.
"""

# 17 COCO keypoint names in canonical YOLO slot order (index = slot)
COCO_KP_NAMES = [
    'Nose',        # 0  — face; absent in SwimXYZ → filled as 0.0 0.0 1.0
    'LEye',        # 1
    'REye',        # 2
    'LEar',        # 3
    'REar',        # 4
    'LShoulder',   # 5
    'RShoulder',   # 6
    'LElbow',      # 7
    'RElbow',      # 8
    'LWrist',      # 9
    'RWrist',      # 10
    'LHip',        # 11
    'RHip',        # 12
    'LKnee',       # 13
    'RKnee',       # 14
    'LAnkle',      # 15
    'RAnkle',      # 16
]

# Keypoint name → YOLO slot index
COCO_KP_INDEX = {name: i for i, name in enumerate(COCO_KP_NAMES)}

# SwimXYZ raw column name → true COCO17 keypoint name.
# Includes alternative spellings found across different video batches.
# Face-related columns, MidHip (pelvis midpoint), and Neck are omitted so
# SWIMXYZ_COL_TO_YOLO_IDX (derived below) silently ignores them.
SWIMXYZ_TO_COCO_NAME = {
    # Upper body — column names match their actual data (body25 files)
    'LShoulder':    'LShoulder',
    'L Clavicle':   'LShoulder',   # alternative name in some batches
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
    # Lower body — body25 column names match their actual data (no shift)
    'RHip':         'RHip',
    'LHip':         'LHip',
    'RKnee':        'RKnee',
    'LKnee':        'LKnee',
    'RAnkle':       'RAnkle',
    'LAnkle':       'LAnkle',
    # Neck (body25 index 1) — omitted; not a COCO17 keypoint
    # MidHip (body25 index 8) — omitted; pelvis midpoint, not in COCO17
    # Face keypoints — omitted; not available in SwimXYZ body25
}

# Derived convenience dict: raw SwimXYZ column name → YOLO slot index.
# Used directly by format_conversion.py's convert_frame().
SWIMXYZ_COL_TO_YOLO_IDX = {
    col: COCO_KP_INDEX[coco_name]
    for col, coco_name in SWIMXYZ_TO_COCO_NAME.items()
}
