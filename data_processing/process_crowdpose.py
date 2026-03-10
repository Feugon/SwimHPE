"""
process_crowdpose.py — Append CrowdPose real-world images to the SwimHPE dataset.

Converts CrowdPose 14-keypoint annotations to our 13-keypoint YOLO format, applies
optional 90°/270° rotation augmentation to ~33% each, and distributes images across
the existing train/val/test splits to double their sizes.

Usage (smoke test — 30 images):
    python data_processing/process_crowdpose.py \
      --images_dir /path/to/crowdpose/images \
      --json_files /path/to/crowdpose_val.json \
      --dataset_dir dataset \
      --n_images 30

Usage (full run — doubles the dataset):
    python data_processing/process_crowdpose.py \
      --images_dir /path/to/crowdpose/images \
      --json_files /path/to/crowdpose_train.json /path/to/crowdpose_val.json \
      --dataset_dir dataset
"""

import os
import sys
import json
import random
import shutil
import argparse
from pathlib import Path

import cv2

# Allow running from project root or from data_processing/ directly
try:
    from data_processing.format_conversion import calculate_bounding_box
    from data_processing.keypoint_mapping import COCO_KP_NAMES
except ModuleNotFoundError:
    from format_conversion import calculate_bounding_box
    from keypoint_mapping import COCO_KP_NAMES

# ---------------------------------------------------------------------------
# CrowdPose → our 13-KP mapping
#
# CrowdPose keypoints (14 total, indices 0–13):
#   0  left_shoulder   → our slot 0  (LShoulder)
#   1  right_shoulder  → our slot 1  (RShoulder)
#   2  left_elbow      → our slot 2  (LElbow)
#   3  right_elbow     → our slot 3  (RElbow)
#   4  left_wrist      → our slot 4  (LWrist)
#   5  right_wrist     → our slot 5  (RWrist)
#   6  left_hip        → our slot 6  (LHip)
#   7  right_hip       → our slot 7  (RHip)
#   8  left_knee       → our slot 8  (LKnee)
#   9  right_knee      → our slot 9  (RKnee)
#   10 left_ankle      → our slot 10 (LAnkle)
#   11 right_ankle     → our slot 11 (RAnkle)
#   12 head            → SKIP (not in our 13-KP model)
#   13 neck            → our slot 12 (Neck)
#
# Slots 0–11 are direct, slot 12 is CrowdPose index 13 (neck).
# ---------------------------------------------------------------------------
CROWDPOSE_TO_OUR_SLOT = {
    0: 0,   # left_shoulder  → LShoulder
    1: 1,   # right_shoulder → RShoulder
    2: 2,   # left_elbow     → LElbow
    3: 3,   # right_elbow    → RElbow
    4: 4,   # left_wrist     → LWrist
    5: 5,   # right_wrist    → RWrist
    6: 6,   # left_hip       → LHip
    7: 7,   # right_hip      → RHip
    8: 8,   # left_knee      → LKnee
    9: 9,   # right_knee     → RKnee
    10: 10, # left_ankle     → LAnkle
    11: 11, # right_ankle    → RAnkle
    # 12: head → skipped
    13: 12, # neck           → Neck
}

KEYPOINT_VISIBLE     = 2.0
KEYPOINT_NOT_VISIBLE = 1.0
KEYPOINT_OCCLUDED    = 0.0

NUM_OUR_KPS = len(COCO_KP_NAMES)  # 13


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_images(split_dir: Path) -> int:
    if not split_dir.exists():
        return 0
    return sum(1 for f in split_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'})


def _load_json_files(json_files: list[str]) -> tuple[list, dict]:
    """Load and merge multiple CrowdPose JSON files.

    Returns:
        images:  list of image dicts (merged, deduplicated by id)
        annots:  dict mapping image_id → list of annotation dicts
    """
    images_by_id: dict[int, dict] = {}
    annots_by_image: dict[int, list] = {}

    for path in json_files:
        print(f"  Loading {path} …")
        with open(path, 'r') as f:
            data = json.load(f)
        for img in data.get('images', []):
            images_by_id[img['id']] = img
        for ann in data.get('annotations', []):
            annots_by_image.setdefault(ann['image_id'], []).append(ann)

    return list(images_by_id.values()), annots_by_image


def _map_keypoints(kps_raw: list, img_w: int, img_h: int) -> list[tuple[float, float, float]]:
    """Convert raw CrowdPose keypoints to our 13-slot normalized format.

    Args:
        kps_raw: flat list of 42 values [x, y, v] × 14 in pixel coords
        img_w, img_h: original image dimensions

    Returns:
        list of 13 (x_norm, y_norm, visibility) tuples in our keypoint order
    """
    result = [(0.0, 0.0, KEYPOINT_NOT_VISIBLE)] * NUM_OUR_KPS

    for cp_idx, our_slot in CROWDPOSE_TO_OUR_SLOT.items():
        base = cp_idx * 3
        px = kps_raw[base]
        py = kps_raw[base + 1]
        v  = kps_raw[base + 2]

        if v == 0:
            # Not annotated — mark as out of bounds
            result[our_slot] = (0.0, 0.0, KEYPOINT_NOT_VISIBLE)
        elif v == 1:
            # Labeled but occluded — position is known, visibility is 0 (occluded)
            x_n = max(0.0, min(1.0, px / img_w))
            y_n = max(0.0, min(1.0, py / img_h))
            result[our_slot] = (x_n, y_n, KEYPOINT_OCCLUDED)
        else:
            # v == 2: labeled and fully visible
            x_n = px / img_w
            y_n = py / img_h
            in_bounds = (0.0 <= x_n <= 1.0) and (0.0 <= y_n <= 1.0)
            x_n = max(0.0, min(1.0, x_n))
            y_n = max(0.0, min(1.0, y_n))
            result[our_slot] = (x_n, y_n, KEYPOINT_VISIBLE if in_bounds else KEYPOINT_NOT_VISIBLE)

    return result


def _rotate_keypoints_90cw(kps: list[tuple]) -> list[tuple]:
    """Apply 90° CW rotation to normalized keypoints.

    Transform: (x_n, y_n) → (1 - y_n, x_n)
    New image: W_new = H_old, H_new = W_old  (handled by cv2.rotate on the image).
    """
    return [(1.0 - y, x, v) for x, y, v in kps]


def _rotate_keypoints_270cw(kps: list[tuple]) -> list[tuple]:
    """Apply 270° CW (= 90° CCW) rotation to normalized keypoints.

    Transform: (x_n, y_n) → (y_n, 1 - x_n)
    """
    return [(y, 1.0 - x, v) for x, y, v in kps]


def _build_kp_dict(kps: list[tuple]) -> dict:
    """Convert list of (x, y, v) tuples to the dict format expected by calculate_bounding_box."""
    return {
        name: {'x': x, 'y': y, 'v': v}
        for name, (x, y, v) in zip(COCO_KP_NAMES, kps)
    }


def _format_yolo_line(bbox: tuple, kps: list[tuple]) -> str:
    """Format a single YOLO pose annotation line."""
    cx, cy, w, h = bbox
    kp_str = ' '.join(f'{x:.6f} {y:.6f} {v:.6f}' for x, y, v in kps)
    return f'0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kp_str}'


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_crowdpose(
    images_dir: str,
    json_files: list[str],
    dataset_dir: str,
    n_images: int | None = None,
    seed: int = 42,
    min_keypoints: int = 10,
):
    """Convert and append CrowdPose images to the existing dataset.

    Args:
        images_dir:    Directory containing CrowdPose .jpg images.
        json_files:    List of CrowdPose JSON annotation file paths.
        dataset_dir:   Root of the existing YOLO dataset (contains images/ and labels/).
        n_images:      Total images to add. Defaults to len(existing dataset).
        seed:          Random seed for reproducibility.
        min_keypoints: Skip a person annotation with fewer than this many annotated KPs.
    """
    dataset_path = Path(dataset_dir)
    images_root  = dataset_path / 'images'
    labels_root  = dataset_path / 'labels'

    # Count existing images per split
    split_counts = {
        'train': _count_images(images_root / 'train'),
        'val':   _count_images(images_root / 'val'),
        'test':  _count_images(images_root / 'test'),
    }
    total_existing = sum(split_counts.values())
    print(f"\nExisting dataset: {split_counts} (total {total_existing})")

    if total_existing == 0:
        raise RuntimeError(f"No existing images found under {images_root}. "
                           "Create the dataset first with prep_data.py.")

    n_images = n_images if n_images is not None else total_existing
    print(f"Target: add {n_images} CrowdPose images (seed={seed})\n")

    # Compute per-split targets so that each split doubles (or scales proportionally)
    scale = n_images / total_existing
    split_targets = {
        split: round(count * scale)
        for split, count in split_counts.items()
    }
    # Correct rounding to hit exact n_images
    diff = n_images - sum(split_targets.values())
    split_targets['train'] += diff  # absorb rounding into train
    print(f"Split targets: {split_targets}")

    # Load annotations
    print("\nLoading JSON annotations …")
    all_images, annots_by_image = _load_json_files(json_files)
    print(f"  {len(all_images)} unique images across JSON files")
    print(f"  {sum(len(v) for v in annots_by_image.values())} annotations total")

    # Filter to images that exist on disk
    images_dir_path = Path(images_dir)
    available = [
        img for img in all_images
        if (images_dir_path / img['file_name']).exists()
    ]
    print(f"  {len(available)} images found on disk")

    if len(available) < n_images:
        print(f"  Warning: only {len(available)} images available, adjusting n_images.")
        n_images = len(available)
        scale = n_images / total_existing
        split_targets = {s: round(c * scale) for s, c in split_counts.items()}
        diff = n_images - sum(split_targets.values())
        split_targets['train'] += diff
        print(f"  Adjusted targets: {split_targets}")

    # Select the least crowded images (fewest annotations per image = fewer people)
    available.sort(key=lambda img: len(annots_by_image.get(img['id'], [])))
    selected = available[:n_images]
    print(f"  Person-count range of selected images: "
          f"{len(annots_by_image.get(selected[0]['id'], []))}–"
          f"{len(annots_by_image.get(selected[-1]['id'], []))} persons/image")

    # Shuffle selected with seed before rotation/split assignment (preserves reproducibility)
    rng = random.Random(seed)
    rng.shuffle(selected)

    group_size = n_images // 3
    rotations = (
        [None]                        * group_size       +   # 0° (no rotation)
        ['90cw']                      * group_size       +   # 90° CW
        ['270cw']                     * (n_images - 2 * group_size)  # 270° CW (remainder)
    )
    # selected[0:group_size]          → no rotation
    # selected[group_size:2*group_size] → 90° CW
    # selected[2*group_size:]         → 270° CW

    # Assign to splits (sequential — first train_target, then val_target, then test)
    split_assign: list[str] = (
        ['train'] * split_targets['train'] +
        ['val']   * split_targets['val']   +
        ['test']  * split_targets['test']
    )

    # Ensure output directories exist
    for split in ('train', 'val', 'test'):
        (images_root / split).mkdir(parents=True, exist_ok=True)
        (labels_root / split).mkdir(parents=True, exist_ok=True)

    # Process images
    stats = {'saved': 0, 'skipped_no_persons': 0, 'skipped_missing': 0,
             'rotation_0': 0, 'rotation_90cw': 0, 'rotation_270cw': 0}

    print(f"\nProcessing {n_images} images …")
    for i, (img_meta, rotation, split) in enumerate(zip(selected, rotations, split_assign)):
        img_path = images_dir_path / img_meta['file_name']
        if not img_path.exists():
            stats['skipped_missing'] += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            stats['skipped_missing'] += 1
            continue

        h_orig, w_orig = img.shape[:2]

        # Apply rotation to image
        if rotation == '90cw':
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            stats['rotation_90cw'] += 1
        elif rotation == '270cw':
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            stats['rotation_270cw'] += 1
        else:
            stats['rotation_0'] += 1

        h_new, w_new = img.shape[:2]

        # Process each person annotation
        anns = annots_by_image.get(img_meta['id'], [])
        yolo_lines = []

        for ann in anns:
            if ann.get('iscrowd', 0):
                continue

            kps_raw = ann['keypoints']  # 42 values [x, y, v] × 14

            # Count fully visible keypoints (v == 2) before rotation/mapping
            n_visible = sum(1 for j in range(14) if kps_raw[j * 3 + 2] == 2)
            if n_visible < min_keypoints:
                continue

            # Map to our 13-KP normalized format
            kps = _map_keypoints(kps_raw, w_orig, h_orig)

            # Apply rotation transform to keypoints (normalized coords)
            if rotation == '90cw':
                kps = _rotate_keypoints_90cw(kps)
            elif rotation == '270cw':
                kps = _rotate_keypoints_270cw(kps)

            # Build bbox from in-bounds keypoints (reuses existing pipeline logic)
            kp_dict = _build_kp_dict(kps)
            bbox = calculate_bounding_box(kp_dict, padding=80,
                                          img_width=w_new, img_height=h_new)
            if bbox is None:
                continue

            yolo_lines.append(_format_yolo_line(bbox, kps))

        if not yolo_lines:
            stats['skipped_no_persons'] += 1
            continue

        # Save image and label
        stem = f"cp_{img_meta['id']}"
        out_img   = images_root / split / f"{stem}.jpg"
        out_label = labels_root / split / f"{stem}.txt"

        cv2.imwrite(str(out_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(out_label, 'w') as f:
            f.write('\n'.join(yolo_lines) + '\n')

        stats['saved'] += 1

        if (i + 1) % 100 == 0 or (i + 1) == n_images:
            print(f"  [{i + 1}/{n_images}] saved={stats['saved']} "
                  f"skip_persons={stats['skipped_no_persons']} "
                  f"skip_missing={stats['skipped_missing']}")

    # Summary
    print("\n" + "=" * 50)
    print("Done!")
    print(f"  Images saved:          {stats['saved']}")
    print(f"  Skipped (no persons):  {stats['skipped_no_persons']}")
    print(f"  Skipped (missing):     {stats['skipped_missing']}")
    print(f"  Rotation 0°:           {stats['rotation_0']}")
    print(f"  Rotation 90° CW:       {stats['rotation_90cw']}")
    print(f"  Rotation 270° CW:      {stats['rotation_270cw']}")

    new_totals = {
        split: _count_images(images_root / split)
        for split in ('train', 'val', 'test')
    }
    print(f"\n  New dataset totals: {new_totals} (total {sum(new_totals.values())})")
    print("=" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Append CrowdPose images to the SwimHPE YOLO dataset."
    )
    parser.add_argument(
        '--images_dir', required=True,
        help='Directory containing CrowdPose .jpg images.'
    )
    parser.add_argument(
        '--json_files', nargs='+', required=True,
        help='One or more CrowdPose JSON annotation files.'
    )
    parser.add_argument(
        '--dataset_dir', default='dataset',
        help='Root of the existing YOLO dataset (default: dataset).'
    )
    parser.add_argument(
        '--n_images', type=int, default=None,
        help='Number of images to add. Default: auto-count existing dataset size.'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42).'
    )
    parser.add_argument(
        '--min_keypoints', type=int, default=10,
        help='Minimum fully visible (v=2) keypoints to keep a person annotation (default: 10).'
    )
    args = parser.parse_args()

    process_crowdpose(
        images_dir=args.images_dir,
        json_files=args.json_files,
        dataset_dir=args.dataset_dir,
        n_images=args.n_images,
        seed=args.seed,
        min_keypoints=args.min_keypoints,
    )


if __name__ == '__main__':
    main()
