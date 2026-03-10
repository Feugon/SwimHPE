"""
import_batch.py — Convert X-AnyLabeling JSON annotations to YOLO .txt and copy
images into a dataset split (e.g. dataset/mixed/images/train).

Reads imageWidth/imageHeight from each JSON so mixed-resolution batches are
handled correctly.

Usage:
    python data_processing/import_batch.py <batch_dir> \
        --images-out dataset/mixed/images/train \
        --labels-out dataset/mixed/labels/train
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from format_conversion import (
    XANY_TO_COCO17_IDX,
    KEYPOINT_VISIBLE,
    KEYPOINT_NOT_VISIBLE,
    calculate_bounding_box,
)

_ROOT = _HERE.parent


def convert_json(json_path: Path) -> str | None:
    """
    Convert a single X-AnyLabeling JSON to a YOLO pose label line.
    Returns the label string, or None if the frame has no annotated keypoints.
    """
    with open(json_path) as f:
        data = json.load(f)

    img_w = data.get("imageWidth",  640)
    img_h = data.get("imageHeight", 360)

    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    bbox_pixels = None

    for shape in data.get("shapes", []):
        label      = shape.get("label", "").lower()
        shape_type = shape.get("shape_type", "")
        points     = shape.get("points", [])
        difficult  = shape.get("difficult", False)

        if shape_type == "rectangle" and label == "person" and len(points) == 4:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            bbox_pixels = (min(xs), min(ys), max(xs), max(ys))

        elif shape_type == "point" and label in XANY_TO_COCO17_IDX and points:
            idx      = XANY_TO_COCO17_IDX[label]
            x, y     = points[0]
            vis      = KEYPOINT_NOT_VISIBLE if difficult else KEYPOINT_VISIBLE
            kps[idx] = [round(x / img_w, 6), round(y / img_h, 6), vis]

    if all(k[2] == 0.0 for k in kps):
        return None

    if bbox_pixels is not None:
        x1, y1, x2, y2 = bbox_pixels
        x_c = round(((x1 + x2) / 2) / img_w, 6)
        y_c = round(((y1 + y2) / 2) / img_h, 6)
        w   = round((x2 - x1) / img_w, 6)
        h   = round((y2 - y1) / img_h, 6)
    else:
        kp_coords = {
            str(i): {"x": k[0], "y": k[1], "v": k[2]}
            for i, k in enumerate(kps) if k[2] > 0.0
        }
        bbox = calculate_bounding_box(kp_coords, padding=20, img_width=img_w, img_height=img_h)
        if bbox is None:
            return None
        x_c, y_c, w, h = bbox

    kp_str = " ".join(f"{k[0]:.6f} {k[1]:.6f} {k[2]:.6f}" for k in kps)
    return f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {kp_str}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import X-AnyLabeling batch annotations into a YOLO dataset split."
    )
    parser.add_argument("batch_dir", help="Directory with *.json + *.jpg files")
    parser.add_argument("--images-out", default="dataset/mixed/images/train",
                        help="Destination for image files")
    parser.add_argument("--labels-out", default="dataset/mixed/labels/train",
                        help="Destination for YOLO .txt label files")
    args = parser.parse_args()

    batch_dir  = Path(args.batch_dir)
    images_out = _ROOT / args.images_out
    labels_out = _ROOT / args.labels_out
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    json_files = sorted(batch_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {batch_dir}")
        sys.exit(1)

    converted = skipped_no_kp = missing_img = 0

    for json_path in json_files:
        label_line = convert_json(json_path)
        if label_line is None:
            skipped_no_kp += 1
            continue

        jpg_src = json_path.with_suffix(".jpg")
        if not jpg_src.exists():
            print(f"  WARNING: image not found: {jpg_src.name}")
            missing_img += 1
            continue

        stem = json_path.stem
        (labels_out / f"{stem}.txt").write_text(label_line + "\n")
        shutil.copy2(jpg_src, images_out / jpg_src.name)
        converted += 1

    print(f"\nDone.")
    print(f"  Converted : {converted}")
    print(f"  Skipped   : {skipped_no_kp} (no keypoints annotated)")
    print(f"  Missing   : {missing_img} (jpg not found)")
    print(f"  Images → {images_out}")
    print(f"  Labels → {labels_out}")


if __name__ == "__main__":
    main()
