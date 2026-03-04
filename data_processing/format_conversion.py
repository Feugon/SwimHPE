import os
import json
import random
import argparse
from pathlib import Path
from ultralytics.utils.downloads import download
import shutil

try:
    from data_processing.keypoint_mapping import (
        SWIMXYZ_COL_TO_YOLO_IDX,
        SWIMXYZ_TO_COCO_NAME,
        COCO_KP_NAMES,
        COCO_KP_INDEX,
    )
except ModuleNotFoundError:
    from keypoint_mapping import (
        SWIMXYZ_COL_TO_YOLO_IDX,
        SWIMXYZ_TO_COCO_NAME,
        COCO_KP_NAMES,
        COCO_KP_INDEX,
    )

# Visibility flag values used in YOLO pose labels
KEYPOINT_VISIBLE     = 2.0   # in-bounds and not occluded
KEYPOINT_NOT_VISIBLE = 1.0   # outside image bounds
KEYPOINT_OCCLUDED    = 0.0   # in-bounds but hidden (water or self-occlusion)

# X-AnyLabeling label → COCO17 keypoint index
XANY_TO_COCO17_IDX = {
    'nose': 0,
    'left_eye': 1,      'right_eye': 2,
    'left_ear': 3,      'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7,    'right_elbow': 8,
    'left_wrist': 9,    'right_wrist': 10,
    'left_hip': 11,     'right_hip': 12,
    'left_knee': 13,    'right_knee': 14,
    'left_ankle': 15,   'right_ankle': 16,
}


def calculate_bounding_box(keypoint_coords, padding=80, img_width=1920, img_height=1080):
    """
    Calculate bounding box from in-bounds keypoint coordinates.

    Includes both visible (v=2) and occluded-but-in-bounds (v=0) keypoints so
    the bounding box covers the full physical extent of the body in the frame.
    Out-of-bounds keypoints (v=1) are excluded.
    """
    if not keypoint_coords:
        return

    x_coords = []
    y_coords = []

    for _, coords in keypoint_coords.items():
        if 'x' in coords and 'y' in coords and 'v' in coords:
            if coords['v'] in (KEYPOINT_VISIBLE, KEYPOINT_OCCLUDED):
                x_coords.append(coords['x'])
                y_coords.append(coords['y'])

    if not x_coords or not y_coords:
        return

    padding_norm_x = padding / img_width
    padding_norm_y = padding / img_height

    x_min = max(0, min(x_coords) - padding_norm_x)
    x_max = min(1, max(x_coords) + padding_norm_x)
    y_min = max(0, min(y_coords) - padding_norm_y)
    y_max = min(1, max(y_coords) + padding_norm_y)

    width    = x_max - x_min
    height   = y_max - y_min
    x_center = x_min + width  / 2
    y_center = y_min + height / 2

    return round(x_center, 6), round(y_center, 6), round(width, 6), round(height, 6)


def convert_frame(
    header: list[str],
    row_values: list[float],
    img_width: int = 1920,
    img_height: int = 1080,
    frame_image=None,
    camera_view: str | None = None,
    min_visible_ratio: float = 0.5,
) -> str | None:
    """
    Convert a single annotation row to a YOLO pose annotation string.

    Args:
        header:            Column names from the annotation file header.
        row_values:        Parsed float values for one frame row.
        img_width:         Image width in pixels.
        img_height:        Image height in pixels.
        frame_image:       BGR numpy array (cv2) for occlusion brightness check,
                           or None to skip brightness-based water detection.
        camera_view:       Normalized view name from occlusion.detect_view_from_path(),
                           e.g. 'water_level', 'underwater', 'above_water', or None
                           for bounds-only visibility.
        min_visible_ratio: Minimum fraction of body keypoints that must be inside the
                           image bounds to keep the frame.  Set to 0.5 to require
                           at least half of the in-frame body joints.

    Returns:
        YOLO annotation string, or None if the frame should be skipped.
    """
    n_vals = len(row_values)

    # Build {true_name: (x_px, y_img_px)} from the raw row
    # True names (LShoulder, LHip, etc.) are obtained via SWIMXYZ_TO_COCO_NAME,
    # which corrects the lower-body column shift in SwimXYZ files.
    true_name_to_pixel: dict[str, tuple[float, float]] = {}

    for i, col_name in enumerate(header):
        if not col_name.endswith('.x'):
            continue
        if i + 1 >= n_vals:
            break
        raw_name  = col_name[:-2]          # strip '.x'
        true_name = SWIMXYZ_TO_COCO_NAME.get(raw_name)
        if true_name is None:
            continue                        # face keypoints and unknowns silently ignored
        x     = row_values[i]
        y_img = img_height - row_values[i + 1]   # flip from Y-up annotation space
        true_name_to_pixel[true_name] = (x, y_img)

    if not true_name_to_pixel:
        return None

    # --- Min-visible filter ---
    # Count keypoints whose coordinates are within image bounds.
    n_total    = len(true_name_to_pixel)
    n_in_bounds = sum(
        1 for (x, y) in true_name_to_pixel.values()
        if (0.0 <= x <= img_width) and (0.0 <= y <= img_height)
    )
    if n_total == 0 or (n_in_bounds / n_total) < min_visible_ratio:
        return None

    # --- Compute visibility flags ---
    if camera_view is not None:
        try:
            from data_processing.occlusion import compute_visibility
        except ImportError:
            from occlusion import compute_visibility
        vis_map, _ = compute_visibility(
            true_name_to_pixel, frame_image, camera_view, img_width, img_height
        )
    else:
        # Bounds-only visibility (no occlusion detection)
        vis_map: dict[str, float] = {}
        for name, (x, y) in true_name_to_pixel.items():
            if (0.0 <= x <= img_width) and (0.0 <= y <= img_height):
                vis_map[name] = KEYPOINT_VISIBLE
            else:
                vis_map[name] = KEYPOINT_NOT_VISIBLE

    # --- Build YOLO keypoints array and normalized coordinates dict ---
    n_kp = len(COCO_KP_NAMES)
    yolo_keypoints = ['0.0'] * (n_kp * 3)

    # Face keypoints (slots 0-4: Nose, LEye, REye, LEar, REar) are absent from
    # SwimXYZ body25 annotations.  Use v=0 (not labeled) so the trainer ignores
    # them in OKS loss entirely — v=1 at (0,0) would corrupt training.
    for i in range(5):
        yolo_keypoints[i * 3 + 2] = f"{KEYPOINT_OCCLUDED:.6f}"

    normalized_keypoint_coords: dict[str, dict] = {}
    has_in_bounds = False

    for true_name, (x, y_img) in true_name_to_pixel.items():
        yolo_idx     = COCO_KP_INDEX[true_name]
        normalized_x = round(max(0.0, min(1.0, x     / img_width)),  6)
        normalized_y = round(max(0.0, min(1.0, y_img / img_height)), 6)
        visibility   = vis_map.get(true_name, KEYPOINT_NOT_VISIBLE)

        normalized_keypoint_coords[true_name] = {
            'x': normalized_x, 'y': normalized_y, 'v': visibility
        }
        yolo_keypoints[yolo_idx * 3]     = str(normalized_x)
        yolo_keypoints[yolo_idx * 3 + 1] = str(normalized_y)
        yolo_keypoints[yolo_idx * 3 + 2] = f"{visibility:.6f}"

        if visibility in (KEYPOINT_VISIBLE, KEYPOINT_OCCLUDED):
            has_in_bounds = True

    if not has_in_bounds:
        return None

    bbox = calculate_bounding_box(
        normalized_keypoint_coords, img_width=img_width, img_height=img_height
    )
    if bbox is None:
        return None

    x_center, y_center, width, height = bbox
    return f"0 {x_center} {y_center} {width} {height} {' '.join(yolo_keypoints)}"


def convert_to_yolo(
    coco_annotation_file,
    img_width: int = 1920,
    img_height: int = 1080,
    frame_dir=None,
    camera_view: str | None = None,
    min_visible_ratio: float = 0.5,
) -> list[str]:
    """
    Convert a SwimXYZ annotation file to a list of YOLO pose annotation strings.

    Each element corresponds to one frame.  Frames that are filtered out
    (too few in-bounds joints) are represented as empty strings so that the
    list length matches the number of annotation rows.

    Args:
        coco_annotation_file: Path to the semicolon-delimited annotation file.
        img_width:            Image width for normalization (default: 1920).
        img_height:           Image height for normalization (default: 1080).
        frame_dir:            Optional directory containing extracted frame JPEGs
                              named frame_NNNN.jpg.  If provided, each frame image
                              is loaded and passed to convert_frame() for
                              brightness-based water occlusion detection.
        camera_view:          Normalized view name for occlusion detection, or None
                              to use bounds-only visibility.
        min_visible_ratio:    Minimum fraction of keypoints that must be in-bounds
                              to keep the frame (default: 0.5).

    Returns:
        List of YOLO annotation strings (empty string = frame filtered out).
    """
    with open(coco_annotation_file, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        print(f"Warning: File {coco_annotation_file} has insufficient data")
        return []

    header = lines[0].strip().rstrip(';').split(';')
    converted_lines: list[str] = []

    for line_idx, line in enumerate(lines[1:], 1):
        try:
            line   = line.strip().rstrip(';').replace(',', '.')
            values = [float(v) for v in line.split(';') if v.strip()]

            frame_image = None
            if frame_dir is not None:
                import cv2 as _cv2
                img_path = Path(frame_dir) / f'frame_{line_idx:04d}.jpg'
                if img_path.exists():
                    frame_image = _cv2.imread(str(img_path))

            result = convert_frame(
                header, values, img_width, img_height,
                frame_image=frame_image,
                camera_view=camera_view,
                min_visible_ratio=min_visible_ratio,
            )
            converted_lines.append(result if result is not None else "")

        except Exception as e:
            print(f"Error processing line {line_idx}: {e}")
            converted_lines.append("")

    return converted_lines


def convert_coco_json_to_swim_format(coco_json_file, output_dir, img_width=1920, img_height=1080):
    """
    Convert standard COCO JSON annotations to swim dataset YOLO format.

    Args:
        coco_json_file (str): Path to the COCO JSON annotation file
        output_dir (str): Directory to save the converted YOLO format files
        img_width (int): Target image width for normalization (default: 1920)
        img_height (int): Target image height for normalization (default: 1080)
    """
    # COCO body-only keypoint mapping (face KPs excluded, body re-indexed 0–11)
    coco_to_yolo_mapping = {
        'left_shoulder': 0,
        'right_shoulder': 1,
        'left_elbow': 2,
        'right_elbow': 3,
        'left_wrist': 4,
        'right_wrist': 5,
        'left_hip': 6,
        'right_hip': 7,
        'left_knee': 8,
        'right_knee': 9,
        'left_ankle': 10,
        'right_ankle': 11,
    }

    # All 17 standard COCO keypoint names in raw-array order (face names kept so
    # index math into ann['keypoints'] stays correct; face KPs are skipped via the
    # mapping dict guard below).
    coco_keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    with open(coco_json_file, 'r') as f:
        coco_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Group annotations by image
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)

    # Process each image
    for image_info in coco_data['images']:
        image_id        = image_info['id']
        image_filename  = image_info['file_name']
        original_width  = image_info['width']
        original_height = image_info['height']

        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path     = os.path.join(output_dir, label_filename)

        converted_lines = []

        if image_id in image_annotations:
            for ann in image_annotations[image_id]:
                if 'keypoints' in ann and len(ann['keypoints']) >= 51:
                    keypoints = ann['keypoints']

                    yolo_keypoints = ['0.0'] * 39  # 13 body keypoints * 3
                    has_visible_keypoints       = False
                    normalized_keypoint_coords  = {}

                    for i, keypoint_name in enumerate(coco_keypoint_names):
                        if keypoint_name not in coco_to_yolo_mapping:
                            continue
                        x = keypoints[i * 3]
                        y = keypoints[i * 3 + 1]
                        v = keypoints[i * 3 + 2]

                        if v > 0:
                            scaled_x  = (x / original_width)  * img_width
                            scaled_y  = (y / original_height) * img_height
                            flipped_y = img_height - scaled_y

                            normalized_x = round(scaled_x  / img_width,  6)
                            normalized_y = round(flipped_y / img_height, 6)

                            if v == 2 and (0 <= scaled_x <= img_width) and (0 <= flipped_y <= img_height):
                                visibility = KEYPOINT_VISIBLE
                                has_visible_keypoints = True
                            else:
                                visibility = KEYPOINT_NOT_VISIBLE

                            yolo_idx = coco_to_yolo_mapping[keypoint_name]

                            normalized_keypoint_coords[keypoint_name] = {
                                'x': normalized_x, 'y': normalized_y, 'v': visibility
                            }

                            yolo_keypoints[yolo_idx * 3]     = str(normalized_x)
                            yolo_keypoints[yolo_idx * 3 + 1] = str(normalized_y)
                            yolo_keypoints[yolo_idx * 3 + 2] = f"{visibility:.6f}"

                    if has_visible_keypoints:
                        bbox = calculate_bounding_box(
                            normalized_keypoint_coords, img_width=img_width, img_height=img_height
                        )
                        if bbox:
                            x_center, y_center, width, height = bbox
                            bbox_line = (
                                f"0 {x_center} {y_center} {width} {height} "
                                f"{' '.join(yolo_keypoints)}"
                            )
                            converted_lines.append(bbox_line)

        with open(label_path, 'w') as f:
            for line in converted_lines:
                f.write(line + '\n')

    print(f"Converted {len(coco_data['images'])} images to swim dataset format in {output_dir}")


def download_coco_pose_val(dataset_path):
    """
    Download the COCO-Pose validation dataset if not already present.

    Args:
        dataset_path (str): Path to the COCO-Pose dataset directory.
    """
    val_zip_url    = "http://images.cocodataset.org/zips/val2017.zip"
    labels_zip_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-pose.zip"

    dataset_dir = Path(dataset_path)
    images_dir  = dataset_dir / "images"
    labels_dir  = dataset_dir / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    if not (images_dir / "val2017").exists():
        print("Downloading COCO-Pose validation images...")
        download([val_zip_url], dir=images_dir, threads=3)

    if not labels_dir.exists():
        print("Downloading COCO-Pose labels...")
        download([labels_zip_url], dir=dataset_dir.parent)


def combine_datasets(
    swim_images_dir, swim_labels_dir,
    filtered_coco_images, filtered_coco_labels,
    output_images_dir, output_labels_dir,
    coco_percentage,
):
    """
    Combine the swim dataset with a specified percentage of COCO images.

    Args:
        swim_images_dir (str): Path to the swim images directory.
        swim_labels_dir (str): Path to the swim labels directory.
        filtered_coco_images (list): List of filtered COCO image paths.
        filtered_coco_labels (list): List of filtered COCO label paths.
        output_images_dir (str): Path to save the combined images directory.
        output_labels_dir (str): Path to save the combined labels directory.
        coco_percentage (float): Percentage of COCO images to include (0.0 to 1.0).
    """
    output_images_dir = Path(output_images_dir)
    output_labels_dir = Path(output_labels_dir)

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    swim_images = list(Path(swim_images_dir).glob("*.jpg"))
    swim_labels = list(Path(swim_labels_dir).glob("*.txt"))

    if len(swim_images) != len(swim_labels):
        raise ValueError("Mismatch between swim images and labels")

    num_coco_images = int(len(swim_images) * coco_percentage)
    num_coco_images = min(num_coco_images, len(filtered_coco_images))

    sampled_coco_indices = random.sample(range(len(filtered_coco_images)), num_coco_images)
    sampled_coco_images  = [filtered_coco_images[i] for i in sampled_coco_indices]
    sampled_coco_labels  = [filtered_coco_labels[i]  for i in sampled_coco_indices]

    for swim_image, swim_label in zip(swim_images, swim_labels):
        shutil.copy(swim_image, output_images_dir / swim_image.name)
        shutil.copy(swim_label, output_labels_dir / swim_label.name)

    for coco_image, coco_label in zip(sampled_coco_images, sampled_coco_labels):
        shutil.copy(coco_image, output_images_dir / coco_image.name)
        shutil.copy(coco_label, output_labels_dir / coco_label.name)

    print(f"Combined dataset saved to {output_images_dir} and {output_labels_dir}")
    print(f"Swim images: {len(swim_images)}, COCO images: {len(sampled_coco_images)}")


def filter_coco_images_and_labels(coco_images_dir, coco_labels_dir):
    """
    Filter COCO images and labels, keeping only those with matching filenames.

    Args:
        coco_images_dir (str): Path to the COCO images directory.
        coco_labels_dir (str): Path to the COCO labels directory.

    Returns:
        list: Filtered list of image paths.
        list: Filtered list of label paths.
    """
    coco_images = list(Path(coco_images_dir).glob("*.jpg"))
    coco_labels = list(Path(coco_labels_dir).glob("*.txt"))

    label_filenames = {label.stem for label in coco_labels}

    filtered_images = [image for image in coco_images if image.stem in label_filenames]
    filtered_labels = [label for label in coco_labels if label.stem in label_filenames]

    return filtered_images, filtered_labels


def convert_xanylabeling_to_yolo(json_dir, output_dir, img_width=640, img_height=360):
    """
    Convert X-AnyLabeling JSON annotations to YOLO pose format (COCO17, 17 keypoints).

    Reads per-frame JSON files exported from X-AnyLabeling v4.x and writes matching
    YOLO .txt label files.  One output file per input JSON, same stem.

    Visibility mapping:
        annotated + difficult=False → 2.0 (visible)
        annotated + difficult=True  → 1.0 (labeled but uncertain)
        not annotated               → 0.0 (missing)

    Bounding box: uses the annotated 'person' rectangle when present; otherwise
    computes it from visible keypoints via calculate_bounding_box().

    Args:
        json_dir:   Directory containing frame_NNNN.json (+ frame_NNNN.jpg) files.
        output_dir: Directory to write YOLO .txt label files.
        img_width:  Image width in pixels (default: 640).
        img_height: Image height in pixels (default: 360).
    """
    json_dir   = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return

    written = skipped = 0
    for json_path in json_files:
        with open(json_path) as f:
            data = json.load(f)

        # Initialize 17 slots as [x_norm, y_norm, vis]
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
                idx     = XANY_TO_COCO17_IDX[label]
                x, y    = points[0]
                vis     = KEYPOINT_NOT_VISIBLE if difficult else KEYPOINT_VISIBLE
                kps[idx] = [round(x / img_width, 6), round(y / img_height, 6), vis]

        # Skip frames with no annotated keypoints
        if all(k[2] == 0.0 for k in kps):
            skipped += 1
            continue

        # Bounding box: use annotated rectangle when available
        if bbox_pixels is not None:
            x1, y1, x2, y2 = bbox_pixels
            x_center = round(((x1 + x2) / 2) / img_width,  6)
            y_center = round(((y1 + y2) / 2) / img_height, 6)
            width    = round((x2 - x1)        / img_width,  6)
            height   = round((y2 - y1)        / img_height, 6)
        else:
            kp_coords = {
                str(i): {'x': k[0], 'y': k[1], 'v': k[2]}
                for i, k in enumerate(kps) if k[2] > 0.0
            }
            bbox = calculate_bounding_box(
                kp_coords, padding=20, img_width=img_width, img_height=img_height
            )
            if bbox is None:
                skipped += 1
                continue
            x_center, y_center, width, height = bbox

        kp_str = " ".join(f"{k[0]:.6f} {k[1]:.6f} {k[2]:.6f}" for k in kps)
        line   = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {kp_str}"
        (output_dir / (json_path.stem + ".txt")).write_text(line + "\n")
        written += 1

    print(f"Converted {written} frames, skipped {skipped} (no keypoints)  →  {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine swim and COCO datasets for training.")
    parser.add_argument("--swim_images_dir", type=str, required=True)
    parser.add_argument("--swim_labels_dir", type=str, required=True)
    parser.add_argument("--coco_images_dir", type=str, required=True)
    parser.add_argument("--coco_labels_dir", type=str, required=True)
    parser.add_argument("--output_images_dir", type=str, required=True)
    parser.add_argument("--output_labels_dir", type=str, required=True)
    parser.add_argument("--coco_percentage", type=float, required=True)

    args = parser.parse_args()

    if not (0.0 <= args.coco_percentage <= 1.0):
        raise ValueError("coco_percentage must be between 0.0 and 1.0")

    filtered_coco_images, filtered_coco_labels = filter_coco_images_and_labels(
        args.coco_images_dir, args.coco_labels_dir
    )

    combine_datasets(
        args.swim_images_dir,
        args.swim_labels_dir,
        filtered_coco_images,
        filtered_coco_labels,
        args.output_images_dir,
        args.output_labels_dir,
        args.coco_percentage,
    )
