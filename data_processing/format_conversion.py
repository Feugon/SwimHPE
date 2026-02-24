import os
import json
import random
import argparse
from pathlib import Path
from ultralytics.utils.downloads import download
import shutil

from data_processing.keypoint_mapping import SWIMXYZ_COL_TO_YOLO_IDX

KEYPOINT_VISIBLE = 2.0
KEYPOINT_NOT_VISIBLE = 1.0

def calculate_bounding_box(keypoint_coords, padding=80, img_width=1920, img_height=1080):
    """
    Calculate bounding box from visible keypoint coordinates only.
    """
    if not keypoint_coords:
        return 
    
    x_coords = []
    y_coords = []
    
    for _, coords in keypoint_coords.items():
        if 'x' in coords and 'y' in coords and 'v' in coords:
            # Only use visible keypoints for bounding box calculation
            if coords['v'] == KEYPOINT_VISIBLE:
                x_coords.append(coords['x'])
                y_coords.append(coords['y'])
    
    if not x_coords or not y_coords:
        return 
    
    # Convert padding from pixels to normalized coordinates (assuming 1920x1080 as reference)
    padding_norm_x = padding / img_width  
    padding_norm_y = padding / img_height  

    x_min = max(0, min(x_coords) - padding_norm_x)
    x_max = min(1, max(x_coords) + padding_norm_x)
    y_min = max(0, min(y_coords) - padding_norm_y)
    y_max = min(1, max(y_coords) + padding_norm_y)

    width = x_max - x_min
    height = y_max - y_min
    
    x_center = x_min + width / 2
    y_center = y_min + height / 2

    return round(x_center, 6), round(y_center, 6), round(width, 6), round(height, 6)


def convert_to_yolo(coco_annotation_file, img_width=1920, img_height=1080):
    """
    Convert COCO format annotation file to YOLO pose format with normalized coordinates.
    
    Args:
        coco_annotation_file (str): Path to the COCO annotation file
        img_width (int): Width of the image for normalization (default: 1920)
        img_height (int): Height of the image for normalization (default: 1080)
    """
    # Column name → YOLO slot index mapping imported from keypoint_mapping.py.
    # See that module for the full discovery story and shift details.
    swimxyz_col_to_yolo = SWIMXYZ_COL_TO_YOLO_IDX

    with open(coco_annotation_file, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        print(f"Warning: File {coco_annotation_file} has insufficient data")
        print(lines)
        return

    header = lines[0].strip().rstrip(';').split(';')

    converted_lines = []

    for line_idx, line in enumerate(lines[1:], 1):
        try:
            line = line.strip().rstrip(';').replace(',', '.')
            values = [float(v) for v in line.split(';') if v.strip()]
            n_vals = len(values)

            yolo_keypoints = ['0.0'] * 51  # 17 keypoints × 3 = 51
            has_visible_keypoints = False
            normalized_keypoint_coords = {}

            for i, col_name in enumerate(header):
                if not col_name.endswith('.x'):
                    continue
                if i + 1 >= n_vals:
                    break  # data ends before this column's y value
                kp_name = col_name[:-2]  # strip '.x'
                if kp_name not in swimxyz_col_to_yolo:
                    continue

                yolo_idx = swimxyz_col_to_yolo[kp_name]
                x = values[i]
                y = values[i + 1]
                flipped_y = img_height - y

                normalized_x = round(x / img_width, 6)
                normalized_y = round(flipped_y / img_height, 6)

                if (0 <= x <= img_width) and (0 <= flipped_y <= img_height):
                    visibility = KEYPOINT_VISIBLE
                    has_visible_keypoints = True
                else:
                    visibility = KEYPOINT_NOT_VISIBLE

                normalized_keypoint_coords[kp_name] = {
                    'x': normalized_x, 'y': normalized_y, 'v': visibility
                }
                yolo_keypoints[yolo_idx * 3]     = str(normalized_x)
                yolo_keypoints[yolo_idx * 3 + 1] = str(normalized_y)
                yolo_keypoints[yolo_idx * 3 + 2] = f"{visibility:.6f}"

            if has_visible_keypoints:
                bbox = calculate_bounding_box(normalized_keypoint_coords)
                if bbox:
                    x_center, y_center, width, height = bbox
                    bbox_line = f"0 {x_center} {y_center} {width} {height} {' '.join(yolo_keypoints)}"
                    converted_lines.append(bbox_line)
                else:
                    converted_lines.append("")
            else:
                converted_lines.append("")

        except Exception as e:
            print(f"Error processing line {line_idx}: {e}")
            continue

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
    # COCO to YOLO keypoint mapping (same as swim dataset)
    coco_to_yolo_mapping = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16
    }
    
    # Standard COCO keypoint names in order
    coco_keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    with open(coco_json_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directory if it doesn't exist
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
        image_id = image_info['id']
        image_filename = image_info['file_name']
        original_width = image_info['width']
        original_height = image_info['height']
        
        # Create corresponding label filename
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(output_dir, label_filename)
        
        converted_lines = []
        
        # Process annotations for this image
        if image_id in image_annotations:
            for ann in image_annotations[image_id]:
                if 'keypoints' in ann and len(ann['keypoints']) >= 51:  # 17 keypoints * 3 = 51
                    keypoints = ann['keypoints']
                    
                    yolo_keypoints = ['0.0'] * 51  # 17 keypoints * 3 coordinates (x, y, v) = 51
                    has_visible_keypoints = False
                    normalized_keypoint_coords = {}
                    
                    # Process each keypoint
                    for i, keypoint_name in enumerate(coco_keypoint_names):
                        x = keypoints[i * 3]
                        y = keypoints[i * 3 + 1]
                        v = keypoints[i * 3 + 2]  # COCO visibility: 0=not labeled, 1=labeled but not visible, 2=labeled and visible
                        
                        if v > 0:  # If keypoint is labeled
                            # Scale coordinates from original image size to target size
                            scaled_x = (x / original_width) * img_width
                            scaled_y = (y / original_height) * img_height
                            
                            # Flip y-coordinate to match swim dataset format
                            flipped_y = img_height - scaled_y
                            
                            # Normalize coordinates to [0,1] range
                            normalized_x = round(scaled_x / img_width, 6)
                            normalized_y = round(flipped_y / img_height, 6)
                            
                            # Convert COCO visibility to swim dataset format
                            if v == 2 and (0 <= scaled_x <= img_width) and (0 <= flipped_y <= img_height):
                                visibility = KEYPOINT_VISIBLE
                                has_visible_keypoints = True
                            else:
                                visibility = KEYPOINT_NOT_VISIBLE
                            
                            # Get YOLO index for this keypoint
                            yolo_idx = coco_to_yolo_mapping[keypoint_name]
                            
                            # Store normalized coordinates for bounding box calculation
                            normalized_keypoint_coords[keypoint_name] = {
                                'x': normalized_x,
                                'y': normalized_y,
                                'v': visibility
                            }
                            
                            # Set YOLO keypoint values
                            yolo_keypoints[yolo_idx * 3] = str(normalized_x)
                            yolo_keypoints[yolo_idx * 3 + 1] = str(normalized_y)
                            yolo_keypoints[yolo_idx * 3 + 2] = f"{visibility:.6f}"
                    
                    # Only create annotation if there are visible keypoints
                    if has_visible_keypoints:
                        bbox = calculate_bounding_box(normalized_keypoint_coords, img_width=img_width, img_height=img_height)
                        if bbox:
                            x_center, y_center, width, height = bbox
                            # Format: class_id x_center y_center width height keypoints...
                            bbox_line = f"0 {x_center} {y_center} {width} {height} {' '.join(yolo_keypoints)}"
                            converted_lines.append(bbox_line)
        
        # Write annotations to file (even if empty)
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
    val_zip_url = "http://images.cocodataset.org/zips/val2017.zip"  # 1G, 5k images
    labels_zip_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-pose.zip"

    dataset_dir = Path(dataset_path)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    # Ensure directories exist
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Download validation images
    if not (images_dir / "val2017").exists():
        print("Downloading COCO-Pose validation images...")
        download([val_zip_url], dir=images_dir, threads=3)

    # Download labels
    if not labels_dir.exists():
        print("Downloading COCO-Pose labels...")
        download([labels_zip_url], dir=dataset_dir.parent)

def combine_datasets(swim_images_dir, swim_labels_dir, filtered_coco_images, filtered_coco_labels, output_images_dir, output_labels_dir, coco_percentage):
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
    # Convert output directories to Path objects
    output_images_dir = Path(output_images_dir)
    output_labels_dir = Path(output_labels_dir)

    # Ensure output directories exist
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Get all swim images and labels
    swim_images = list(Path(swim_images_dir).glob("*.jpg"))
    swim_labels = list(Path(swim_labels_dir).glob("*.txt"))

    # Ensure swim datasets have matching image-label pairs
    if len(swim_images) != len(swim_labels):
        raise ValueError("Mismatch between swim images and labels")

    # Calculate the number of COCO images to include
    num_coco_images = int(len(swim_images) * coco_percentage)

    # Ensure num_coco_images does not exceed the number of available filtered COCO images
    num_coco_images = min(num_coco_images, len(filtered_coco_images))

    # Randomly sample COCO images and labels
    sampled_coco_indices = random.sample(range(len(filtered_coco_images)), num_coco_images)
    sampled_coco_images = [filtered_coco_images[i] for i in sampled_coco_indices]
    sampled_coco_labels = [filtered_coco_labels[i] for i in sampled_coco_indices]

    # Copy swim images and labels to the output directory
    for swim_image, swim_label in zip(swim_images, swim_labels):
        shutil.copy(swim_image, output_images_dir / swim_image.name)
        shutil.copy(swim_label, output_labels_dir / swim_label.name)

    # Copy sampled COCO images and labels to the output directory
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

    # Create a set of label filenames (without extensions)
    label_filenames = {label.stem for label in coco_labels}

    # Filter images to keep only those with matching labels
    filtered_images = [image for image in coco_images if image.stem in label_filenames]
    filtered_labels = [label for label in coco_labels if label.stem in label_filenames]

    return filtered_images, filtered_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine swim and COCO datasets for training.")
    parser.add_argument("--swim_images_dir", type=str, required=True, help="Path to the swim images directory.")
    parser.add_argument("--swim_labels_dir", type=str, required=True, help="Path to the swim labels directory.")
    parser.add_argument("--coco_images_dir", type=str, required=True, help="Path to the COCO images directory.")
    parser.add_argument("--coco_labels_dir", type=str, required=True, help="Path to the COCO labels directory.")
    parser.add_argument("--output_images_dir", type=str, required=True, help="Path to save the combined images directory.")
    parser.add_argument("--output_labels_dir", type=str, required=True, help="Path to save the combined labels directory.")
    parser.add_argument("--coco_percentage", type=float, required=True, help="Percentage of COCO images to include (0.0 to 1.0).")

    args = parser.parse_args()

    # Validate coco_percentage
    if not (0.0 <= args.coco_percentage <= 1.0):
        raise ValueError("coco_percentage must be between 0.0 and 1.0")

    # Filter COCO images and labels
    filtered_coco_images, filtered_coco_labels = filter_coco_images_and_labels(args.coco_images_dir, args.coco_labels_dir)

    # Combine datasets
    combine_datasets(
        args.swim_images_dir,
        args.swim_labels_dir,
        filtered_coco_images,
        filtered_coco_labels,
        args.output_images_dir,
        args.output_labels_dir,
        args.coco_percentage
    )