import os
import cv2
import argparse
import shutil
import random
import subprocess
from pathlib import Path
from PIL import Image

from format_conversion import convert_frame

VIDEO_EXTENSION      = '.webm'
ANNOTATION_EXTENSION = '.txt'
FPS = 60


def find_matching_files(video_folder, annotation_folder, is_3d, annotation_type):
    """
    Find matching video and annotation files based on their relative paths.

    Args:
        video_folder (str): Root folder containing .webm files
        annotation_folder (str): Root folder containing .txt files
        is_3d (bool): Match 3D annotations (filename contains '3d') if True, 2D if False
        annotation_type (str): 'base', 'body25', or 'COCO'

    Returns:
        list: Tuples of (video_path, annotation_path) for matched pairs

    Logic:
        1. Find all .webm files in video folder
        2. Find all .txt files where parent folder matches annotation_type
        3. Filter annotations by 2d/3d in filename
        4. Match when video stem equals annotation's grandparent folder name
        5. Verify relative directory paths match up to video level
    """
    allowed_annotation_types = ['base', 'body25', 'COCO']
    if annotation_type == 'coco':
        annotation_type = 'COCO'
    if annotation_type not in allowed_annotation_types:
        raise ValueError(
            f"Invalid annotation type: {annotation_type}. "
            f"Allowed types: {allowed_annotation_types}"
        )

    try:
        video_folder      = Path(video_folder)
        annotation_folder = Path(annotation_folder)

        video_files           = list(video_folder.rglob(f'*{VIDEO_EXTENSION}'))
        all_annotation_files  = list(annotation_folder.rglob(f'*{ANNOTATION_EXTENSION}'))
        annotation_files      = []

        for annotation_file in all_annotation_files:
            if annotation_file.parent.name.lower() == annotation_type.lower():
                if is_3d:
                    if '3d' in annotation_file.name.lower():
                        annotation_files.append(annotation_file)
                else:
                    if '2d' in annotation_file.name.lower():
                        annotation_files.append(annotation_file)

    except Exception as e:
        print(f"Error occurred while finding matching files: {e}")
        return []

    matches = []
    for video_file in video_files:
        video_stem = video_file.stem
        for annotation_file in annotation_files:
            video_folder_in_annotation_path = annotation_file.parent.parent.name
            if video_folder_in_annotation_path == video_stem:
                video_rel_path        = video_file.relative_to(video_folder)
                annotation_base_path  = annotation_file.parent.parent.relative_to(annotation_folder)
                if video_rel_path.parent == annotation_base_path.parent:
                    matches.append((video_file, annotation_file))
                    break

    return matches


def create_cleaned_dataset(
    matches,
    output_folder,
    limit_of_videos=None,
    mode="override",
    min_visible_ratio: float = 0.5,
) -> dict:
    """
    Create cleaned dataset by extracting frames from videos and converting
    annotations to YOLO pose format with proper occlusion detection.

    Args:
        matches (list):            List of (video_path, annotation_path) tuples.
        output_folder (str/Path):  Path to output directory.
        limit_of_videos (int):     Maximum number of videos to process.
        mode (str):                'override' to clear directory first,
                                   'append' to add to existing (not yet implemented).
        min_visible_ratio (float): Min fraction of keypoints that must be in-bounds
                                   to keep a frame (default: 0.5).

    Returns:
        dict with keys:
            total_annotation_frames: total annotation rows processed
            frames_saved:            frames written to disk
            frames_skipped:          frames filtered out (below threshold or empty)
    """
    output_path = Path(output_folder)

    if mode == "override":
        if output_path.exists():
            print(f"Override mode: Clearing existing directory {output_path}")
            shutil.rmtree(output_path)
        output_path.mkdir(exist_ok=True)
    elif mode == "append":
        output_path.mkdir(exist_ok=True)
        print("TODO: Append mode not yet implemented, defaulting to override behavior")
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'override' or 'append'")

    images_output = output_path / 'images'
    labels_output = output_path / 'labels'
    images_output.mkdir(exist_ok=True)
    labels_output.mkdir(exist_ok=True)

    if limit_of_videos:
        matches = matches[:limit_of_videos]

    print(f"Creating cleaned dataset in: {output_path}")
    print(f"Processing {len(matches)} video pairs")

    # Lazy import to avoid hard dependency when occlusion not needed
    try:
        from occlusion import detect_view_from_path
    except ImportError:
        try:
            from data_processing.occlusion import detect_view_from_path
        except ImportError:
            detect_view_from_path = lambda p: None

    total_annotation_frames = 0
    total_frames_saved      = 0
    total_frames_skipped    = 0

    for img_num, (video_file, annotation_file) in enumerate(matches, 1):
        print(f"\nProcessing video {img_num}/{len(matches)}: {video_file.name}")

        # Detect camera view for occlusion
        camera_view = detect_view_from_path(video_file)
        if camera_view:
            print(f"  Camera view: {camera_view}")

        try:
            temp_frames_dir = output_path / f"temp_frames_{img_num}"
            temp_frames_dir.mkdir(exist_ok=True)

            ffmpeg_cmd = [
                'ffmpeg', '-i', str(video_file),
                '-vf', f'fps={FPS}',
                str(temp_frames_dir / 'frame_%04d.jpg'),
                '-y'
            ]
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  Error extracting frames: {result.stderr[:200]}")
                shutil.rmtree(temp_frames_dir, ignore_errors=True)
                continue

            frame_files = sorted(temp_frames_dir.glob('frame_*.jpg'))
            if not frame_files:
                print("  No frames extracted.")
                shutil.rmtree(temp_frames_dir, ignore_errors=True)
                continue

            # Get image dimensions from the first frame
            with Image.open(frame_files[0]) as img:
                img_width, img_height = img.size
            print(f"  Frame size: {img_width}x{img_height}")

            # Read and parse annotation file
            with open(annotation_file, 'r') as f:
                ann_lines = f.readlines()

            if len(ann_lines) < 2:
                print(f"  Warning: annotation file has insufficient data, skipping.")
                shutil.rmtree(temp_frames_dir, ignore_errors=True)
                continue

            header     = ann_lines[0].strip().rstrip(';').split(';')
            data_lines = ann_lines[1:]

            num_video_frames      = len(frame_files)
            num_annotation_frames = len(data_lines)

            if num_video_frames != num_annotation_frames:
                print(
                    f"  Warning: frame count mismatch — "
                    f"video={num_video_frames}, annotation={num_annotation_frames}"
                )
            num_frames_to_process = min(num_video_frames, num_annotation_frames)
            total_annotation_frames += num_frames_to_process

            frames_saved   = 0
            frames_skipped = 0

            for frame_idx in range(num_frames_to_process):
                source_frame = frame_files[frame_idx]

                # Parse annotation row
                ann_line = data_lines[frame_idx].strip().rstrip(';').replace(',', '.')
                try:
                    values = [float(v) for v in ann_line.split(';') if v.strip()]
                except ValueError:
                    frames_skipped += 1
                    continue

                # Load frame image for occlusion brightness check
                frame_image = None
                if camera_view is not None:
                    frame_image = cv2.imread(str(source_frame))

                annotation = convert_frame(
                    header, values, img_width, img_height,
                    frame_image=frame_image,
                    camera_view=camera_view,
                    min_visible_ratio=min_visible_ratio,
                )

                if annotation is None:
                    frames_skipped += 1
                    continue

                frame_num       = frame_idx + 1
                image_name      = f"{img_num:04d}_{frame_num:04d}.jpg"
                annotation_name = f"{img_num:04d}_{frame_num:04d}.txt"

                shutil.copy2(source_frame, images_output / image_name)
                with open(labels_output / annotation_name, 'w') as f:
                    f.write(annotation)
                frames_saved += 1

            total_frames_saved   += frames_saved
            total_frames_skipped += frames_skipped

            if frames_skipped > 0:
                print(
                    f"  Saved {frames_saved}, skipped {frames_skipped} "
                    f"({frames_skipped / num_frames_to_process:.0%}) frames"
                )
            else:
                print(f"  Saved all {frames_saved} frames")

            shutil.rmtree(temp_frames_dir, ignore_errors=True)

        except Exception as e:
            print(f"  Error processing {video_file.name}: {e}")
            if 'temp_frames_dir' in locals():
                shutil.rmtree(temp_frames_dir, ignore_errors=True)
            continue

    print(f"\nDataset creation complete!")
    print(f"Total videos processed:    {len(matches)}")
    print(f"Total annotation frames:   {total_annotation_frames}")
    print(f"Frames saved:              {total_frames_saved}")
    print(f"Frames skipped/filtered:   {total_frames_skipped}")
    if total_annotation_frames:
        kept_pct = total_frames_saved / total_annotation_frames
        print(f"Keep rate:                 {kept_pct:.1%}")
    print(f"Images → {images_output}")
    print(f"Labels → {labels_output}")

    return {
        'total_annotation_frames': total_annotation_frames,
        'frames_saved':   total_frames_saved,
        'frames_skipped': total_frames_skipped,
    }


def split_dataset(dataset_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split the cleaned dataset into train, validation, and test sets.

    Args:
        dataset_folder (str/Path): Path to the cleaned dataset folder
                                   containing 'images' and 'labels' subfolders
        train_ratio (float): Proportion of data for training (default: 0.8)
        val_ratio (float): Proportion of data for validation (default: 0.1)
        test_ratio (float): Proportion of data for testing (default: 0.1)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0. Current sum: {train_ratio + val_ratio + test_ratio}"
        )

    dataset_path   = Path(dataset_folder)
    images_folder  = dataset_path / 'images'
    labels_folder  = dataset_path / 'labels'

    if not images_folder.exists() or not labels_folder.exists():
        print(f"Error: Expected 'images' and 'labels' subfolders in {dataset_path}")
        return

    image_files = sorted(list(images_folder.glob('*.jpg')))
    label_files = sorted(list(labels_folder.glob('*.txt')))

    if len(image_files) != len(label_files):
        print(
            f"Warning: Mismatch between images ({len(image_files)}) "
            f"and labels ({len(label_files)})"
        )
        return

    if len(image_files) == 0:
        print("Error: No image files found to split")
        return

    print(f"Splitting {len(image_files)} samples into train/val/test...")

    total_samples = len(image_files)
    indices       = list(range(total_samples))
    random.seed(42)
    random.shuffle(indices)

    train_end = int(train_ratio * total_samples)
    val_end   = train_end + int(val_ratio * total_samples)

    splits = {
        'train': indices[:train_end],
        'val':   indices[train_end:val_end],
        'test':  indices[val_end:],
    }

    for split_name, split_indices in splits.items():
        split_images_dir = dataset_path / 'images' / split_name
        split_labels_dir = dataset_path / 'labels' / split_name
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        for idx in split_indices:
            image_file = image_files[idx]
            label_file = label_files[idx]
            shutil.copy2(image_file, split_images_dir / image_file.name)
            shutil.copy2(label_file, split_labels_dir / label_file.name)
            os.remove(image_file)
            os.remove(label_file)

    print(f"\nDataset split complete!")
    print(f"Train: {len(splits['train'])} samples ({len(splits['train'])/total_samples:.1%})")
    print(f"Val:   {len(splits['val'])} samples ({len(splits['val'])/total_samples:.1%})")
    print(f"Test:  {len(splits['test'])} samples ({len(splits['test'])/total_samples:.1%})")
    print(f"\nSplit directories created:")
    for split_name in splits:
        print(f"  {dataset_path / 'images' / split_name}")
        print(f"  {dataset_path / 'labels' / split_name}")


def main():
    parser = argparse.ArgumentParser(
        description='Create cleaned dataset by matching videos with annotations'
    )
    parser.add_argument('video_folder',      help='Path to video dataset folder')
    parser.add_argument('annotation_folder', help='Path to annotations folder')
    parser.add_argument('--is_3d', action='store_true',
                        help='Use 3D annotations (default: False for 2D)')
    parser.add_argument('--annotation_type', choices=['base', 'body25', 'coco'],
                        default='body25', help='Annotation type to use (default: body25)')
    parser.add_argument('--test_matches', action='store_true',
                        help='Test mode: only find matches without creating dataset')
    parser.add_argument('--limit_videos', type=int,
                        help='Maximum number of videos to process')
    parser.add_argument('--mode', choices=['override', 'append'], default='override',
                        help='Dataset creation mode (default: override)')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio',   type=float, default=0.1)
    parser.add_argument('--test_ratio',  type=float, default=0.1)
    parser.add_argument('--min_visible_ratio', type=float, default=0.5,
                        help='Min fraction of keypoints in-bounds to keep frame (default: 0.5)')

    args = parser.parse_args()

    if not os.path.exists(args.video_folder):
        print(f"Error: Video folder '{args.video_folder}' does not exist")
        return
    if not os.path.exists(args.annotation_folder):
        print(f"Error: Annotation folder '{args.annotation_folder}' does not exist")
        return

    print(f"Video folder:       {args.video_folder}")
    print(f"Annotation folder:  {args.annotation_folder}")
    print(f"3D annotations:     {args.is_3d}")
    print(f"Annotation type:    {args.annotation_type}")
    print(f"Min visible ratio:  {args.min_visible_ratio}")
    print("-" * 50)

    matches = find_matching_files(
        args.video_folder, args.annotation_folder,
        args.is_3d, args.annotation_type
    )

    if not matches:
        print("No matching video-annotation pairs found!")
        print("Please check:")
        print("1. Video and annotation folder paths are correct")
        print("2. Annotation type folder exists")
        print("3. 2D/3D files exist with correct naming")
        return

    print(f"Found {len(matches)} matching pairs:")

    if args.test_matches:
        print("Test matches mode complete. Use without --test_matches to create dataset.")
        return

    video_folder_name = Path(args.video_folder).name
    output_folder     = Path.cwd() / f"cleaned_{video_folder_name}"
    print(f"Output folder: {output_folder}")
    print("-" * 50)

    create_cleaned_dataset(
        matches, output_folder, args.limit_videos, args.mode, args.min_visible_ratio
    )

    split_dataset(output_folder, args.train_ratio, args.val_ratio, args.test_ratio)


if __name__ == "__main__":
    main()
