import os
from format_conversion import convert_to_yolo
import argparse
import shutil
from pathlib import Path
import subprocess
import random
from PIL import Image

VIDEO_EXTENSION = '.webm'
ANNOTATION_EXTENSION = '.txt'
FPS = 60

# TODO: Add pelvis/cam argument
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
        raise ValueError(f"Invalid annotation type: {annotation_type}. Allowed types: {allowed_annotation_types}")
   

    try:
        video_folder = Path(video_folder)
        annotation_folder = Path(annotation_folder)

        video_files = list(video_folder.rglob(f'*{VIDEO_EXTENSION}'))
        
        all_annotation_files = list(annotation_folder.rglob(f'*{ANNOTATION_EXTENSION}'))
        annotation_files = []
        
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
                video_rel_path = video_file.relative_to(video_folder)
                annotation_base_path = annotation_file.parent.parent.relative_to(annotation_folder)
                
                if video_rel_path.parent == annotation_base_path.parent:
                    matches.append((video_file, annotation_file))
                    break
    
    return matches


def create_cleaned_dataset(matches, output_folder, limit_of_videos=None, mode="override"):
    """
    Create cleaned dataset by extracting frames from videos and splitting annotations.
    
    Args:
        matches (list): List of (video_path, annotation_path) tuples
        output_folder (str/Path): Path to output directory
        limit_of_videos (int, optional): Maximum number of videos to process
        mode (str): "override" to clear directory first, "append" to add to existing
    """
    output_path = Path(output_folder)
    
    if mode == "override":
        if output_path.exists():
            print(f"Override mode: Clearing existing directory {output_path}")
            shutil.rmtree(output_path)
        output_path.mkdir(exist_ok=True)
    elif mode == "append":
        # TODO: Implement append mode logic
        # - Check existing files to determine next img_num
        # - Avoid overwriting existing files
        # - Update summary with new additions
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
    
    total_frames_processed = 0
    
    for img_num, (video_file, annotation_file) in enumerate(matches, 1):
        print(f"\nProcessing video {img_num}/{len(matches)}: {video_file.name}")
        print(f"Video file: {video_file}")
        print(f"Annotation file: {annotation_file}")
        
        try:
            temp_frames_dir = output_path / f"temp_frames_{img_num}"
            temp_frames_dir.mkdir(exist_ok=True)
            
            ffmpeg_cmd = [
                'ffmpeg', '-i', str(video_file),
                '-vf', f'fps={FPS}', 
                str(temp_frames_dir / 'frame_%04d.jpg'),
                '-y'  # Overwrite existing files
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error extracting frames from {video_file.name}: {result.stderr}")
                shutil.rmtree(temp_frames_dir, ignore_errors=True)
                continue
            
            frame_files = sorted(temp_frames_dir.glob('frame_*.jpg'))
            num_video_frames = len(frame_files)
            
            # Get image dimensions from the first frame
            if frame_files:
                with Image.open(frame_files[0]) as img:
                    print(img.size)
                    img_width, img_height = img.size
            else:
                raise ValueError("Couldn't get image dimensions")
            
            print(f"Converting {annotation_file.name} to YOLO format...")
            print(f"Using image dimensions: {img_width}x{img_height}")
            frame_annotations = convert_to_yolo(annotation_file, img_width, img_height)
            num_annotation_frames = len(frame_annotations)
            
            if num_video_frames != num_annotation_frames:
                print(f"Warning: Frame count mismatch for {video_file.name}")
                print(f"Video frames: {num_video_frames}, Annotation frames: {num_annotation_frames}")
                num_frames_to_process = min(num_video_frames, num_annotation_frames)
            else:
                num_frames_to_process = num_video_frames
                print(f"Successfully matched {num_frames_to_process} frames")
            
            for frame_idx in range(num_frames_to_process):
                frame_num = frame_idx + 1
                
                image_name = f"{img_num:04d}_{frame_num:04d}.jpg"
                annotation_name = f"{img_num:04d}_{frame_num:04d}.txt"
                
                source_frame = frame_files[frame_idx]
                dest_image = images_output / image_name
                shutil.copy2(source_frame, dest_image)
                
                dest_annotation = labels_output / annotation_name
                with open(dest_annotation, 'w') as f:
                    f.write(frame_annotations[frame_idx])
            
            total_frames_processed += num_frames_to_process
            print(f"Processed {num_frames_to_process} frames from {video_file.name}")
            
            shutil.rmtree(temp_frames_dir, ignore_errors=True)
            
        except Exception as e:
            print(f"Error processing {video_file.name}: {e}")
            if 'temp_frames_dir' in locals():
                shutil.rmtree(temp_frames_dir, ignore_errors=True)
            continue
    
    print(f"\nDataset creation complete!")
    print(f"Total videos processed: {len(matches)}")
    print(f"Total frames extracted: {total_frames_processed}")
    print(f"Images saved to: {images_output}")
    print(f"Labels saved to: {labels_output}")

def split_dataset(dataset_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split the cleaned dataset into train, validation, and test sets.
    
    Args:
        dataset_folder (str/Path): Path to the cleaned dataset folder containing 'images' and 'labels' subfolders
        train_ratio (float): Proportion of data for training (default: 0.8)
        val_ratio (float): Proportion of data for validation (default: 0.1) 
        test_ratio (float): Proportion of data for testing (default: 0.1)
    """
    # This could be a try-except block that adjusts the values
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0. Current sum: {train_ratio + val_ratio + test_ratio}")

    dataset_path = Path(dataset_folder)
    images_folder = dataset_path / 'images'
    labels_folder = dataset_path / 'labels'
    
    if not images_folder.exists() or not labels_folder.exists():
        print(f"Error: Expected 'images' and 'labels' subfolders in {dataset_path}")
        return
    
    image_files = sorted(list(images_folder.glob('*.jpg')))
    label_files = sorted(list(labels_folder.glob('*.txt')))
    
    if len(image_files) != len(label_files):
        print(f"Warning: Mismatch between images ({len(image_files)}) and labels ({len(label_files)})")
        return
    
    if len(image_files) == 0:
        print("Error: No image files found to split")
        return
    
    
    print(f"Splitting {len(image_files)} samples into train/val/test...")
    
    total_samples = len(image_files)
    indices = list(range(total_samples))
    random.seed(42)  
    random.shuffle(indices)
    
    train_end = int(train_ratio * total_samples)
    val_end = train_end + int(val_ratio * total_samples)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
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
    print(f"Train: {len(train_indices)} samples ({len(train_indices)/total_samples:.1%})")
    print(f"Val:   {len(val_indices)} samples ({len(val_indices)/total_samples:.1%})")
    print(f"Test:  {len(test_indices)} samples ({len(test_indices)/total_samples:.1%})")
    print(f"\nSplit directories created:")
    print(f"  {dataset_path / 'images' / 'train'}")
    print(f"  {dataset_path / 'images' / 'val'}")
    print(f"  {dataset_path / 'images' / 'test'}")
    print(f"  {dataset_path / 'labels' / 'train'}")
    print(f"  {dataset_path / 'labels' / 'val'}")
    print(f"  {dataset_path / 'labels' / 'test'}")

def main():
    #TODO: Add a verbose flag, if false then don't print as much
    parser = argparse.ArgumentParser(description='Create cleaned dataset by matching videos with annotations')
    parser.add_argument('video_folder', help='Path to video dataset folder')
    parser.add_argument('annotation_folder', help='Path to annotations folder')
    parser.add_argument('--is_3d', action='store_true', help='Use 3D annotations (default: False for 2D)')
    parser.add_argument('--annotation_type', choices=['base', 'body25', 'coco'], default='body25',
                       help='Annotation type to use (default: body25)')
    parser.add_argument('--test_matches', action='store_true', help='Test mode: only find matches without creating dataset')
    parser.add_argument('--limit_videos', type=int, help='Maximum number of videos to process')
    parser.add_argument('--mode', choices=['override', 'append'], default='override',
                       help='Dataset creation mode: override (clear directory) or append (default: override)')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Proportion of data for training (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Proportion of data for validation (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Proportion of data for testing (default: 0.1)')

    args = parser.parse_args()
    
    if not os.path.exists(args.video_folder):
        print(f"Error: Video folder '{args.video_folder}' does not exist")
        return
    
    if not os.path.exists(args.annotation_folder):
        print(f"Error: Annotation folder '{args.annotation_folder}' does not exist")
        return
    
    print(f"Video folder: {args.video_folder}")
    print(f"Annotation folder: {args.annotation_folder}")
    print(f"3D annotations: {args.is_3d}")
    print(f"Annotation type: {args.annotation_type}")
    print("-" * 50)
    
    matches = find_matching_files(args.video_folder, args.annotation_folder, 
                                args.is_3d, args.annotation_type)
    
    if not matches:
        print("No matching video-annotation pairs found!")
        print("Please check:")
        print("1. Video and annotation folder paths are correct")
        print("2. Annotation type folder exists")
        print("3. 2D/3D files exist with correct naming")
        return
    
    print(f"Found {len(matches)} matching pairs:")
    
    if args.test_matches:
        print("Test matches mode complete. Use without --test_matches flag to create cleaned dataset.")
        return
    
    video_folder_name = Path(args.video_folder).name
    output_folder = Path.cwd() / f"cleaned_{video_folder_name}"
    print(f"Output folder: {output_folder}")
    print("-" * 50)
    
    create_cleaned_dataset(matches, output_folder, args.limit_videos, args.mode)
    
    split_dataset(output_folder, args.train_ratio, args.val_ratio, args.test_ratio)

if __name__ == "__main__":
    main()
    #convert_coco_to_yolo('/Users/artemkiryukhin/Downloads/Freestyle/Front/Swimmer_Skin_0,75_Muscle_8/Water_Quantity_0,75_Height_1,5/Lighting_rotx_110_roty_190/Speed_3/position_3,75/body25/2D_cam.txt')