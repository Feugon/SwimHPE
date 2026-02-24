import argparse
import os
import shutil
import subprocess
from pathlib import Path
from PIL import Image
from format_conversion import convert_to_yolo

FPS_DEFAULT = 60


def process_single(video_path: str, annotation_file: str, output_dir: str, fps: int = FPS_DEFAULT, prefix: str | None = None):
    video_path = Path(video_path)
    annotation_file = Path(annotation_file)
    output_root = Path(output_dir)
    images_output = output_root / 'images'
    labels_output = output_root / 'labels'
    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation not found: {annotation_file}")

    # Extract frames
    temp_frames_dir = output_root / f"temp_frames_{video_path.stem}"
    temp_frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        ffmpeg_cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', f'fps={fps}',
            str(temp_frames_dir / 'frame_%04d.jpg'),
            '-y'
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")

        frame_files = sorted(temp_frames_dir.glob('frame_*.jpg'))
        if not frame_files:
            raise RuntimeError("No frames extracted from video")

        # Determine image dimensions from first frame
        with Image.open(frame_files[0]) as img:
            img_width, img_height = img.size

        # Convert annotation file to YOLO pose format lines
        frame_annotations = convert_to_yolo(str(annotation_file), img_width, img_height)
        if frame_annotations is None:
            frame_annotations = []

        num_video_frames = len(frame_files)
        num_annotation_frames = len(frame_annotations)
        num_frames_to_process = min(num_video_frames, num_annotation_frames) if num_annotation_frames else 0

        if num_frames_to_process == 0:
            # If no per-frame lines provided, still export frames with empty labels
            num_frames_to_process = num_video_frames
            frame_annotations = [""] * num_frames_to_process

        # Naming prefix
        file_prefix = prefix if prefix else video_path.stem

        for idx in range(num_frames_to_process):
            # Skip frames where the swimmer is outside the frame (empty annotation)
            annotation = frame_annotations[idx] if idx < len(frame_annotations) else ""
            if not annotation or annotation.strip() == "":
                continue

            frame_num = idx + 1
            image_name = f"{file_prefix}_{frame_num:04d}.jpg"
            label_name = f"{file_prefix}_{frame_num:04d}.txt"

            shutil.copy2(frame_files[idx], images_output / image_name)
            with open(labels_output / label_name, 'w') as f:
                f.write(annotation)
    finally:
        shutil.rmtree(temp_frames_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description='Extract frames from a single video and convert its annotations to YOLO keypoint .txt (COCO-Pose) without splitting.')
    parser.add_argument('video', type=str, help='Path to the video file (.webm/.mp4/etc.)')
    parser.add_argument('annotation', type=str, help='Path to the source annotation file (COCO-style text as used in context)')
    parser.add_argument('--output_dir', type=str, default=str(Path.cwd() / 'cleaned_single'), help='Output directory (will contain images/ and labels/)')
    parser.add_argument('--fps', type=int, default=FPS_DEFAULT, help='Frame extraction rate (frames per second)')
    parser.add_argument('--prefix', type=str, default=None, help='Filename prefix for outputs (default: video stem)')
    args = parser.parse_args()

    process_single(args.video, args.annotation, args.output_dir, fps=args.fps, prefix=args.prefix)


if __name__ == '__main__':
    main()
