import argparse
import cv2
import shutil
import subprocess
from pathlib import Path
from PIL import Image
from format_conversion import convert_frame

FPS_DEFAULT = 60


def process_single(
    video_path: str,
    annotation_file: str,
    output_dir: str,
    fps: int = FPS_DEFAULT,
    prefix: str | None = None,
    min_visible_ratio: float = 0.5,
):
    """
    Extract frames from a single video and convert its annotations to YOLO
    keypoint .txt format with occlusion detection and min-visible filtering.

    Args:
        video_path:         Path to the video file (.webm/.mp4/etc.)
        annotation_file:    Path to the SwimXYZ annotation .txt file.
        output_dir:         Output directory (will contain images/ and labels/).
        fps:                Frame extraction rate.
        prefix:             Filename prefix for outputs (default: video stem).
        min_visible_ratio:  Min fraction of keypoints in-bounds to keep a frame.
    """
    video_path      = Path(video_path)
    annotation_file = Path(annotation_file)
    output_root     = Path(output_dir)
    images_output   = output_root / 'images'
    labels_output   = output_root / 'labels'
    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation not found: {annotation_file}")

    # Detect camera view for occlusion
    try:
        from occlusion import detect_view_from_path
    except ImportError:
        from data_processing.occlusion import detect_view_from_path
    camera_view = detect_view_from_path(video_path)
    if camera_view:
        print(f"Camera view detected: {camera_view}")
    else:
        print("Camera view not recognised — using bounds-only visibility.")

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

        # Read annotation file
        with open(annotation_file, 'r') as f:
            ann_lines = f.readlines()

        if len(ann_lines) < 2:
            print(f"Warning: annotation file has insufficient data.")
            return

        header     = ann_lines[0].strip().rstrip(';').split(';')
        data_lines = ann_lines[1:]

        num_video_frames      = len(frame_files)
        num_annotation_frames = len(data_lines)
        if num_video_frames != num_annotation_frames:
            print(
                f"Warning: frame count mismatch — "
                f"video={num_video_frames}, annotation={num_annotation_frames}"
            )
        num_frames_to_process = min(num_video_frames, num_annotation_frames)

        # Naming prefix
        file_prefix = prefix if prefix else video_path.stem

        frames_saved   = 0
        frames_skipped = 0

        for idx in range(num_frames_to_process):
            source_frame = frame_files[idx]

            # Parse annotation row
            ann_line = data_lines[idx].strip().rstrip(';').replace(',', '.')
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

            frame_num  = idx + 1
            image_name = f"{file_prefix}_{frame_num:04d}.jpg"
            label_name = f"{file_prefix}_{frame_num:04d}.txt"

            shutil.copy2(source_frame, images_output / image_name)
            with open(labels_output / label_name, 'w') as f:
                f.write(annotation)
            frames_saved += 1

        print(f"Saved {frames_saved} frames, skipped {frames_skipped}.")

    finally:
        shutil.rmtree(temp_frames_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Extract frames from a single video and convert its annotations to '
            'YOLO keypoint .txt format with occlusion detection.'
        )
    )
    parser.add_argument('video',      type=str, help='Path to the video file (.webm/.mp4/etc.)')
    parser.add_argument('annotation', type=str, help='Path to the SwimXYZ annotation .txt file')
    parser.add_argument('--output_dir', type=str,
                        default=str(Path.cwd() / 'cleaned_single'),
                        help='Output directory (will contain images/ and labels/)')
    parser.add_argument('--fps', type=int, default=FPS_DEFAULT,
                        help='Frame extraction rate (frames per second)')
    parser.add_argument('--prefix', type=str, default=None,
                        help='Filename prefix for outputs (default: video stem)')
    parser.add_argument('--min_visible_ratio', type=float, default=0.5,
                        help='Min fraction of keypoints in-bounds to keep frame (default: 0.5)')
    args = parser.parse_args()

    process_single(
        args.video, args.annotation, args.output_dir,
        fps=args.fps, prefix=args.prefix,
        min_visible_ratio=args.min_visible_ratio,
    )


if __name__ == '__main__':
    main()
