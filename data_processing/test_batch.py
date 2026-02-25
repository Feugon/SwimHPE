"""
Test-batch runner for the SwimHPE data pipeline.

Processes a small sample of videos from each side-view camera
(Side_above_water, Side_water_level, Side_underwater) and prints a
per-view quality report.  Optionally saves annotated PNG frames for
visual inspection.

Usage:
    python data_processing/test_batch.py \\
        /path/to/videos /path/to/annotations \\
        [--n_videos 3]          # videos per camera view (default: 3)
        [--output_dir test_output]
        [--visualize]           # save annotated PNG frames every --viz_interval frames
        [--viz_interval 30]
        [--annotation_type body25]
        [--min_visible_ratio 0.5]
        [--seed 42]

Output structure:
    <output_dir>/
        Side_above_water/
            images/   *.jpg
            labels/   *.txt
            viz/      *_annotated.png  (if --visualize)
        Side_water_level/  ...
        Side_underwater/   ...
        summary.txt
"""

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Allow running from project root: python data_processing/test_batch.py
sys.path.insert(0, str(Path(__file__).parent))

from format_conversion import convert_frame, KEYPOINT_VISIBLE, KEYPOINT_OCCLUDED, KEYPOINT_NOT_VISIBLE
from occlusion import detect_view_from_path, KNOWN_VIEWS
from process_mixed import collect_matches_per_view
from keypoint_mapping import COCO_KP_NAMES, COCO_KP_INDEX

FPS = 60

# Visualization colors (BGR for cv2)
_COLOR_VISIBLE = (0,   220,   0)   # green
_COLOR_OCCLUDED = (130, 90,  220)  # purple-pink (all v=0, self or water)
_COLOR_OOB      = (160, 160, 160)  # grey (out of bounds)


def _process_video(
    video_file: Path,
    annotation_file: Path,
    images_out: Path,
    labels_out: Path,
    camera_view: str | None,
    img_num: int,
    min_visible_ratio: float,
) -> dict:
    """
    Extract frames from one video and convert annotations.

    Returns a stats dict:
        total_frames, frames_saved, frames_skipped
    """
    temp_dir = images_out.parent / f"_tmp_{img_num}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    stats = {'total_frames': 0, 'frames_saved': 0, 'frames_skipped': 0}

    try:
        result = subprocess.run(
            [
                'ffmpeg', '-i', str(video_file),
                '-vf', f'fps={FPS}',
                str(temp_dir / 'frame_%04d.jpg'),
                '-y',
            ],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"    FFmpeg failed: {result.stderr[:120]}")
            return stats

        frame_files = sorted(temp_dir.glob('frame_*.jpg'))
        if not frame_files:
            return stats

        with Image.open(frame_files[0]) as im:
            img_w, img_h = im.size

        with open(annotation_file, 'r') as f:
            ann_lines = f.readlines()
        if len(ann_lines) < 2:
            return stats

        header     = ann_lines[0].strip().rstrip(';').split(';')
        data_lines = ann_lines[1:]

        n_frames = min(len(frame_files), len(data_lines))
        stats['total_frames'] = n_frames

        for idx in range(n_frames):
            src_frame = frame_files[idx]
            ann_line  = data_lines[idx].strip().rstrip(';').replace(',', '.')
            try:
                values = [float(v) for v in ann_line.split(';') if v.strip()]
            except ValueError:
                stats['frames_skipped'] += 1
                continue

            frame_image = cv2.imread(str(src_frame)) if camera_view else None

            annotation = convert_frame(
                header, values, img_w, img_h,
                frame_image=frame_image,
                camera_view=camera_view,
                min_visible_ratio=min_visible_ratio,
            )

            if annotation is None:
                stats['frames_skipped'] += 1
                continue

            frame_num  = idx + 1
            stem       = f"{img_num:04d}_{frame_num:04d}"
            shutil.copy2(src_frame, images_out / f"{stem}.jpg")
            (labels_out / f"{stem}.txt").write_text(annotation)
            stats['frames_saved'] += 1

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return stats


def _draw_keypoints(image: np.ndarray, label_path: Path) -> np.ndarray:
    """
    Draw YOLO keypoints on a copy of `image` with color-coded visibility.

    Reads the label file, de-normalizes coordinates, and draws:
      - Green  circle: v = 2.0 (visible)
      - Purple circle: v = 0.0 (occluded, self or water)
      - Grey   dot:    v = 1.0 (out of bounds)
    """
    h, w = image.shape[:2]
    out  = image.copy()

    if not label_path.exists():
        return out

    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        # parts[0] = class, 1-4 = bbox, 5+ = keypoints (x y v) × 13
        kp_data = parts[5:]
        n_kp = len(kp_data) // 3
        for i in range(n_kp):
            nx = float(kp_data[i * 3])
            ny = float(kp_data[i * 3 + 1])
            v  = float(kp_data[i * 3 + 2])

            px = int(np.clip(nx * w, 0, w - 1))
            py = int(np.clip(ny * h, 0, h - 1))

            if v == KEYPOINT_VISIBLE:
                color  = _COLOR_VISIBLE
                radius = 5
                thick  = -1
            elif v == KEYPOINT_OCCLUDED:
                color  = _COLOR_OCCLUDED
                radius = 6
                thick  = 2   # hollow ring for occluded
            else:  # out of bounds
                color  = _COLOR_OOB
                radius = 3
                thick  = -1

            cv2.circle(out, (px, py), radius, color, thick)

            # Keypoint name label
            name = COCO_KP_NAMES[i] if i < len(COCO_KP_NAMES) else str(i)
            cv2.putText(out, name, (px + 6, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)

    return out


def _visualize_view(
    images_dir: Path,
    labels_dir: Path,
    viz_dir: Path,
    interval: int,
):
    """Save annotated PNG for every `interval`-th saved frame in a view directory."""
    viz_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted(images_dir.glob('*.jpg'))
    saved_count = 0
    for i, img_path in enumerate(image_files):
        if i % interval != 0:
            continue
        label_path = labels_dir / img_path.with_suffix('.txt').name
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        annotated    = _draw_keypoints(img, label_path)
        out_png      = viz_dir / (img_path.stem + '_annotated.png')
        cv2.imwrite(str(out_png), annotated)
        saved_count += 1
    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description="Test-batch runner: process a small sample and inspect quality."
    )
    parser.add_argument("video_root",      help="Root folder with per-view video subfolders")
    parser.add_argument("annotation_root", help="Root folder with per-view annotation subfolders")
    parser.add_argument("--n_videos",   type=int,   default=3,
                        help="Videos to sample per view (default: 3)")
    parser.add_argument("--output_dir", type=str,   default="test_output",
                        help="Output directory (default: test_output)")
    parser.add_argument("--visualize",  action="store_true",
                        help="Save annotated PNG frames for inspection")
    parser.add_argument("--viz_interval", type=int, default=30,
                        help="Save every Nth saved frame as PNG (default: 30)")
    parser.add_argument("--annotation_type", choices=["base", "body25", "coco"],
                        default="body25")
    parser.add_argument("--min_visible_ratio", type=float, default=0.5,
                        help="Min in-bounds keypoint fraction to keep frame (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SwimHPE Test Batch Runner")
    print("=" * 60)
    print(f"Video root:       {args.video_root}")
    print(f"Annotation root:  {args.annotation_root}")
    print(f"Videos per view:  {args.n_videos}")
    print(f"Min visible:      {args.min_visible_ratio:.0%}")
    print(f"Output:           {output_root}")
    print("=" * 60)

    # Collect matches per view (side views only)
    print("\n--- Finding matches ---")
    matches_by_view = collect_matches_per_view(
        args.video_root, args.annotation_root, args.annotation_type
    )
    if not matches_by_view:
        print("No side-view matches found. Check video_root / annotation_root paths.")
        return

    # Sample n_videos per view
    random.seed(args.seed)
    sampled_by_view: dict[str, list] = {}
    for view, matches in sorted(matches_by_view.items()):
        n = min(args.n_videos, len(matches))
        sampled_by_view[view] = random.sample(matches, n)
        print(f"  {view}: sampling {n}/{len(matches)} videos")

    # Process each view
    all_stats: dict[str, dict] = {}

    for view, sample in sampled_by_view.items():
        view_dir    = output_root / view
        images_out  = view_dir / 'images'
        labels_out  = view_dir / 'labels'
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        camera_view = KNOWN_VIEWS[view]   # normalized name (e.g., 'water_level')

        view_stats = {'total_frames': 0, 'frames_saved': 0, 'frames_skipped': 0}

        print(f"\n--- Processing {view} ({len(sample)} videos) ---")
        for i, (video_file, annotation_file) in enumerate(sample, 1):
            print(f"  [{i}/{len(sample)}] {video_file.name}")
            vstats = _process_video(
                video_file, annotation_file,
                images_out, labels_out,
                camera_view=camera_view,
                img_num=(i + len(all_stats) * 1000),   # unique prefix across views
                min_visible_ratio=args.min_visible_ratio,
            )
            view_stats['total_frames']  += vstats['total_frames']
            view_stats['frames_saved']  += vstats['frames_saved']
            view_stats['frames_skipped'] += vstats['frames_skipped']
            print(
                f"    → saved {vstats['frames_saved']}, "
                f"skipped {vstats['frames_skipped']} / {vstats['total_frames']}"
            )

        all_stats[view] = view_stats

        if args.visualize:
            viz_dir = view_dir / 'viz'
            n_viz = _visualize_view(images_out, labels_out, viz_dir, args.viz_interval)
            print(f"  Saved {n_viz} annotated PNGs → {viz_dir}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    header_fmt = f"{'View':<22} {'Total':>8} {'Saved':>8} {'Skipped':>8} {'Keep%':>7}"
    print(header_fmt)
    print("-" * 60)

    grand_total = grand_saved = grand_skipped = 0
    summary_lines = [header_fmt, "-" * 60]

    for view, s in sorted(all_stats.items()):
        total   = s['total_frames']
        saved   = s['frames_saved']
        skipped = s['frames_skipped']
        pct     = f"{saved/total:.1%}" if total else "n/a"
        row = f"{view:<22} {total:>8} {saved:>8} {skipped:>8} {pct:>7}"
        print(row)
        summary_lines.append(row)
        grand_total   += total
        grand_saved   += saved
        grand_skipped += skipped

    print("-" * 60)
    grand_pct = f"{grand_saved/grand_total:.1%}" if grand_total else "n/a"
    totals_row = f"{'TOTAL':<22} {grand_total:>8} {grand_saved:>8} {grand_skipped:>8} {grand_pct:>7}"
    print(totals_row)
    summary_lines += ["-" * 60, totals_row]

    summary_path = output_root / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"\nSummary written to: {summary_path}")

    if args.visualize:
        print("\nVisualization key:")
        print("  Green circle  — visible (v=2.0)")
        print("  Purple ring   — occluded, self or water (v=0.0)")
        print("  Grey dot      — out of bounds (v=1.0)")


if __name__ == "__main__":
    main()
