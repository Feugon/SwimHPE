"""
reconstruct_dataset.py — Reconstruct hand-labeled frames from a manifest.

Downloads YouTube videos via yt-dlp, extracts the exact frames, and writes
YOLO pose label files from the stored keypoint coordinates.

Usage:
    python data_processing/reconstruct_dataset.py
    python data_processing/reconstruct_dataset.py --manifest annotations/manifest.json \
                                                  --output-dir unlabeled_data/reconstructed
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from download_yt import download_video

# Inline the two items we need from format_conversion to avoid its ultralytics import
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


def calculate_bounding_box(keypoint_coords, padding=20, img_width=640, img_height=360):
    """Compute a normalized [x_c, y_c, w, h] bounding box from visible keypoints."""
    xs = [v['x'] * img_width  for v in keypoint_coords.values() if v['v'] != 1.0]
    ys = [v['y'] * img_height for v in keypoint_coords.values() if v['v'] != 1.0]
    if not xs:
        return None
    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = min(img_width,  max(xs) + padding)
    y2 = min(img_height, max(ys) + padding)
    x_c = ((x1 + x2) / 2) / img_width
    y_c = ((y1 + y2) / 2) / img_height
    w   = (x2 - x1) / img_width
    h   = (y2 - y1) / img_height
    return x_c, y_c, w, h

KEYPOINT_VISIBLE = 2.0
KEYPOINT_NOT_VISIBLE = 1.0
KEYPOINT_MISSING = 0.0

DEFAULT_FMT = "best[height<=1080]"


def extract_specific_frames(video_path: Path, out_dir: Path, fps: int,
                            frame_numbers: set[int]) -> dict[int, Path]:
    """Extract all frames at the given fps, keep only the needed ones.

    Returns {frame_number: Path} for frames that were successfully extracted.
    """
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        result = subprocess.run(
            ['ffmpeg', '-i', str(video_path),
             '-vf', f'fps={fps}',
             str(tmp_dir / 'frame_%04d.jpg'),
             '-y'],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[:300]}")

        out_dir.mkdir(parents=True, exist_ok=True)
        extracted = {}
        for frame_num in frame_numbers:
            src = tmp_dir / f"frame_{frame_num:04d}.jpg"
            if src.exists():
                dst = out_dir / src.name
                shutil.copy2(src, dst)
                extracted[frame_num] = dst
            else:
                print(f"    WARNING: frame_{frame_num:04d}.jpg not found in extracted frames")
        return extracted
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def manifest_entry_to_yolo(entry: dict) -> str:
    """Convert a manifest entry to YOLO pose label lines (one line per person)."""
    img_w = entry["img_width"]
    img_h = entry["img_height"]
    lines = []

    for person in entry["persons"]:
        # Initialize 17 keypoint slots
        kps = [[0.0, 0.0, KEYPOINT_MISSING] for _ in range(17)]

        for kp in person["keypoints"]:
            joint = kp["joint"].lower()
            idx = XANY_TO_COCO17_IDX.get(joint)
            if idx is None:
                continue
            vis = KEYPOINT_NOT_VISIBLE if kp.get("difficult", False) else KEYPOINT_VISIBLE
            kps[idx] = [round(kp["x"] / img_w, 6), round(kp["y"] / img_h, 6), vis]

        # Skip persons with no annotated keypoints
        if all(k[2] == KEYPOINT_MISSING for k in kps):
            continue

        # Bounding box
        bbox_px = person.get("bbox_pixels")
        if bbox_px:
            x1, y1, x2, y2 = bbox_px
            x_center = round(((x1 + x2) / 2) / img_w, 6)
            y_center = round(((y1 + y2) / 2) / img_h, 6)
            width = round((x2 - x1) / img_w, 6)
            height = round((y2 - y1) / img_h, 6)
        else:
            kp_coords = {
                str(i): {'x': k[0], 'y': k[1], 'v': k[2]}
                for i, k in enumerate(kps) if k[2] > 0.0
            }
            bbox = calculate_bounding_box(kp_coords, padding=20,
                                          img_width=img_w, img_height=img_h)
            if bbox is None:
                continue
            x_center, y_center, width, height = bbox

        kp_str = " ".join(f"{k[0]:.6f} {k[1]:.6f} {k[2]:.6f}" for k in kps)
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {kp_str}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct hand-labeled frames from a manifest."
    )
    parser.add_argument(
        "--manifest", default="annotations/manifest.json",
        help="Path to manifest.json (default: annotations/manifest.json)"
    )
    parser.add_argument(
        "--output-dir", default="unlabeled_data/reconstructed",
        help="Root output directory (default: unlabeled_data/reconstructed)"
    )
    parser.add_argument(
        "--format", default=DEFAULT_FMT, dest="fmt",
        help=f"yt-dlp format string (default: {DEFAULT_FMT!r})"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip downloading; only write YOLO labels from manifest (frames must already exist)"
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)

    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Manifest: {len(manifest)} annotated frames")

    # Group entries by youtube_id
    by_video: dict[str, list[dict]] = defaultdict(list)
    for entry in manifest:
        by_video[entry["youtube_id"]].append(entry)

    print(f"Videos: {len(by_video)}")
    print("-" * 50)

    total_frames = 0
    total_labels = 0

    for vid_id, entries in by_video.items():
        dir_name = entries[0]["dir_name"]
        fps = entries[0]["fps"]
        frame_numbers = {e["frame_number"] for e in entries}
        vid_out_dir = output_dir / dir_name

        print(f"\n[{vid_id}] {dir_name}")
        print(f"  Frames needed: {len(frame_numbers)}")

        if not args.skip_download:
            url = f"https://www.youtube.com/watch?v={vid_id}"
            tmp_file = Path(tempfile.mktemp(suffix=".mp4"))
            try:
                print("  Downloading...")
                download_video(url, args.fmt, tmp_file)

                print(f"  Extracting frames at {fps} fps...")
                extracted = extract_specific_frames(tmp_file, vid_out_dir, fps, frame_numbers)
                print(f"  Extracted: {len(extracted)} frames")
                total_frames += len(extracted)
            except RuntimeError as e:
                print(f"  ERROR: {e}")
                print("  Skipping video.")
                continue
            finally:
                if tmp_file.exists():
                    tmp_file.unlink()
        else:
            print("  Skipping download (--skip-download)")

        # Write YOLO labels
        labels_written = 0
        for entry in entries:
            frame_num = entry["frame_number"]
            yolo_lines = manifest_entry_to_yolo(entry)
            if not yolo_lines:
                continue
            label_path = vid_out_dir / f"frame_{frame_num:04d}.txt"
            vid_out_dir.mkdir(parents=True, exist_ok=True)
            label_path.write_text(yolo_lines + "\n")
            labels_written += 1

        print(f"  Labels written: {labels_written}")
        total_labels += labels_written

    print("\n" + "-" * 50)
    print(f"Done.")
    print(f"  Total frames extracted: {total_frames}")
    print(f"  Total labels written: {total_labels}")
    print(f"  Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
