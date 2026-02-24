"""
Process a mixed dataset from multiple camera views.

Usage:
    python process_mixed.py /path/to/videos /path/to/annotations --output_dir /path/to/output --max_frames 50000
"""

import argparse
import random
from pathlib import Path
from prep_data import find_matching_files, create_cleaned_dataset, split_dataset


def collect_matches_per_view(video_root, annotation_root, annotation_type="body25", is_3d=False):
    """Find video-annotation matches grouped by camera view."""
    video_root = Path(video_root)
    annotation_root = Path(annotation_root)

    views = [d.name for d in video_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    print(f"Found views: {views}")

    matches_by_view = {}
    for view in sorted(views):
        video_folder = video_root / view
        annotation_folder = annotation_root / view
        if not annotation_folder.exists():
            print(f"  Skipping {view}: no matching annotation folder")
            continue
        matches = find_matching_files(video_folder, annotation_folder, is_3d, annotation_type)
        if matches:
            matches_by_view[view] = matches
            print(f"  {view}: {len(matches)} video-annotation pairs")
        else:
            print(f"  {view}: no matches found")

    return matches_by_view


def sample_matches(matches_by_view, max_frames, frames_per_video=235, seed=42):
    """
    Sample roughly equal number of videos from each view to hit max_frames.

    Args:
        matches_by_view: dict mapping view name to list of (video, annotation) tuples
        max_frames: target total frame count
        frames_per_video: estimated usable frames per video (after removing out-of-frame)
        seed: random seed for reproducibility
    """
    random.seed(seed)
    num_views = len(matches_by_view)
    frames_per_view = max_frames // num_views
    videos_per_view = max(1, frames_per_view // frames_per_video)

    sampled = []
    for view, matches in sorted(matches_by_view.items()):
        n = min(videos_per_view, len(matches))
        selected = random.sample(matches, n)
        sampled.extend(selected)
        print(f"  {view}: sampled {n} videos (~{n * frames_per_video} frames)")

    random.shuffle(sampled)
    print(f"\nTotal sampled: {len(sampled)} videos (~{len(sampled) * frames_per_video} estimated frames)")
    return sampled


def main():
    parser = argparse.ArgumentParser(description="Process a mixed multi-view swim dataset.")
    parser.add_argument("video_root", help="Root folder containing per-view video subfolders")
    parser.add_argument("annotation_root", help="Root folder containing per-view annotation subfolders")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: cleaned_mixed in cwd)")
    parser.add_argument("--max_frames", type=int, default=50000, help="Target number of frames")
    parser.add_argument("--annotation_type", choices=["base", "body25", "coco"], default="body25")
    parser.add_argument("--is_3d", action="store_true")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir or str(Path.cwd() / "cleaned_mixed")

    print("=== Finding matches per view ===")
    matches_by_view = collect_matches_per_view(
        args.video_root, args.annotation_root, args.annotation_type, args.is_3d
    )

    if not matches_by_view:
        print("No matches found across any view!")
        return

    print(f"\n=== Sampling ~{args.max_frames} frames across {len(matches_by_view)} views ===")
    sampled_matches = sample_matches(matches_by_view, args.max_frames, seed=args.seed)

    print(f"\n=== Creating dataset in {output_dir} ===")
    create_cleaned_dataset(sampled_matches, output_dir)

    print(f"\n=== Splitting dataset ===")
    split_dataset(output_dir, args.train_ratio, args.val_ratio, args.test_ratio)


if __name__ == "__main__":
    main()
