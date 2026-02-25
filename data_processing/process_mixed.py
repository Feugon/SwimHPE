"""
Process a mixed dataset from multiple camera views.

Only Side_above_water, Side_water_level, and Side_underwater views are exported.
Front and Aerial views are excluded.

Usage:
    python process_mixed.py /path/to/videos /path/to/annotations \\
        --output_dir /path/to/output --max_frames 50000
"""

import argparse
import random
from pathlib import Path
from prep_data import find_matching_files, create_cleaned_dataset, split_dataset
from occlusion import KNOWN_VIEWS


def collect_matches_per_view(video_root, annotation_root, annotation_type="body25", is_3d=False):
    """
    Find video-annotation matches grouped by camera view.

    Only the three side views (Side_above_water, Side_water_level, Side_underwater)
    are included.  Front and Aerial subdirectories are ignored.
    """
    video_root      = Path(video_root)
    annotation_root = Path(annotation_root)

    # Filter to known side-view directories only
    all_dirs = [d for d in video_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    side_view_dirs = [d for d in all_dirs if d.name in KNOWN_VIEWS]

    if not side_view_dirs:
        print(f"No supported side-view subdirectories found in {video_root}")
        print(f"Expected one or more of: {sorted(KNOWN_VIEWS.keys())}")
        return {}

    views = sorted(d.name for d in side_view_dirs)
    print(f"Found side views: {views}")

    matches_by_view = {}
    for view in views:
        video_folder      = video_root      / view
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
        matches_by_view:  dict mapping view name to list of (video, annotation) tuples
        max_frames:       target total frame count
        frames_per_video: estimated usable frames per video (after filtering)
        seed:             random seed for reproducibility
    """
    random.seed(seed)
    num_views        = len(matches_by_view)
    frames_per_view  = max_frames // num_views
    videos_per_view  = max(1, frames_per_view // frames_per_video)

    sampled = []
    for view, matches in sorted(matches_by_view.items()):
        n        = min(videos_per_view, len(matches))
        selected = random.sample(matches, n)
        sampled.extend(selected)
        print(f"  {view}: sampled {n} videos (~{n * frames_per_video} frames)")

    random.shuffle(sampled)
    print(
        f"\nTotal sampled: {len(sampled)} videos "
        f"(~{len(sampled) * frames_per_video} estimated frames)"
    )
    return sampled


def main():
    parser = argparse.ArgumentParser(description="Process a mixed multi-view swim dataset.")
    parser.add_argument("video_root",      help="Root folder containing per-view video subfolders")
    parser.add_argument("annotation_root", help="Root folder containing per-view annotation subfolders")
    parser.add_argument("--output_dir",    default=None,
                        help="Output directory (default: cleaned_mixed in cwd)")
    parser.add_argument("--max_frames",    type=int, default=50000,
                        help="Target number of frames")
    parser.add_argument("--annotation_type", choices=["base", "body25", "coco"], default="body25")
    parser.add_argument("--is_3d",         action="store_true")
    parser.add_argument("--train_ratio",   type=float, default=0.8)
    parser.add_argument("--val_ratio",     type=float, default=0.1)
    parser.add_argument("--test_ratio",    type=float, default=0.1)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--min_visible_ratio", type=float, default=0.5,
                        help="Min fraction of keypoints in-bounds to keep frame (default: 0.5)")
    args = parser.parse_args()

    output_dir = args.output_dir or str(Path.cwd() / "cleaned_mixed")

    print("=== Finding matches per view (side views only) ===")
    matches_by_view = collect_matches_per_view(
        args.video_root, args.annotation_root, args.annotation_type, args.is_3d
    )

    if not matches_by_view:
        print("No matches found across any supported view!")
        return

    print(
        f"\n=== Sampling ~{args.max_frames} frames across "
        f"{len(matches_by_view)} views ==="
    )
    sampled_matches = sample_matches(matches_by_view, args.max_frames, seed=args.seed)

    print(f"\n=== Creating dataset in {output_dir} ===")
    create_cleaned_dataset(sampled_matches, output_dir,
                           min_visible_ratio=args.min_visible_ratio)

    print(f"\n=== Splitting dataset ===")
    split_dataset(output_dir, args.train_ratio, args.val_ratio, args.test_ratio)


if __name__ == "__main__":
    main()
