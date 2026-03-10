"""
make_batches.py — Download 6 YouTube swimming videos and create two annotation batches.

Each batch gets ~17 evenly-spaced frames from each video (~100 frames total per batch).
Intermediate frame directories are deleted after batch creation.

Usage:
    python data_processing/make_batches.py

Output:
    unlabeled_data/batches/batch_002/   — ~102 frames
    unlabeled_data/batches/batch_003/   — ~102 frames
"""

import shutil
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root or from data_processing/
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from download_yt import get_video_info, extract_frames, sanitize

import tempfile
import subprocess

URLS = [
    "https://www.youtube.com/watch?v=m3BsRGK9RSQ",
    "https://www.youtube.com/watch?v=wLLbvett15o",
    "https://www.youtube.com/watch?v=ljGppH9qgBA",
    "https://www.youtube.com/watch?v=_SjuNR8W3Zc",
    "https://www.youtube.com/watch?v=7UqIlG1sMNs",
    "https://www.youtube.com/watch?v=9F_qz4FZZXk",
]

FRAMES_PER_VIDEO_PER_BATCH = 17   # 17 × 6 videos = 102 per batch
N_BATCHES = 2
FPS = 1
FMT = "best[height<=1080]"
TITLE_PREFIX_LEN = 25

_ROOT = _HERE.parent
UNLABELED_DIR = _ROOT / "unlabeled_data"
BATCHES_DIR = UNLABELED_DIR / "batches"


def download_video(url: str, fmt: str, dest: Path) -> None:
    result = subprocess.run(
        ['yt-dlp', '-f', fmt, '-o', str(dest), url],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed: {result.stderr[:300]}")


def pick_frames(frame_files: list[Path], n: int) -> list[Path]:
    """Return n evenly-spaced files from a sorted list."""
    if len(frame_files) <= n:
        return frame_files
    indices = np.linspace(0, len(frame_files) - 1, n, dtype=int)
    return [frame_files[i] for i in indices]


def main() -> None:
    frames_per_video = FRAMES_PER_VIDEO_PER_BATCH * N_BATCHES  # 34 total per video

    # Create batch output directories
    batch_dirs = []
    for i in range(1, N_BATCHES + 1):
        # Find the next available batch number
        pass
    # Use batch_002 and batch_003 (batch_001 already exists)
    existing = sorted(BATCHES_DIR.glob("batch_*"))
    next_num = (int(existing[-1].name.split("_")[1]) + 1) if existing else 1
    batch_dirs = []
    for i in range(N_BATCHES):
        d = BATCHES_DIR / f"batch_{next_num + i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        batch_dirs.append(d)

    print(f"Creating {N_BATCHES} batches:")
    for d in batch_dirs:
        print(f"  {d}")
    print(f"Frames per video per batch: {FRAMES_PER_VIDEO_PER_BATCH}")
    print(f"Videos: {len(URLS)}")
    print("-" * 60)

    for url_idx, url in enumerate(URLS, 1):
        print(f"\n[{url_idx}/{len(URLS)}] {url}")

        # Get metadata
        print("  Fetching video info...")
        title, video_id = get_video_info(url)
        folder_name = sanitize(title) or sanitize(video_id)
        print(f"  Title:  {title}")
        print(f"  Folder: {folder_name}")

        frame_dir = UNLABELED_DIR / folder_name
        prefix = folder_name[:TITLE_PREFIX_LEN] + "__"

        # Download to temp file and extract frames
        tmp_file = Path(tempfile.mktemp(suffix='.mp4'))
        try:
            print("  Downloading...")
            download_video(url, FMT, tmp_file)

            print(f"  Extracting frames at {FPS} fps...")
            extract_frames(tmp_file, frame_dir, FPS)
        finally:
            if tmp_file.exists():
                tmp_file.unlink()

        # Sort extracted frames
        all_frames = sorted(frame_dir.glob("frame_*.jpg"))
        print(f"  Total frames extracted: {len(all_frames)}")

        if len(all_frames) < frames_per_video:
            print(f"  WARNING: only {len(all_frames)} frames available, need {frames_per_video}")

        # Pick frames_per_video evenly-spaced frames, split across batches
        selected = pick_frames(all_frames, frames_per_video)
        chunks = [
            selected[i * FRAMES_PER_VIDEO_PER_BATCH:(i + 1) * FRAMES_PER_VIDEO_PER_BATCH]
            for i in range(N_BATCHES)
        ]

        for batch_dir, chunk in zip(batch_dirs, chunks):
            for src in chunk:
                dst = batch_dir / (prefix + src.name)
                shutil.copy2(src, dst)
            print(f"  → {batch_dir.name}: {len(chunk)} frames")

        # Delete intermediate frame directory
        shutil.rmtree(frame_dir)
        print(f"  Deleted: {frame_dir}")

    print("\n" + "-" * 60)
    print("Done.")
    for d in batch_dirs:
        count = len(list(d.glob("*.jpg")))
        print(f"  {d.name}: {count} frames")


if __name__ == "__main__":
    main()
