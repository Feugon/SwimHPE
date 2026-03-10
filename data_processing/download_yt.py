"""
download_yt.py — Download YouTube videos and extract frames into unlabeled_data/.

Usage:
    python data_processing/download_yt.py <URL> [<URL> ...] [options]
    python data_processing/download_yt.py --url-file urls.txt --fps 2

Output structure:
    unlabeled_data/
    └── <sanitized_video_title>/
        ├── frame_0001.jpg
        ├── frame_0002.jpg
        └── ...
"""

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

DEFAULT_FPS = 1
DEFAULT_FORMAT = "best[height<=1080]"


def sanitize(name: str) -> str:
    """Convert a video title into a safe directory name."""
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'\s+', '_', name).strip('_')
    return name[:80] or 'video'


def get_video_info(url: str) -> tuple[str, str]:
    """Return (title, video_id) for a YouTube URL without downloading."""
    result = subprocess.run(
        ['yt-dlp', '--print', 'title', '--print', 'id', url],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp info failed: {result.stderr[:300]}")
    lines = result.stdout.strip().splitlines()
    if len(lines) >= 2:
        return lines[0], lines[1]
    return 'unknown', 'unknown'


def download_video(url: str, fmt: str, dest: Path) -> None:
    """Download a YouTube URL to dest using yt-dlp."""
    result = subprocess.run(
        ['yt-dlp', '-f', fmt, '-o', str(dest), url],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed: {result.stderr[:300]}")


def extract_frames(video_path: Path, out_dir: Path, fps: int) -> int:
    """Extract frames from video_path into out_dir at the given fps.

    Returns the number of frames written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            'ffmpeg', '-i', str(video_path),
            '-vf', f'fps={fps}',
            str(out_dir / 'frame_%04d.jpg'),
            '-y'
        ],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:300]}")
    return len(list(out_dir.glob('frame_*.jpg')))


def process_url(url: str, output_dir: Path, fps: int, fmt: str, keep_video: bool) -> int:
    """Download one URL, extract frames, clean up. Returns frame count."""
    print(f"\nURL: {url}")

    # Get metadata first (no download)
    print("  Fetching video info...")
    title, video_id = get_video_info(url)
    folder_name = sanitize(title) or sanitize(video_id)
    print(f"  Title:  {title}")
    print(f"  ID:     {video_id}")
    print(f"  Folder: {folder_name}")

    out_dir = output_dir / folder_name

    # Download to a temporary file
    tmp_file = Path(tempfile.mktemp(suffix='.mp4'))
    try:
        print("  Downloading...")
        download_video(url, fmt, tmp_file)

        print(f"  Extracting frames at {fps} fps...")
        n_frames = extract_frames(tmp_file, out_dir, fps)
        print(f"  Frames saved: {n_frames}  →  {out_dir}")

        if keep_video:
            dest = out_dir / f"{folder_name}.mp4"
            shutil.move(str(tmp_file), str(dest))
            print(f"  Video kept:   {dest}")

        return n_frames

    finally:
        if tmp_file.exists() and not keep_video:
            tmp_file.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download YouTube videos and extract frames into unlabeled_data/."
    )
    parser.add_argument(
        'urls', nargs='*',
        help="One or more YouTube URLs"
    )
    parser.add_argument(
        '--url-file', metavar='FILE',
        help="Text file with one YouTube URL per line (additive with positional URLs)"
    )
    parser.add_argument(
        '--output-dir', default='./unlabeled_data',
        help="Root output directory (default: ./unlabeled_data)"
    )
    parser.add_argument(
        '--fps', type=int, default=DEFAULT_FPS,
        help=f"Frames per second to extract (default: {DEFAULT_FPS})"
    )
    parser.add_argument(
        '--format', default=DEFAULT_FORMAT, dest='fmt',
        help=f"yt-dlp format string (default: {DEFAULT_FORMAT!r})"
    )
    parser.add_argument(
        '--keep-video', action='store_true',
        help="Keep downloaded video file inside the output folder"
    )
    args = parser.parse_args()

    # Collect and deduplicate URLs
    urls: list[str] = list(args.urls)
    if args.url_file:
        url_path = Path(args.url_file)
        if not url_path.exists():
            parser.error(f"--url-file not found: {url_path}")
        for line in url_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                urls.append(line)
    urls = list(dict.fromkeys(urls))  # deduplicate, preserve order

    if not urls:
        parser.error("No URLs provided. Pass URLs as positional args or via --url-file.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("-" * 50)
    print(f"URLs to process:  {len(urls)}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"FPS:              {args.fps}")
    print(f"Format:           {args.fmt}")
    print("-" * 50)

    total_frames = 0
    successes = 0

    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}]", end='')
        try:
            n = process_url(url, output_dir, args.fps, args.fmt, args.keep_video)
            total_frames += n
            successes += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            print("  Skipping to next URL.")

    print("\n" + "-" * 50)
    print(f"Done.")
    print(f"  Processed: {successes}/{len(urls)} URLs")
    print(f"  Total frames extracted: {total_frames}")
    print(f"  Output: {output_dir.resolve()}")


if __name__ == '__main__':
    main()
