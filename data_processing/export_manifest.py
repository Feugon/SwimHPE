"""
export_manifest.py — Export hand-labeled YOLO annotations into a lightweight JSON manifest.

The manifest stores only coordinates (no images), keyed by YouTube video ID + frame number,
so that anyone can reconstruct the dataset from the manifest alone.

Reads from labeled_data/labels/ (YOLO .txt format) which contains all 420 hand-labeled frames.

Usage:
    python data_processing/export_manifest.py
    python data_processing/export_manifest.py --sources annotations/youtube_sources.json \
                                              --labels-dir labeled_data/labels \
                                              --output annotations/manifest.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from download_yt import get_video_info, sanitize

# Batch-file title prefix length (must match make_batches.py)
TITLE_PREFIX_LEN = 25

# COCO17 keypoint index → joint name
COCO17_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

# Hard-coded prefix → youtube_id for files that lost their source mapping
# (labeled before the batch system was set up)
HARDCODED_PREFIX_MAP = {
    'hl_train': 'omhYbRTOQEk',  # 2_beat_kick_freestyle_step_by_step_advanced_swimming
    'hl_val': 'HNav_fzh0vI',    # 25m_underwater_underwater_swimming
}


def parse_frame_number(filename: str) -> int | None:
    """Extract the frame number from a filename like 'frame_0004.txt' or 'prefix__frame_0004.txt'."""
    m = re.search(r'frame_(\d+)', filename)
    return int(m.group(1)) if m else None


def parse_file_prefix(stem: str) -> str | None:
    """Extract the prefix from 'prefix__frame_NNNN' or return None for unprefixed 'frame_NNNN'."""
    parts = stem.split('__frame_')
    if len(parts) == 2:
        return parts[0]
    return None


def parse_yolo_label(txt_path: Path, img_width: int, img_height: int) -> list[dict]:
    """Parse a YOLO pose label file into manifest person dicts.

    Each line: 0 x_c y_c w h kp1_x kp1_y kp1_v ... kp17_x kp17_y kp17_v
    Coordinates are normalized [0,1]; we convert back to pixels.
    """
    text = txt_path.read_text().strip()
    if not text:
        return []

    persons = []
    for line in text.splitlines():
        vals = line.split()
        if len(vals) < 5 + 17 * 3:
            continue

        # Bounding box (normalized → pixels)
        x_c, y_c, w, h = float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4])
        x1 = round((x_c - w / 2) * img_width, 2)
        y1 = round((y_c - h / 2) * img_height, 2)
        x2 = round((x_c + w / 2) * img_width, 2)
        y2 = round((y_c + h / 2) * img_height, 2)

        # Keypoints (normalized → pixels)
        keypoints = []
        for i in range(17):
            kx = float(vals[5 + i * 3])
            ky = float(vals[5 + i * 3 + 1])
            kv = float(vals[5 + i * 3 + 2])
            if kv == 0.0:
                continue  # not annotated
            keypoints.append({
                'joint': COCO17_NAMES[i],
                'x': round(kx * img_width, 2),
                'y': round(ky * img_height, 2),
                'difficult': kv == 1.0,  # 1.0 = uncertain, 2.0 = visible
            })

        if keypoints:
            persons.append({
                'bbox_pixels': [x1, y1, x2, y2],
                'keypoints': keypoints,
            })

    return persons


def resolve_sources(sources_path: Path) -> tuple[dict, dict]:
    """Query yt-dlp for each URL and build lookup tables.

    Returns (by_dir, by_prefix) where:
        by_dir:    full sanitized dir name → {youtube_id, url, dir_name}
        by_prefix: 25-char prefix → same dict

    Each entry in youtube_sources.json can be either:
        - a plain URL string
        - an object {"url": "...", "dir_name": "..."} for videos whose title changed
    """
    with open(sources_path) as f:
        entries = json.load(f)

    by_dir: dict[str, dict] = {}
    by_prefix: dict[str, dict] = {}

    for entry in entries:
        if isinstance(entry, str):
            url = entry
            dir_name_override = None
        else:
            url = entry['url']
            dir_name_override = entry.get('dir_name')

        try:
            title, video_id = get_video_info(url)
        except RuntimeError as e:
            print(f"  WARNING: could not resolve {url}: {e}")
            continue

        dir_name = dir_name_override or sanitize(title) or sanitize(video_id)
        prefix = dir_name[:TITLE_PREFIX_LEN]

        info = {'youtube_id': video_id, 'url': url, 'dir_name': dir_name}
        by_dir[dir_name] = info
        by_prefix[prefix] = info
        print(f"  {video_id}  {dir_name}")

    return by_dir, by_prefix


def collect_from_yolo_labels(labels_dir: Path, by_dir: dict, by_prefix: dict,
                             img_width: int = 640, img_height: int = 360) -> list[dict]:
    """Read YOLO .txt labels from a flat directory and build manifest entries."""
    entries = []
    skipped_dup = 0
    skipped_unknown = 0

    # Track (youtube_id, frame_num) to deduplicate
    seen: set[tuple[str, int]] = set()

    for txt_path in sorted(labels_dir.glob('*.txt')):
        frame_num = parse_frame_number(txt_path.name)
        if frame_num is None:
            continue

        prefix = parse_file_prefix(txt_path.stem)

        # Resolve youtube_id and dir_name
        info = None
        if prefix and prefix in HARDCODED_PREFIX_MAP:
            vid_id = HARDCODED_PREFIX_MAP[prefix]
            # Find dir_name from by_dir (reverse lookup by youtube_id)
            for d_info in by_dir.values():
                if d_info['youtube_id'] == vid_id:
                    info = d_info
                    break
            if not info:
                info = {'youtube_id': vid_id, 'dir_name': prefix}
        elif prefix:
            info = by_prefix.get(prefix[:TITLE_PREFIX_LEN])
        else:
            # Unprefixed frame_NNNN — these are duplicates of hl_val
            vid_id = HARDCODED_PREFIX_MAP.get('hl_val')
            if vid_id:
                key = (vid_id, frame_num)
                if key in seen:
                    skipped_dup += 1
                    continue
                for d_info in by_dir.values():
                    if d_info['youtube_id'] == vid_id:
                        info = d_info
                        break
                if not info:
                    info = {'youtube_id': vid_id, 'dir_name': 'hl_val'}

        if not info:
            skipped_unknown += 1
            print(f"  WARNING: no source for '{txt_path.name}' — skipping")
            continue

        key = (info['youtube_id'], frame_num)
        if key in seen:
            skipped_dup += 1
            continue
        seen.add(key)

        persons = parse_yolo_label(txt_path, img_width, img_height)
        if not persons:
            continue

        entries.append({
            'youtube_id': info['youtube_id'],
            'dir_name': info['dir_name'],
            'frame_number': frame_num,
            'fps': 1,
            'img_width': img_width,
            'img_height': img_height,
            'persons': persons,
        })

    if skipped_dup:
        print(f"  Skipped {skipped_dup} duplicates")
    if skipped_unknown:
        print(f"  Skipped {skipped_unknown} unknown sources")

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export hand-labeled YOLO annotations to a lightweight JSON manifest."
    )
    parser.add_argument(
        '--sources', default='annotations/youtube_sources.json',
        help='Path to youtube_sources.json (default: annotations/youtube_sources.json)'
    )
    parser.add_argument(
        '--labels-dir', default='labeled_data/labels',
        help='Directory with YOLO .txt label files (default: labeled_data/labels)'
    )
    parser.add_argument(
        '--output', default='annotations/manifest.json',
        help='Output manifest path (default: annotations/manifest.json)'
    )
    parser.add_argument(
        '--img-width', type=int, default=640,
        help='Image width in pixels (default: 640)'
    )
    parser.add_argument(
        '--img-height', type=int, default=360,
        help='Image height in pixels (default: 360)'
    )
    args = parser.parse_args()

    sources_path = Path(args.sources)
    labels_dir = Path(args.labels_dir)
    output_path = Path(args.output)

    if not sources_path.exists():
        print(f"ERROR: sources file not found: {sources_path}")
        sys.exit(1)
    if not labels_dir.exists():
        print(f"ERROR: labels directory not found: {labels_dir}")
        sys.exit(1)

    print("Resolving YouTube sources...")
    by_dir, by_prefix = resolve_sources(sources_path)
    print(f"  Resolved {len(by_dir)} videos\n")

    print(f"Collecting from {labels_dir}...")
    entries = collect_from_yolo_labels(labels_dir, by_dir, by_prefix,
                                      args.img_width, args.img_height)
    print(f"  Collected {len(entries)} unique annotated frames\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"Manifest written to {output_path}")


if __name__ == '__main__':
    main()
