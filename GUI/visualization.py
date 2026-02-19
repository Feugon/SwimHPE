#!/usr/bin/env python3
import os
import sys
import json
import argparse
from typing import Dict, Tuple, Optional, List

import cv2
import matplotlib.pyplot as plt

# Defaults per user request
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_JSON = os.path.join(_PROJECT_ROOT, "cycle", "angles_json", "cycle1.json")
DEFAULT_IMAGES_DIR = os.path.join(_PROJECT_ROOT, "data_processing", "cleaned_Freestyle_part2", "images", "train")

# Import decoder
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cycle'))
from pose_decode import decode_pose, load_label_keypoints, compute_angles_from_kp  # type: ignore


def load_cycle_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def frame_to_image_path(images_dir: str, entry: Dict) -> Optional[str]:
    base: Optional[str] = None
    fval = entry.get("file")
    if isinstance(fval, str) and len(fval) > 0:
        base = os.path.splitext(os.path.basename(fval))[0]
    if not base:
        base = entry.get("frame")
    if not base:
        return None
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        cand = os.path.join(images_dir, base + ext)
        if os.path.isfile(cand):
            return cand
    return None


def pretty_angles(label: str, ang: Dict[str,float]) -> str:
    import math
    keys = [
        'torso_tilt','shoulder_axis',
        'L_shoulder_to_elbow','L_elbow_flexion','L_hip_flexion','L_knee_flexion',
        'R_shoulder_to_elbow','R_elbow_flexion','R_hip_flexion','R_knee_flexion'
    ]
    s = [f"{label}:"]
    for k in keys:
        if k in ang:
            s.append(f"  {k:>20}: {ang[k]:+.3f} rad ({__import__('math').degrees(ang[k]):+.1f} deg)")
    return "\n".join(s)


def draw_pose(ax, pts: List[Tuple[float, float]], color='tab:blue', joints_color='tab:red'):
    ax.clear()
    bones = [
        (5,6), (11,12), (5,11), (6,12),
        (5,7), (7,9), (6,8), (8,10),
        (11,13), (13,15), (12,14), (14,16)
    ]
    for i,j in bones:
        x1,y1 = pts[i]; x2,y2 = pts[j]
        ax.plot([x1,x2],[y1,y2], '-', color=color, linewidth=2)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.scatter(xs, ys, s=12, color=joints_color)
    ax.set_aspect('equal'); ax.axis('off')


def show_example(entry: Dict, images_dir: str, fig: Optional[plt.Figure] = None, debug: bool=False, align_to_image: bool=True, overlay_label: bool=False):
    if fig is None:
        fig = plt.figure(figsize=(10,5))
    fig.clf()
    ax_img = fig.add_subplot(1,2,1)
    ax_pose = fig.add_subplot(1,2,2)

    img_path = frame_to_image_path(images_dir, entry)
    img = None
    if img_path and os.path.isfile(img_path):
        img = cv2.imread(img_path)
        if img is None:
            ax_img.text(0.5,0.5, 'Failed to read image', ha='center', va='center'); ax_img.axis('off')
        else:
            ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax_img.set_title(os.path.basename(img_path)); ax_img.axis('off')
    else:
        ax_img.text(0.5,0.5, 'Image not found', ha='center', va='center'); ax_img.axis('off')

    poses = entry.get('poses') or []
    if not poses:
        ax_pose.text(0.5,0.5, 'No pose in JSON', ha='center', va='center'); ax_pose.axis('off')
    else:
        angles = poses[0].get('angles', {}) or {}
        pts, kp_pix, lab_pts_canon = decode_pose(angles, entry, images_dir, align_to_image=align_to_image, use_label_lengths=True)
        draw_pose(ax_pose, pts)
        ax_pose.set_title('Reconstructed pose (angles only)')
        if overlay_label and kp_pix is not None:
            # Overlay pixel label skeleton (not canonical)
            lab_pts = [( (kp_pix[5][0]+kp_pix[6][0])/2.0, (kp_pix[5][1]+kp_pix[6][1])/2.0 )]*5 + \
                      [ (kp_pix[5][0],kp_pix[5][1]), (kp_pix[6][0],kp_pix[6][1]), (kp_pix[7][0],kp_pix[7][1]), (kp_pix[8][0],kp_pix[8][1]), (kp_pix[9][0],kp_pix[9][1]), (kp_pix[10][0],kp_pix[10][1]),
                        (kp_pix[11][0],kp_pix[11][1]), (kp_pix[12][0],kp_pix[12][1]), (kp_pix[13][0],kp_pix[13][1]), (kp_pix[14][0],kp_pix[14][1]), (kp_pix[15][0],kp_pix[15][1]), (kp_pix[16][0],kp_pix[16][1]) ]
            for i,j in [(5,6),(11,12),(5,11),(6,12),(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16)]:
                x1,y1 = lab_pts[i]; x2,y2 = lab_pts[j]
                ax_pose.plot([x1,x2],[y1,y2], '-', color='gray', linewidth=2, alpha=0.6)
            ax_pose.scatter([p[0] for p in lab_pts],[p[1] for p in lab_pts], s=10, color='black', alpha=0.6)
        if debug:
            print("=== DEBUG ===")
            print(pretty_angles("JSON angles", {k: float(v) for k,v in angles.items()}))
            if kp_pix is not None:
                comp = compute_angles_from_kp(kp_pix)
                print(pretty_angles("Label-derived (pixel)", comp))

    fig.tight_layout(); plt.show(block=True)


def main():
    parser = argparse.ArgumentParser(description='Visualize anchor poses from cycles JSON with images side-by-side.')
    parser.add_argument('--json', default=DEFAULT_JSON, help='Path to angles JSON')
    parser.add_argument('--images', default=DEFAULT_IMAGES_DIR, help='Path to images directory')
    parser.add_argument('--index', type=int, default=0, help='Frame index to show')
    parser.add_argument('--all', action='store_true', help='Iterate all frames')
    parser.add_argument('--debug', action='store_true', help='Print JSON angles and label-derived angles')
    parser.add_argument('--overlay-label', action='store_true', help='Overlay label skeleton on pose')
    parser.add_argument('--no-align', action='store_true', help='Do not align to image orientation')
    args = parser.parse_args()

    data = load_cycle_json(args.json)
    frames = data.get('frames') or []
    if not frames:
        print('No frames found in JSON'); return

    if args.all:
        fig = plt.figure(figsize=(10,5))
        for i, entry in enumerate(frames):
            print(f"Showing {i+1}/{len(frames)}: {entry.get('file') or entry.get('frame')}")
            show_example(entry, args.images, fig=fig, debug=args.debug, align_to_image=(not args.no_align), overlay_label=args.overlay_label)
    else:
        idx = max(0, min(len(frames)-1, int(args.index)))
        entry = frames[idx]
        show_example(entry, args.images, debug=args.debug, align_to_image=(not args.no_align), overlay_label=args.overlay_label)


if __name__ == '__main__':
    main()
