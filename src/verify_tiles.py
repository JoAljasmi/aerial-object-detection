"""Sanity-check a tiled dataset: class counts and a few annotated previews.

    python verify_tiles.py --tiled-dir data/tiled/train --num 6

Always look at the previews before training; it's the cheapest way to catch
broken labels.
"""

import argparse
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from dota import DOTA_CLASSES, draw_obbs, parse_dota_label


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiled-dir", required=True,
                    help="folder containing images/ and labelTxt/")
    ap.add_argument("--num", type=int, default=6, help="annotated previews to render")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    img_dir = os.path.join(args.tiled_dir, "images")
    lbl_dir = os.path.join(args.tiled_dir, "labelTxt")
    for d in (img_dir, lbl_dir):
        if not os.path.isdir(d):
            sys.exit(f"Not found: {d}")

    images = sorted(f for f in os.listdir(img_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg")))
    n_labels = sum(1 for f in os.listdir(lbl_dir) if f.endswith(".txt"))

    per_class = Counter()
    unknown = Counter()
    empty = total = 0
    for fn in images:
        objs = parse_dota_label(os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt"))
        empty += not objs
        total += len(objs)
        for _, cls, _ in objs:
            per_class[cls] += 1
            if cls not in DOTA_CLASSES:
                unknown[cls] += 1

    print(f"patches      : {len(images)}  (labels: {n_labels})")
    print(f"objects      : {total}  (avg {total / max(len(images), 1):.1f}/patch)")
    print(f"empty patches: {empty}")
    for cls, n in per_class.most_common():
        print(f"  {cls:<18} {n}")
    if len(images) != n_labels:
        print(f"WARNING: image/label count mismatch ({len(images)} vs {n_labels})")
    if unknown:
        print(f"WARNING: unexpected classes: {dict(unknown)}")
    if not total:
        print("WARNING: no objects found")

    out = os.path.join(args.tiled_dir, "previews")
    os.makedirs(out, exist_ok=True)
    random.seed(args.seed)
    for fn in random.sample(images, min(args.num, len(images))):
        stem = os.path.splitext(fn)[0]
        objs = parse_dota_label(os.path.join(lbl_dir, stem + ".txt"))
        draw_obbs(Image.open(os.path.join(img_dir, fn)), objs).save(
            os.path.join(out, stem + ".png"))
    print(f"-> previews in {out}/")


if __name__ == "__main__":
    main()