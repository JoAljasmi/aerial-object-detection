"""Convert tiled DOTA patches to YOLO-OBB format and split into train/val/test.

Labels go from `x1 y1 .. x4 y4 class difficult` (pixels, names) to
`class_id x1 y1 .. x4 y4` (normalized). The split is by SOURCE image, not by
patch: overlapping patches from one original image would otherwise leak across
the train/test boundary and inflate the score. Output is the standard
Ultralytics layout plus a dataset YAML.

    python prepare_dataset.py --tiled-dirs data/tiled/train data/tiled/val \
                              --out-dir data/yolo --seed 42
"""

import argparse
import os
import random
import shutil
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from dota import DOTA_CLASSES, parse_dota_label

CLASS_TO_ID = {name: i for i, name in enumerate(DOTA_CLASSES)}


def source_stem(filename):
    """'P0042__1024__0_0.png' -> 'P0042', the image the patch was cut from."""
    return os.path.basename(filename).split("__")[0]


def to_yolo_obb(objs, width, height, drop_difficult):
    """Convert parsed objects to normalized YOLO-OBB label lines."""
    lines, skipped = [], Counter()
    for corners, cls, diff in objs:
        if drop_difficult and diff == 1:
            continue
        if cls not in CLASS_TO_ID:
            skipped[cls] += 1
            continue
        coords = []
        for x, y in corners:
            coords.append(min(max(x / width, 0.0), 1.0))
            coords.append(min(max(y / height, 0.0), 1.0))
        lines.append(f"{CLASS_TO_ID[cls]} " + " ".join(f"{v:.6f}" for v in coords))
    return lines, skipped


def assign_splits(sources, train, val, seed):
    """Map each source image to 'train'/'val'/'test' by shuffled ratio."""
    sources = sorted(sources)
    random.seed(seed)
    random.shuffle(sources)
    n_train = int(len(sources) * train)
    n_val = int(len(sources) * val)
    split = {}
    for s in sources[:n_train]:
        split[s] = "train"
    for s in sources[n_train:n_train + n_val]:
        split[s] = "val"
    for s in sources[n_train + n_val:]:
        split[s] = "test"
    return split


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiled-dirs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true",
                    help="move patches instead of copying (saves disk)")
    ap.add_argument("--drop-difficult", action="store_true")
    args = ap.parse_args()
    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "ratios must sum to 1"

    # Group every patch by the source image it came from.
    groups = defaultdict(list)
    for tiled in args.tiled_dirs:
        img_dir = os.path.join(tiled, "images")
        lbl_dir = os.path.join(tiled, "labelTxt")
        if not os.path.isdir(img_dir):
            sys.exit(f"Not found: {img_dir}")
        for fn in os.listdir(img_dir):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                label = os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt")
                groups[source_stem(fn)].append((os.path.join(img_dir, fn), label))

    split_of = assign_splits(groups.keys(), args.train, args.val, args.seed)

    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            os.makedirs(os.path.join(args.out_dir, kind, split), exist_ok=True)

    place = shutil.move if args.move else shutil.copy
    counts = {s: {"sources": 0, "patches": 0, "objects": 0}
              for s in ("train", "val", "test")}
    unknown = Counter()
    for src, patches in groups.items():
        split = split_of[src]
        counts[split]["sources"] += 1
        for img_path, lbl_path in patches:
            with Image.open(img_path) as im:
                width, height = im.size
            lines, skipped = to_yolo_obb(
                parse_dota_label(lbl_path), width, height, args.drop_difficult)
            unknown.update(skipped)
            stem = os.path.splitext(os.path.basename(img_path))[0]
            with open(os.path.join(args.out_dir, "labels", split, stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            place(img_path, os.path.join(args.out_dir, "images", split,
                                         os.path.basename(img_path)))
            counts[split]["patches"] += 1
            counts[split]["objects"] += len(lines)

    yaml_path = os.path.join(args.out_dir, "dota-obb.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(args.out_dir)}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n\nnames:\n")
        for i, name in enumerate(DOTA_CLASSES):
            f.write(f"  {i}: {name}\n")

    for split in ("train", "val", "test"):
        c = counts[split]
        print(f"{split:<5}: {c['sources']:>4} sources, {c['patches']:>6} patches, "
              f"{c['objects']:>7} objects")
    if unknown:
        print(f"skipped unknown classes: {dict(unknown)}")
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()