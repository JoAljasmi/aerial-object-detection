"""
Stage 3 - Convert tiled DOTA patches to YOLO-OBB format and build a
train/val/test split that is SAFE from data leakage.

Two jobs:
  1. CONVERT each label line
        x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult   (pixels, name)
     into YOLO-OBB
        class_index x1 y1 x2 y2 x3 y3 x4 y4             (normalized 0-1, index)

  2. SPLIT by SOURCE IMAGE, not by patch. Tiling made many overlapping patches
     per original image; if two patches from the same source land on opposite
     sides of the train/test line, the model sees test content while training
     and your mAP is inflated and meaningless. We group patches by their source
     stem (the 'P0042' in 'P0042__1024__0_0.png') and put each whole group in
     exactly one split.

Output (the standard Ultralytics layout, plus a dataset YAML):
    DOTA-yolo/
      images/{train,val,test}/*.png
      labels/{train,val,test}/*.txt
      dota-obb.yaml

By default it COPIES patches (non-destructive - your DOTA-tiled/ stays intact,
so you can re-split later). If disk is tight, pass --move instead (patches are
relocated; you'd re-tile to redo the split).

Usage:
    python scripts/03_to_yolo_split.py \
        --tiled-dirs DOTA-tiled/train DOTA-tiled/val \
        --out-dir DOTA-yolo \
        --train 0.8 --val 0.1 --test 0.1 --seed 42
"""

import argparse
import os
import random
import shutil
import sys
from collections import Counter, defaultdict

for _cand in (os.path.dirname(os.path.abspath(__file__)),
              os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
              os.getcwd()):
    if os.path.exists(os.path.join(_cand, "dota_utils.py")):
        sys.path.insert(0, _cand)
        break

from PIL import Image
from dota_utils import DOTA_CLASSES, parse_dota_label

CLASS_TO_IDX = {name: i for i, name in enumerate(DOTA_CLASSES)}


def source_stem(filename):
    """'P0042__1024__0_0.png' -> 'P0042' (the original image it was cut from)."""
    return os.path.basename(filename).split("__")[0]


def convert_label(objs, w, h, drop_difficult):
    """DOTA objects -> list of YOLO-OBB label lines (normalized, class index)."""
    lines, skipped = [], Counter()
    for pts, cls, diff in objs:
        if drop_difficult and diff == 1:
            continue
        if cls not in CLASS_TO_IDX:
            skipped[cls] += 1
            continue
        idx = CLASS_TO_IDX[cls]
        norm = []
        for x, y in pts:
            norm.append(min(max(x / w, 0.0), 1.0))   # normalize + clamp to [0,1]
            norm.append(min(max(y / h, 0.0), 1.0))
        lines.append(f"{idx} " + " ".join(f"{v:.6f}" for v in norm))
    return lines, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiled-dirs", nargs="+", required=True,
                    help="one or more tiled dirs, each with images/ and labelTxt/")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true",
                    help="move patches instead of copying (saves disk, destructive)")
    ap.add_argument("--drop-difficult", action="store_true",
                    help="exclude DOTA 'difficult' instances from labels")
    args = ap.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, \
        "train+val+test must sum to 1.0"

    # --- gather every patch, grouped by its SOURCE image -----------------
    # source_stem -> list of (img_path, lbl_path)
    groups = defaultdict(list)
    for tdir in args.tiled_dirs:
        idir, ldir = os.path.join(tdir, "images"), os.path.join(tdir, "labelTxt")
        if not os.path.isdir(idir):
            sys.exit(f"Not found: {idir}")
        for fn in os.listdir(idir):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                lbl = os.path.join(ldir, os.path.splitext(fn)[0] + ".txt")
                groups[source_stem(fn)].append((os.path.join(idir, fn), lbl))

    sources = sorted(groups)                       # sorted for determinism
    random.seed(args.seed)
    random.shuffle(sources)
    n = len(sources)
    n_tr = int(n * args.train)
    n_va = int(n * args.val)
    split_of = {}
    for s in sources[:n_tr]:            split_of[s] = "train"
    for s in sources[n_tr:n_tr + n_va]: split_of[s] = "val"
    for s in sources[n_tr + n_va:]:     split_of[s] = "test"

    # --- make output dirs ------------------------------------------------
    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            os.makedirs(os.path.join(args.out_dir, sub, split), exist_ok=True)

    # --- convert + place -------------------------------------------------
    place = shutil.move if args.move else shutil.copy
    counts = {s: {"src": 0, "patch": 0, "obj": 0} for s in ("train", "val", "test")}
    unknown = Counter()
    for src in sources:
        split = split_of[src]
        counts[split]["src"] += 1
        for img_path, lbl_path in groups[src]:
            with Image.open(img_path) as im:
                w, h = im.size
            objs = parse_dota_label(lbl_path)
            lines, skipped = convert_label(objs, w, h, args.drop_difficult)
            unknown.update(skipped)
            stem = os.path.splitext(os.path.basename(img_path))[0]
            with open(os.path.join(args.out_dir, "labels", split, stem + ".txt"),
                      "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            place(img_path, os.path.join(args.out_dir, "images", split,
                                         os.path.basename(img_path)))
            counts[split]["patch"] += 1
            counts[split]["obj"] += len(lines)

    # --- dataset YAML ----------------------------------------------------
    yaml_path = os.path.join(args.out_dir, "dota-obb.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(args.out_dir)}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n\n")
        f.write("names:\n")
        for i, name in enumerate(DOTA_CLASSES):
            f.write(f"  {i}: {name}\n")

    # --- report ----------------------------------------------------------
    print(f"source images: {n}  ->  "
          f"train {counts['train']['src']} / val {counts['val']['src']} / "
          f"test {counts['test']['src']}")
    for split in ("train", "val", "test"):
        c = counts[split]
        print(f"  {split:<5} : {c['patch']:>6} patches, {c['obj']:>7} objects")
    if unknown:
        print(f"[WARN] skipped unknown classes: {dict(unknown)}")
    print(f"\nwrote {yaml_path}")
    print("Leakage check: split is by source image, so no source appears in two "
          "splits (guaranteed by construction).")


if __name__ == "__main__":
    main()
