

import argparse
import os
import random
import sys
from collections import Counter

# Locate dota_utils.py whether this script sits in scripts/, the repo root, etc.
for _cand in (os.path.dirname(os.path.abspath(__file__)),
              os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
              os.getcwd()):
    if os.path.exists(os.path.join(_cand, "dota_utils.py")):
        sys.path.insert(0, _cand)
        break

from PIL import Image
from dota_utils import DOTA_CLASSES, draw_obbs, parse_dota_label


def main():
    ap = argparse.ArgumentParser(description="Verify tiled DOTA output.")
    ap.add_argument("--tiled-dir", required=True,
                    help="folder containing images/ and labelTxt/")
    ap.add_argument("--num", type=int, default=6, help="annotated previews to render")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    img_dir = os.path.join(args.tiled_dir, "images")
    lbl_dir = os.path.join(args.tiled_dir, "labelTxt")
    for d in (img_dir, lbl_dir):
        if not os.path.isdir(d):
            sys.exit(f"Not found: {d}  (did tiling write here?)")

    imgs = sorted(f for f in os.listdir(img_dir)
                  if f.lower().endswith((".png", ".jpg", ".jpeg")))
    lbls = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]

    # --- counts ---------------------------------------------------------
    total_obj = 0
    per_class = Counter()
    empty = 0
    unknown = Counter()
    for f in imgs:
        objs = parse_dota_label(os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt"))
        if not objs:
            empty += 1
        total_obj += len(objs)
        for _, cls, _ in objs:
            per_class[cls] += 1
            if cls not in DOTA_CLASSES:
                unknown[cls] += 1

    print(f"tiled dir          : {args.tiled_dir}")
    print(f"patch images       : {len(imgs)}")
    print(f"label files        : {len(lbls)}")
    print(f"total object inst. : {total_obj}")
    print(f"avg objects/patch  : {total_obj / max(len(imgs), 1):.1f}")
    print(f"patches w/ 0 objs  : {empty}  (should be ~0; tiler skips empties)")
    print("per class          :")
    for cls, n in per_class.most_common():
        print(f"    {cls:<18} {n}")
    if unknown:
        print(f"[WARN] labels not in the 15 DOTA classes: {dict(unknown)}")

    # --- sanity flags ---------------------------------------------------
    if len(imgs) != len(lbls):
        print(f"[WARN] image/label count mismatch ({len(imgs)} vs {len(lbls)})")
    if total_obj == 0:
        print("[WARN] zero objects found - something is wrong with the labels.")

    # --- visual check ---------------------------------------------------
    out = os.path.join(args.tiled_dir, "_check_previews")
    os.makedirs(out, exist_ok=True)
    random.seed(args.seed)
    sample = random.sample(imgs, min(args.num, len(imgs)))
    for f in sample:
        stem = os.path.splitext(f)[0]
        objs = parse_dota_label(os.path.join(lbl_dir, stem + ".txt"))
        draw_obbs(Image.open(os.path.join(img_dir, f)), objs).save(
            os.path.join(out, stem + "_check.png"))
    print(f"\nWrote {len(sample)} annotated previews to {out}/")
    print("--> OPEN those and confirm the boxes hug the objects before Stage 3.")


if __name__ == "__main__":
    main()
