"""
Stage 2 - Tiling, BATCH runner.

This is the thin command-line wrapper. All the actual geometry lives in
dota_utils.py (so the notebook and this script share one implementation). Use
this to tile the whole dataset; use notebooks/01_explore_data.ipynb to
understand/inspect tiling on a single image interactively.

See dota_utils.iter_tiles for the boundary handling (overlap, iof drop, OBB
re-fit, coordinate translation, offset-encoded filenames).

Usage:
    python scripts/02_tile_dota.py \
        --images-dir DOTA/train/images --labels-dir DOTA/train/labelTxt \
        --out-dir DOTA-tiled/train --patch 1024 --gap 200 --preview 3
"""

import argparse
import os
import sys

# Make the repo-root module importable when run as scripts/02_tile_dota.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from dota_utils import draw_obbs, parse_dota_label, tile_image_to_disk


def main():
    ap = argparse.ArgumentParser(description="Tile DOTA images into OBB patches.")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--patch", type=int, default=1024)
    ap.add_argument("--gap", type=int, default=200, help="overlap between patches")
    ap.add_argument("--iof", type=float, default=0.7,
                    help="keep object if this fraction is inside the patch")
    ap.add_argument("--preview", type=int, default=0,
                    help="render this many annotated patches to verify")
    args = ap.parse_args()

    imgs = sorted(f for f in os.listdir(args.images_dir)
                  if f.lower().endswith((".png", ".jpg", ".jpeg")))
    print(f"Tiling {len(imgs)} images "
          f"(patch={args.patch}, gap={args.gap}, iof>={args.iof}) ...")

    tot_p = tot_i = 0
    for fn in imgs:
        stem = os.path.splitext(fn)[0]
        p, i = tile_image_to_disk(
            os.path.join(args.images_dir, fn),
            os.path.join(args.labels_dir, stem + ".txt"),
            args.out_dir, args.patch, args.gap, args.iof)
        print(f"  {stem}: {p} patches, {i} instances")
        tot_p += p
        tot_i += i
    print(f"\nTotal: {tot_p} patches, {tot_i} object-instances -> {args.out_dir}/")

    if args.preview:
        prev = os.path.join(args.out_dir, "previews")
        os.makedirs(prev, exist_ok=True)
        img_dir = os.path.join(args.out_dir, "images")
        lbl_dir = os.path.join(args.out_dir, "labelTxt")
        for fn in sorted(os.listdir(img_dir))[:args.preview]:
            stem = os.path.splitext(fn)[0]
            objs = parse_dota_label(os.path.join(lbl_dir, stem + ".txt"))
            draw_obbs(Image.open(os.path.join(img_dir, fn)), objs).save(
                os.path.join(prev, stem + "_preview.png"))
        print(f"  wrote previews to {prev}/")


if __name__ == "__main__":
    main()