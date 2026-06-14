"""Tile a directory of DOTA images into fixed-size overlapping patches.

    python tile.py --images-dir data/raw/train/images \
                   --labels-dir data/raw/train/labelTxt \
                   --out-dir data/tiled/train --preview 3

The geometry lives in dota.iter_tiles; this just runs it over a directory.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from dota import draw_obbs, parse_dota_label, tile_image_to_disk


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--patch", type=int, default=1024)
    ap.add_argument("--gap", type=int, default=200, help="overlap between patches")
    ap.add_argument("--iof", type=float, default=0.7,
                    help="keep an object if this fraction of it is inside the patch")
    ap.add_argument("--preview", type=int, default=0,
                    help="render this many annotated patches afterwards")
    args = ap.parse_args()

    images = sorted(f for f in os.listdir(args.images_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg")))
    print(f"Tiling {len(images)} images (patch={args.patch}, gap={args.gap})")

    total_patches = total_objects = 0
    for fn in images:
        stem = os.path.splitext(fn)[0]
        patches, objects = tile_image_to_disk(
            os.path.join(args.images_dir, fn),
            os.path.join(args.labels_dir, stem + ".txt"),
            args.out_dir, args.patch, args.gap, args.iof)
        total_patches += patches
        total_objects += objects
    print(f"-> {total_patches} patches, {total_objects} objects in {args.out_dir}/")

    if args.preview:
        prev_dir = os.path.join(args.out_dir, "previews")
        img_dir = os.path.join(args.out_dir, "images")
        lbl_dir = os.path.join(args.out_dir, "labelTxt")
        os.makedirs(prev_dir, exist_ok=True)
        for fn in sorted(os.listdir(img_dir))[:args.preview]:
            stem = os.path.splitext(fn)[0]
            objs = parse_dota_label(os.path.join(lbl_dir, stem + ".txt"))
            draw_obbs(Image.open(os.path.join(img_dir, fn)), objs).save(
                os.path.join(prev_dir, stem + ".png"))
        print(f"-> previews in {prev_dir}/")


if __name__ == "__main__":
    main()