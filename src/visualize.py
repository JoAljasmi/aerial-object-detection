"""Render a DOTA image with its oriented-box labels and print size stats.

    python visualize.py --image img.png --label img.txt --out annotated.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from dota import draw_obbs, label_stats, parse_dota_label


def visualize(image_path, label_path, out_path):
    objs = parse_dota_label(label_path)
    image = Image.open(image_path).convert("RGB")
    draw_obbs(image, objs).save(out_path)

    s = label_stats(objs, *image.size)
    print(f"{os.path.basename(image_path)}: {s['width']}x{s['height']} px, "
          f"{s['n_objects']} objects")
    print(f"  size px  min {s['size_min']:.0f} / median {s['size_median']:.0f} "
          f"/ max {s['size_max']:.0f}")
    print(f"  under 40 px: {100 * s['frac_under_40px']:.0f}%")
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="annotated.png")
    args = ap.parse_args()
    visualize(args.image, args.label, args.out)


if __name__ == "__main__":
    main()