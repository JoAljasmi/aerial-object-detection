"""
Stage 1 - See the data.

Before touching a model, you should understand exactly what DOTA gives you.
This script parses DOTA's annotation format, draws the oriented bounding boxes
on the image, and prints statistics that make the *tiling* problem (Stage 2)
concrete: huge images, lots of tiny objects.

--------------------------------------------------------------------------
DOTA-v1.0 annotation format
--------------------------------------------------------------------------
Each image `P0001.png` has a label file `P0001.txt`. The first two lines are
metadata (present in the original release, sometimes absent in re-packaged
versions, so we skip them defensively):

    imagesource:GoogleEarth
    gsd:0.146                      # ground sample distance, metres/pixel

Every following line is ONE object, an oriented box given as 4 corner points
in clockwise order, plus a class name and a 'difficult' flag (0/1):

    x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult

Key point: the box is a *quadrilateral* (8 numbers), NOT the (x, y, w, h)
you see in COCO. That is because a plane or ship photographed from above sits
at an arbitrary angle, and an axis-aligned box would be mostly empty space /
would overlap its neighbours. This is the whole reason we use an OBB model.

Usage:
    # one image + its label
    python scripts/01_visualize_dota.py --image path/to/P0001.png \
                                        --label path/to/P0001.txt \
                                        --out annotated.png

    # or scan a folder pair and render the first N
    python scripts/01_visualize_dota.py --images-dir DOTA/train/images \
                                        --labels-dir DOTA/train/labelTxt \
                                        --out-dir previews --num 5
"""

import argparse
import colorsys
import os
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# The 15 DOTA-v1.0 classes, in the canonical order Ultralytics uses (0..14).
DOTA_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool",
]


def class_color(name: str) -> tuple:
    """Deterministic distinct color per class (so the same class is always
    the same color across images)."""
    idx = DOTA_CLASSES.index(name) if name in DOTA_CLASSES else 0
    hue = idx / max(len(DOTA_CLASSES), 1)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def parse_dota_label(path: str) -> list:
    """Return a list of dicts: {'points': [(x,y)*4], 'cls': str, 'difficult': int}."""
    objects = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            # Skip metadata lines ('imagesource:...', 'gsd:...') and blanks.
            if len(parts) < 9:
                continue
            try:
                coords = [float(v) for v in parts[:8]]
            except ValueError:
                continue  # header line, not an object
            points = list(zip(coords[0::2], coords[1::2]))  # [(x1,y1),...,(x4,y4)]
            objects.append({
                "points": points,
                "cls": parts[8],
                "difficult": int(parts[9]) if len(parts) > 9 else 0,
            })
    return objects


def obb_longest_side(points: list) -> float:
    """Length of the longest edge of the quad - a rough 'object size' in pixels."""
    pts = np.array(points)
    edges = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
    return float(edges.max())


def draw(image_path: str, label_path: str, out_path: str) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    objects = parse_dota_label(label_path)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for obj in objects:
        color = class_color(obj["cls"])
        # An oriented box is just a closed polygon through its 4 corners.
        draw.polygon(obj["points"], outline=color, width=3)
        x, y = obj["points"][0]
        draw.text((x, y - 18), obj["cls"], fill=color, font=font)

    img.save(out_path)

    # --- Statistics: this is the part that motivates tiling -------------
    w, h = img.size
    counts = Counter(o["cls"] for o in objects)
    sizes = [obj_long for obj in objects
             if (obj_long := obb_longest_side(obj["points"]))]

    print(f"\n{os.path.basename(image_path)}")
    print(f"  image size      : {w} x {h} px")
    print(f"  objects         : {len(objects)}")
    if sizes:
        arr = np.array(sizes)
        tiny = int((arr < 40).sum())
        print(f"  object size (longest edge, px): "
              f"min {arr.min():.0f} / median {np.median(arr):.0f} / max {arr.max():.0f}")
        print(f"  objects < 40 px : {tiny} "
              f"({100 * tiny / len(sizes):.0f}%)  <-- why we must tile, not downsize")
    print("  per class       : " +
          ", ".join(f"{k}:{v}" for k, v in counts.most_common()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize DOTA oriented bounding boxes.")
    ap.add_argument("--image")
    ap.add_argument("--label")
    ap.add_argument("--out", default="annotated.png")
    ap.add_argument("--images-dir")
    ap.add_argument("--labels-dir")
    ap.add_argument("--out-dir", default="previews")
    ap.add_argument("--num", type=int, default=5)
    args = ap.parse_args()

    if args.image and args.label:
        draw(args.image, args.label, args.out)
        print(f"\nWrote {args.out}")
        return

    if args.images_dir and args.labels_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        imgs = sorted(f for f in os.listdir(args.images_dir)
                      if f.lower().endswith((".png", ".jpg", ".jpeg")))[:args.num]
        for fname in imgs:
            stem = os.path.splitext(fname)[0]
            label = os.path.join(args.labels_dir, stem + ".txt")
            if not os.path.exists(label):
                print(f"  (no label for {fname}, skipping)")
                continue
            out = os.path.join(args.out_dir, stem + "_annotated.png")
            draw(os.path.join(args.images_dir, fname), label, out)
        print(f"\nWrote previews to {args.out_dir}/")
        return

    ap.error("Provide either (--image and --label) or (--images-dir and --labels-dir).")


if __name__ == "__main__":
    main()