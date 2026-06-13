"""
dota_utils.py - shared logic imported by BOTH the exploration notebooks and the
batch scripts. One source of truth: fix a bug here, both fix.

Grouped by concern:
  - I/O + classes : parse_dota_label, DOTA_CLASSES
  - drawing       : class_color, draw_obbs           (for inline viz / previews)
  - stats         : obb_longest_side, image_label_stats
  - tiling core   : window_starts, refit_obb, iter_tiles
"""

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon, box

# DOTA-v1.0 classes in canonical order (index 0..14, matches Ultralytics YAML).
DOTA_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool",
]


# --------------------------------------------------------------------------- #
# I/O                                                                         #
# --------------------------------------------------------------------------- #
def parse_dota_label(path):
    """Return [(points[(x,y)*4], cls:str, difficult:int), ...].

    Skips DOTA's metadata header lines ('imagesource:...', 'gsd:...') and any
    malformed/blank lines defensively.
    """
    objs = []
    if not os.path.exists(path):
        return objs
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 9:
                continue
            try:
                c = [float(v) for v in p[:8]]
            except ValueError:
                continue  # header line, not coordinates
            objs.append((list(zip(c[0::2], c[1::2])), p[8],
                         int(p[9]) if len(p) > 9 else 0))
    return objs


# --------------------------------------------------------------------------- #
# Drawing                                                                     #
# --------------------------------------------------------------------------- #
def class_color(name):
    """Deterministic distinct RGB per class (same class -> same color always)."""
    import colorsys
    idx = DOTA_CLASSES.index(name) if name in DOTA_CLASSES else 0
    r, g, b = colorsys.hsv_to_rgb(idx / len(DOTA_CLASSES), 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_obbs(image, objs, width=3, labels=True):
    """Draw oriented boxes on a PIL image or HxWx3 array. Returns a PIL.Image
    (so notebooks can `display()` it and scripts can `.save()` it)."""
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    img = img.convert("RGB")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for pts, cls, _ in objs:
        color = class_color(cls)
        d.polygon(pts, outline=color, width=width)
        if labels:
            d.text((pts[0][0], pts[0][1] - 18), cls, fill=color, font=font)
    return img


# --------------------------------------------------------------------------- #
# Stats                                                                       #
# --------------------------------------------------------------------------- #
def obb_longest_side(points):
    """Longest edge of the quad - a rough object 'size' in pixels."""
    pts = np.array(points)
    return float(np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1).max())


def image_label_stats(objs, w, h):
    """Dict of summary stats - the numbers that motivate tiling."""
    from collections import Counter
    sizes = np.array([obb_longest_side(p) for p, _, _ in objs]) if objs else np.array([])
    return {
        "width": w, "height": h, "n_objects": len(objs),
        "size_min": float(sizes.min()) if sizes.size else 0,
        "size_median": float(np.median(sizes)) if sizes.size else 0,
        "size_max": float(sizes.max()) if sizes.size else 0,
        "frac_under_40px": float((sizes < 40).mean()) if sizes.size else 0,
        "per_class": dict(Counter(c for _, c, _ in objs)),
    }


# --------------------------------------------------------------------------- #
# Tiling core                                                                 #
# --------------------------------------------------------------------------- #
def window_starts(length, patch, stride):
    """Top-left positions covering the axis; last window clamped to the edge."""
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def refit_obb(geom):
    """Min-area rotated rect (4 corner pts) of a shapely intersection, or None.
    For an object fully inside a window this returns the original box."""
    if geom.is_empty or geom.area <= 1.0:
        return None
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    coords = np.array(geom.exterior.coords, dtype=np.float32)
    return cv2.boxPoints(cv2.minAreaRect(coords))


def iter_tiles(img_array, objs, patch=1024, gap=200, iof_thresh=0.7):
    """Yield (left, top, crop_array, patch_objs) for each NON-EMPTY patch.

    patch_objs are in the SAME (points, cls, difficult) shape as parse_dota_label,
    but coordinates are relative to the patch. This generator is the single
    implementation used by both the notebook (display in memory) and the batch
    script (write to disk) - neither duplicates the geometry.
    """
    H, W = img_array.shape[:2]
    stride = patch - gap
    polys = [(Polygon(p), c, d) for p, c, d in objs
             if Polygon(p).is_valid and Polygon(p).area > 0]

    for top in window_starts(H, patch, stride):
        for left in window_starts(W, patch, stride):
            win = box(left, top, left + patch, top + patch)
            crop = img_array[top:top + patch, left:left + patch]
            if crop.shape[:2] != (patch, patch):           # pad edge crops
                padded = np.full((patch, patch, 3), 114, np.uint8)
                padded[:crop.shape[0], :crop.shape[1]] = crop
                crop = padded

            patch_objs = []
            for poly, cls, diff in polys:
                inter = poly.intersection(win)
                if inter.is_empty or inter.area / poly.area < iof_thresh:
                    continue
                pts = refit_obb(inter)
                if pts is None:
                    continue
                pts = pts - np.array([left, top])          # -> patch coords
                patch_objs.append(([tuple(pt) for pt in pts], cls, diff))

            if patch_objs:                                  # skip empty patches
                yield left, top, crop, patch_objs


def tile_image_to_disk(img_path, label_path, out_dir, patch=1024, gap=200,
                       iof_thresh=0.7):
    """Batch helper: write all non-empty patches of one image to disk. Returns
    (n_patches, n_instances). Filenames encode the offset for Stage 6 stitching."""
    img = np.array(Image.open(img_path).convert("RGB"))
    objs = parse_dota_label(label_path)
    stem = os.path.splitext(os.path.basename(img_path))[0]
    img_out = os.path.join(out_dir, "images")
    lbl_out = os.path.join(out_dir, "labelTxt")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    n_patches = n_inst = 0
    for left, top, crop, patch_objs in iter_tiles(img, objs, patch, gap, iof_thresh):
        name = f"{stem}__{patch}__{left}_{top}"
        Image.fromarray(crop).save(os.path.join(img_out, name + ".png"))
        with open(os.path.join(lbl_out, name + ".txt"), "w") as f:
            for pts, cls, diff in patch_objs:
                flat = " ".join(f"{v:.1f}" for pt in pts for v in pt)
                f.write(f"{flat} {cls} {diff}\n")
        n_patches += 1
        n_inst += len(patch_objs)
    return n_patches, n_inst