"""DOTA dataset utilities: label parsing, drawing, stats, and tiling."""

import os
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon, box

# DOTA-v1.0 classes in canonical order; the list index is the class id.
DOTA_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool",
]


def parse_dota_label(path):
    """Parse a DOTA .txt label into [(corners, class_name, difficult), ...].

    Each object line is `x1 y1 x2 y2 x3 y3 x4 y4 class difficult`; the file's
    `imagesource`/`gsd` header lines are skipped.
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
                continue
            corners = list(zip(c[0::2], c[1::2]))
            difficult = int(p[9]) if len(p) > 9 else 0
            objs.append((corners, p[8], difficult))
    return objs


def class_color(name):
    """Stable RGB colour for a class, evenly spaced around the hue wheel."""
    import colorsys
    idx = DOTA_CLASSES.index(name) if name in DOTA_CLASSES else 0
    r, g, b = colorsys.hsv_to_rgb(idx / len(DOTA_CLASSES), 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_obbs(image, objs, width=3, labels=True):
    """Draw oriented boxes on an image (PIL or HxWx3 array); returns a PIL image."""
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for corners, cls, _ in objs:
        color = class_color(cls)
        draw.polygon(corners, outline=color, width=width)
        if labels:
            draw.text((corners[0][0], corners[0][1] - 18), cls, fill=color, font=font)
    return img


def longest_edge(corners):
    """Length in px of the longest edge of a quad, used as a rough object size."""
    pts = np.array(corners)
    return float(np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1).max())


def label_stats(objs, width, height):
    """Summary stats for one image's objects (counts, size distribution)."""
    sizes = np.array([longest_edge(c) for c, _, _ in objs]) if objs else np.array([])
    return {
        "width": width, "height": height, "n_objects": len(objs),
        "size_min": float(sizes.min()) if sizes.size else 0,
        "size_median": float(np.median(sizes)) if sizes.size else 0,
        "size_max": float(sizes.max()) if sizes.size else 0,
        "frac_under_40px": float((sizes < 40).mean()) if sizes.size else 0,
        "per_class": dict(Counter(c for _, c, _ in objs)),
    }


def window_starts(length, patch, stride):
    """Window start positions covering `length`, with the last clamped to the edge."""
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def refit_obb(geom):
    """Return the 4 corners of the min-area rotated rect around a clipped polygon.

    Clipping a rotated box to a window can produce >4 vertices; this re-fits it
    to a valid oriented box. Returns None for degenerate slivers.
    """
    if geom.is_empty or geom.area <= 1.0:
        return None
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    coords = np.array(geom.exterior.coords, dtype=np.float32)
    return cv2.boxPoints(cv2.minAreaRect(coords))


def iter_tiles(image, objs, patch=1024, gap=200, iof_thresh=0.7):
    """Yield (left, top, crop, patch_objs) for each non-empty patch of an image.

    Windows overlap by `gap`. An object is kept in a patch only if at least
    `iof_thresh` of its area falls inside, so boundary objects survive whole in
    one patch instead of being split across two. Object coordinates are returned
    relative to the patch.
    """
    height, width = image.shape[:2]
    stride = patch - gap
    polys = [(Polygon(c), cls, diff) for c, cls, diff in objs
             if Polygon(c).is_valid and Polygon(c).area > 0]

    for top in window_starts(height, patch, stride):
        for left in window_starts(width, patch, stride):
            window = box(left, top, left + patch, top + patch)
            crop = image[top:top + patch, left:left + patch]
            if crop.shape[:2] != (patch, patch):
                padded = np.full((patch, patch, 3), 114, np.uint8)
                padded[:crop.shape[0], :crop.shape[1]] = crop
                crop = padded

            patch_objs = []
            for poly, cls, diff in polys:
                inter = poly.intersection(window)
                if inter.is_empty or inter.area / poly.area < iof_thresh:
                    continue
                pts = refit_obb(inter)
                if pts is None:
                    continue
                pts = pts - np.array([left, top])
                patch_objs.append(([tuple(pt) for pt in pts], cls, diff))

            if patch_objs:
                yield left, top, crop, patch_objs


def tile_image_to_disk(img_path, label_path, out_dir, patch=1024, gap=200,
                       iof_thresh=0.7):
    """Tile one image to disk and return (n_patches, n_objects).

    Patch filenames encode the source stem and window offset
    (`<stem>__<patch>__<left>_<top>`) so detections can later be mapped back to
    full-image coordinates.
    """
    image = np.array(Image.open(img_path).convert("RGB"))
    objs = parse_dota_label(label_path)
    stem = os.path.splitext(os.path.basename(img_path))[0]
    img_out = os.path.join(out_dir, "images")
    lbl_out = os.path.join(out_dir, "labelTxt")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    n_patches = n_objects = 0
    for left, top, crop, patch_objs in iter_tiles(image, objs, patch, gap, iof_thresh):
        name = f"{stem}__{patch}__{left}_{top}"
        Image.fromarray(crop).save(os.path.join(img_out, name + ".png"))
        with open(os.path.join(lbl_out, name + ".txt"), "w") as f:
            for corners, cls, diff in patch_objs:
                flat = " ".join(f"{v:.1f}" for pt in corners for v in pt)
                f.write(f"{flat} {cls} {diff}\n")
        n_patches += 1
        n_objects += len(patch_objs)
    return n_patches, n_objects