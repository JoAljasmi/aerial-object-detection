"""Run the trained detector on full-size images by tiling, then stitch the patch
detections back into one set of boxes on the full image.

    python detect.py --weights runs/obb/dota_obb/weights/best.pt \
                     --images-dir data/raw/val/images --out-dir predictions

Each image is sliced into overlapping patches (same windowing as training), the
model runs on each patch, detections are shifted into full-image coordinates,
and rotated NMS removes the duplicates the overlap creates.
"""

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from dota import DOTA_CLASSES, draw_obbs, window_starts


def rotated_nms(polys, scores, classes, iou_thresh):
    """Greedy per-class NMS on oriented boxes, using shapely for rotated IoU.

    An axis-aligned bounding-box test pre-filters obvious non-overlaps so the
    expensive polygon intersection only runs on real candidates.
    """
    shapes = [Polygon(p) for p in polys]
    bounds = [s.bounds for s in shapes]            # (minx, miny, maxx, maxy)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    removed = [False] * len(order)
    keep = []

    for pos_i, i in enumerate(order):
        if removed[pos_i]:
            continue
        keep.append(i)
        bi, si, ai = bounds[i], shapes[i], shapes[i].area
        for pos_j in range(pos_i + 1, len(order)):
            if removed[pos_j]:
                continue
            j = order[pos_j]
            if classes[j] != classes[i]:
                continue
            bj = bounds[j]
            if bi[2] < bj[0] or bj[2] < bi[0] or bi[3] < bj[1] or bj[3] < bi[1]:
                continue                            # bounding boxes don't touch
            inter = si.intersection(shapes[j]).area
            if inter <= 0:
                continue
            if inter / (ai + shapes[j].area - inter) > iou_thresh:
                removed[pos_j] = True
    return keep


def detect_full_image(model, image, patch, gap, conf, iou):
    """Tile, predict per patch, shift to full-image coords, and merge with NMS.

    Returns [(corners, class_name, confidence), ...] in full-image coordinates.
    """
    height, width = image.shape[:2]
    stride = patch - gap
    polys, confs, classes = [], [], []

    for top in window_starts(height, patch, stride):
        for left in window_starts(width, patch, stride):
            crop = image[top:top + patch, left:left + patch]
            if crop.shape[:2] != (patch, patch):
                padded = np.full((patch, patch, 3), 114, np.uint8)
                padded[:crop.shape[0], :crop.shape[1]] = crop
                crop = padded

            # pass a PIL image so Ultralytics treats it as RGB (numpy = BGR)
            result = model.predict(Image.fromarray(crop), conf=conf, verbose=False)[0]
            if result.obb is None or len(result.obb) == 0:
                continue

            poly = result.obb.xyxyxyxy.cpu().numpy()   # (n, 4, 2)
            poly[..., 0] += left                       # shift into full-image coords
            poly[..., 1] += top
            for k in range(len(poly)):
                polys.append(poly[k])
                confs.append(float(result.obb.conf[k]))
                classes.append(int(result.obb.cls[k]))

    if not polys:
        return []

    keep = rotated_nms(polys, confs, classes, iou)
    return [([tuple(pt) for pt in polys[i]], DOTA_CLASSES[classes[i]], confs[i])
            for i in keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out-dir", default="predictions")
    ap.add_argument("--patch", type=int, default=1024)
    ap.add_argument("--gap", type=int, default=200)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU for merging overlaps")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    os.makedirs(args.out_dir, exist_ok=True)

    images = sorted(f for f in os.listdir(args.images_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg")))
    for fn in images:
        image = np.array(Image.open(os.path.join(args.images_dir, fn)).convert("RGB"))
        dets = detect_full_image(model, image, args.patch, args.gap, args.conf, args.iou)
        objs = [(corners, cls, 0) for corners, cls, _ in dets]
        draw_obbs(image, objs).save(os.path.join(args.out_dir, os.path.splitext(fn)[0] + ".png"))
        print(f"{fn}: {len(dets)} objects")
    print(f"-> annotated images in {args.out_dir}/")


if __name__ == "__main__":
    main()