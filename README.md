Aerial Object Detection on DOTA
Build an oriented-bounding-box (OBB) detector for aerial imagery, end-to-end,
understanding every stage. We start with YOLOv8-OBB to a real mAP number, then
add DETR as a contrast.
Roadmap
StageWhatWhy it matters0Environment checkConfirm the GPU works before anything else1See the data (you are here)Understand OBB + DOTA's quad format2TilingSlice giant images into 1024px patches; small objects survive3Format convert + splitDOTA quads -> YOLO-OBB; build train/val/test4Train yolov8-obbBackbone/neck/head, angle prediction, loss5EvaluatemAP50 vs mAP50-95, rotated IoU, per-class AP6Inference + stitchingMerge patch detections, NMS across seams7DETR contrastTransformer set-prediction vs anchor-based
Setup (Stage 0)
bashpip install -r requirements.txt
python check_setup.py          # must report CUDA available: True
If CUDA is False, install a torch build matching your CUDA version (see the
note in requirements.txt).
Get the data
Download DOTA-v1.0 from the official source:
https://captain-whu.github.io/DOTA/dataset.html
You want the training and validation sets. Each comes as:

image archives (part1.zip, part2.zip, ...) -> extract into an images/ folder
an annotation archive -> extract into a labelTxt/ folder of .txt files

Arrange it like this:
DOTA/
  train/
    images/      P0000.png, P0001.png, ...
    labelTxt/    P0000.txt, P0001.txt, ...
  val/
    images/
    labelTxt/
Note: DOTA's official test set has no public labels (you'd submit to their
evaluation server for an official score). For a reproducible project we'll
instead carve our own held-out test split from the labeled train+val data in
Stage 3 — that's the mAP number that goes in your writeup.
Tip: to test the whole pipeline first without the multi-GB download, Ultralytics
ships an 8-image sample (dota8.yaml) that auto-downloads. Good for shaking out
bugs before committing to the full run.
See the data (Stage 1)
bash# one image
python scripts/01_visualize_dota.py \
    --image DOTA/train/images/P0001.png \
    --label DOTA/train/labelTxt/P0001.txt \
    --out annotated.png

# or render the first 5 of a folder
python scripts/01_visualize_dota.py \
    --images-dir DOTA/train/images \
    --labels-dir DOTA/train/labelTxt \
    --out-dir previews --num 5
It draws the oriented boxes and prints stats per image — image size, object
count, and how many objects are under 40 px. Look at that last number: it's why
Stage 2 (tiling) exists. Downsizing a 4000px image to 640 destroys those tiny
objects; tiling preserves them.