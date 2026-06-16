# Aerial Object Detection on DOTA

An end-to-end oriented-bounding-box (OBB) detector for aerial imagery. It takes a
full-resolution overhead image and returns rotated boxes around objects in 15
classes — planes, ships, vehicles, harbors, and so on — handling the awkward
realities of aerial data: huge images, tiny objects, and targets at arbitrary
angles.

The project covers the whole pipeline: slicing large images into patches,
converting to YOLO-OBB format with a leakage-safe split, fine-tuning YOLOv8-OBB,
evaluating with rotated-IoU mAP, and running inference on full images by tiling
and stitching the results back together.

## Why it's built this way

Three design decisions drive the pipeline:

- **Tiling.** DOTA images run to several thousand pixels a side and are dense
  with sub-40px objects. Resizing a 4000px image to 640px erases those objects,
  so each image is sliced into overlapping 1024px patches at near-native
  resolution instead. Objects on a patch seam are kept whole in the patch that
  holds most of them.
- **Oriented boxes.** Seen from above, a ship or vehicle sits at any angle. An
  axis-aligned box would be mostly empty space and overlap its neighbours, so the
  model predicts four-corner oriented boxes.
- **Leakage-safe split.** The train/val/test split is by *source image*, not by
  patch — overlapping patches from one original image would otherwise leak across
  the boundary and inflate the score.

At inference the same windowing runs in reverse: tile the full image, predict per
patch, shift each detection back into full-image coordinates, and merge the
duplicates created by the overlap with rotated NMS.

## Results

Tiling DOTA-v1.0 (train + val) produced ~12,800 patches, split 80/10/10 by source
image into 10,216 / 1,281 / 1,303. Fine-tuning YOLOv8n-OBB reached, on the
validation split:

| metric | value |
|--------|-------|
| mAP@50 | 0.88 |
| mAP@50–95 | 0.67 |
| precision | 0.81 |
| recall | 0.84 |

## Honest limitations

- The base `yolov8n-obb` checkpoint was already pretrained on DOTA, so the model
  had effectively learned this task before fine-tuning. The validation mAP peaked
  at the first epoch and didn't improve afterwards — a sign there was little new
  to learn. Read these numbers as confirmation that the pipeline is wired
  correctly, not as a model trained from scratch.
- Reported metrics are on the validation split. A leakage-safe held-out test
  split exists; scoring `best.pt` on it via `notebooks/evaluate.ipynb` gives the
  cleanest read.
- For a genuinely earned learning curve, train from random init
  (`--model yolov8n-obb.yaml`) instead of the pretrained checkpoint — lower final
  mAP, but honestly the model's own.

## Layout

```
aerial-object-detection/
├── src/
│   ├── dota.py             # label parsing, drawing, stats, tiling
│   ├── check_setup.py      # verify torch/CUDA/ultralytics
│   ├── visualize.py        # render OBB labels on an image
│   ├── tile.py             # slice images into overlapping patches
│   ├── verify_tiles.py     # QA the tiled output before training
│   ├── prepare_dataset.py  # convert to YOLO-OBB + train/val/test split
│   ├── train.py            # train YOLOv8-OBB
│   └── detect.py           # full-image inference + stitching
├── notebooks/
│   ├── explore_data.ipynb  # interactive data + tiling walkthrough
│   └── evaluate.ipynb      # test-set metrics and sample predictions
├── data/                   # datasets (gitignored)
├── runs/                   # training outputs (gitignored)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python src/check_setup.py        # expects "CUDA : True"
```

If CUDA is False, install a torch build matching your GPU (see requirements.txt).

## Data

Download **DOTA-v1.0** (train + val) from
https://captain-whu.github.io/DOTA/dataset.html and arrange it as:

```
data/raw/
├── train/{images, labelTxt}
└── val/{images, labelTxt}
```

DOTA's official test set has no public labels, so the held-out test split is
carved from the labeled data in `prepare_dataset.py`.

## Pipeline

```bash
# 1. inspect a sample
python src/visualize.py --image data/raw/train/images/P0001.png \
                        --label data/raw/train/labelTxt/P0001.txt --out annotated.png

# 2. tile train and val into 1024px patches
python src/tile.py --images-dir data/raw/train/images \
                   --labels-dir data/raw/train/labelTxt --out-dir data/tiled/train
python src/tile.py --images-dir data/raw/val/images \
                   --labels-dir data/raw/val/labelTxt --out-dir data/tiled/val

# 3. QA the tiles
python src/verify_tiles.py --tiled-dir data/tiled/train

# 4. convert to YOLO-OBB and split by source image
python src/prepare_dataset.py --tiled-dirs data/tiled/train data/tiled/val \
                              --out-dir data/yolo --seed 42

# 5. smoke-test, then train
python src/train.py --epochs 3 --fraction 0.05 --name smoke
python src/train.py --epochs 100 --batch 2 --workers 2 --name dota_obb

# 6. run on full images (tile + predict + stitch)
python src/detect.py --weights runs/obb/dota_obb/weights/best.pt \
                     --images-dir data/raw/val/images --out-dir predictions
```

Then open `notebooks/evaluate.ipynb` to score the model on the test split.

## Stack

YOLOv8-OBB (Ultralytics), PyTorch, Shapely and OpenCV for the tiling geometry.
Built and trained on a single RTX 5060 Ti (16 GB).

## Possible next steps

- Train from random init for an uncontaminated benchmark.
- Ablate tiling vs. whole-image resizing to quantify what tiling buys.
- Export to ONNX/TensorRT and measure inference latency.