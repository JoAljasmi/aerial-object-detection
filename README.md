# Aerial Object Detection on DOTA

An oriented-bounding-box (OBB) detector for aerial imagery, built end-to-end:
tiling large images, converting to YOLO-OBB format with a leakage-safe split,
fine-tuning YOLOv8-OBB, and evaluating with rotated-IoU mAP. DETR is added later
as an anchor-free contrast.

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
│   └── train.py            # train YOLOv8-OBB
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
carved from the labeled data in `prepare_dataset.py` — that split produces the
reported mAP. To shake out the pipeline without the full download, Ultralytics'
`dota8.yaml` auto-downloads an 8-image sample.

## Pipeline

```bash
# 1. inspect a sample (and see why tiling is needed — many sub-40px objects)
python src/visualize.py --image data/raw/train/images/P0001.png \
                        --label data/raw/train/labelTxt/P0001.txt --out annotated.png

# 2. tile train and val into 1024px patches
python src/tile.py --images-dir data/raw/train/images \
                   --labels-dir data/raw/train/labelTxt \
                   --out-dir data/tiled/train --preview 3
python src/tile.py --images-dir data/raw/val/images \
                   --labels-dir data/raw/val/labelTxt --out-dir data/tiled/val

# 3. QA the tiles (look at the previews)
python src/verify_tiles.py --tiled-dir data/tiled/train

# 4. convert to YOLO-OBB and split by source image
python src/prepare_dataset.py --tiled-dirs data/tiled/train data/tiled/val \
                              --out-dir data/yolo --seed 42

# 5. smoke-test, then train
python src/train.py --epochs 3 --fraction 0.05 --name smoke
python src/train.py --epochs 100 --batch 2 --workers 2 --name dota_obb
```

Then open `notebooks/evaluate.ipynb` to score `runs/obb/dota_obb/weights/best.pt`
on the test split.

## Roadmap

| Stage | Status |
|-------|--------|
| Visualize data + OBB format | done |
| Tiling | done |
| YOLO-OBB convert + leakage-safe split | done |
| Train YOLOv8-OBB | done |
| Evaluate (mAP50/50-95, per-class) | done |
| Inference + patch stitching | planned |
| DETR contrast | planned |