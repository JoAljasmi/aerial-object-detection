"""Train YOLOv8-OBB on the prepared DOTA patches.

Run as a script, not in a notebook (training is long-running and detached).
Smoke-test the loop first, then launch the real run:

    python train.py --epochs 3 --fraction 0.05 --name smoke
    python train.py --epochs 100 --batch 2 --workers 2 --name dota_obb

Outputs land in runs/obb/<name>/: weights/best.pt (best val mAP), weights/last.pt,
and plots (results, PR curve, confusion matrix).
"""

import os

# Must be set before torch initialises CUDA. expandable_segments fights memory
# fragmentation; the gc threshold reclaims cached blocks before they OOM.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)

import argparse


def main():
    ap = argparse.ArgumentParser(description="Train YOLOv8-OBB on DOTA.")
    ap.add_argument("--data", default="data/yolo/dota-obb.yaml")
    # nano model, pretrained on DOTAv1 — fast baseline; scale to -s/-m for accuracy.
    ap.add_argument("--model", default="yolov8n-obb.pt")
    ap.add_argument("--epochs", type=int, default=100)
    # Must match the tile size, or patches get downsampled and small objects suffer.
    ap.add_argument("--imgsz", type=int, default=1024)
    # Fixed batch, not AutoBatch: the loss assigner's memory scales with object
    # density, which AutoBatch can't profile, so it overshoots on dense patches.
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0", help="'0' = first GPU, 'cpu' to force CPU")
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of data to use")
    ap.add_argument("--name", default="dota_obb")
    # Dataloader workers hold decoded patches in system RAM; lower on 16 GB machines.
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed. Run: pip install ultralytics")

    model = YOLO(args.model)

    # The nightly Blackwell build doesn't reliably free validation memory, which
    # OOMs at the next epoch's start; clear the cache after each epoch.
    import gc
    import torch

    def free_cuda(trainer):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.add_callback("on_fit_epoch_end", free_cuda)

    # Ultralytics already defaults OBB runs to runs/obb/, so don't set project.
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        fraction=args.fraction,
        name=args.name,
        resume=args.resume,
        plots=True,
        val=True,
    )
    print(f"Done. Best weights: runs/obb/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()