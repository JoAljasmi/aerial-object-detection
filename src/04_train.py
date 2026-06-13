"""
Stage 4 - Train YOLOv8-OBB on the prepared DOTA patches.

This is a thin, heavily-commented wrapper over Ultralytics' training loop. Run
it as a SCRIPT (not in a notebook): training is a long, detached job, and a
dropped notebook kernel would kill it.

Workflow:
  1. SMOKE TEST first - a few epochs on a small fraction, to prove the loop runs
     end-to-end before you commit hours:
        python scripts/04_train.py --epochs 3 --fraction 0.05 --name smoke
  2. REAL RUN once the smoke test completes cleanly:
        python scripts/04_train.py --epochs 100 --name dota_obb

Outputs land in runs/obb/<name>/:
  - weights/best.pt   <- best model by val mAP (this is what Stage 5 evaluates)
  - weights/last.pt   <- last epoch (use to --resume if interrupted)
  - results.png, confusion_matrix.png, PR_curve.png, val batch previews
"""

import os

# Mitigate CUDA memory fragmentation (the "X GiB free but can't allocate Y MiB"
# failure in the loss assigner). MUST be set before torch initializes CUDA, so
# it lives at the very top, before any import that pulls in torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse


def main():
    ap = argparse.ArgumentParser(description="Train YOLOv8-OBB on DOTA.")
    ap.add_argument("--data", default="data/yolo/dota-obb.yaml",
                    help="dataset YAML from Stage 3 (new layout: data/yolo/)")
    # yolov8n-obb is the nano model: fastest, lowest VRAM, good first baseline.
    # It is PRETRAINED on DOTAv1, so this is fine-tuning - it starts already
    # knowing aerial objects and converges faster. Scale up to -s/-m later for
    # higher accuracy at the cost of speed/VRAM.
    ap.add_argument("--model", default="yolov8n-obb.pt")
    ap.add_argument("--epochs", type=int, default=100)
    # imgsz MUST match your patch size (1024). If you leave it at the 640
    # default, Ultralytics downsizes every patch to 640 and you throw away the
    # resolution that tiling worked so hard to preserve - small objects suffer.
    ap.add_argument("--imgsz", type=int, default=1024)
    # batch: how many patches per step. We do NOT use AutoBatch (-1) here:
    # AutoBatch profiles forward/backward with empty dummy targets, so it can't
    # see the loss assigner's memory, which scales with object count. Aerial
    # patches are dense (hundreds of objects), so AutoBatch overshoots and OOMs
    # in TaskAlignedAssigner. A fixed, modest batch is safer. Start at 8; if you
    # still OOM in the assigner, drop to 4.
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0", help="'0' = first GPU, 'cpu' to force CPU")
    # patience: stop early if val mAP hasn't improved for this many epochs.
    ap.add_argument("--patience", type=int, default=50)
    # fraction: train on this fraction of the data (use <1.0 for smoke tests).
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--name", default="dota_obb", help="run name under runs/obb/")
    ap.add_argument("--resume", action="store_true",
                    help="resume an interrupted run from its last.pt")
    args = ap.parse_args()

    # Import here (not at top) so the file is inspectable without ultralytics
    # installed, and so a missing-install gives a clear message.
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics is not installed. In your (3.12) venv:\n"
            "    pip install ultralytics\n"
            "then re-run check_setup.py to confirm torch sees your GPU.")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        fraction=args.fraction,
        # NOTE: don't set project="runs/obb" - Ultralytics already defaults OBB
        # runs to runs/obb/, so setting it would nest to runs/obb/runs/obb/<name>.
        name=args.name,
        resume=args.resume,
        # sensible defaults; Ultralytics' augmentation (mosaic, flips, scaling)
        # is reasonable for aerial imagery out of the box - don't over-tune yet.
        plots=True,        # save loss curves, PR curve, confusion matrix
        val=True,          # evaluate on the val split each epoch
    )
    print("\nDone. Best weights: runs/obb/%s/weights/best.pt" % args.name)


if __name__ == "__main__":
    main()