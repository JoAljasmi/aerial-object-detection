

import os

# Mitigate CUDA memory fragmentation + OOM at epoch boundaries. MUST be set
# before torch initializes CUDA, so it lives at the very top, before any import
# that pulls in torch.
#   expandable_segments:        hand out non-contiguous blocks (fights fragmentation)
#   garbage_collection_threshold: reclaim cached memory once usage passes 80%
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)

import argparse


def main():
    ap = argparse.ArgumentParser(description="Train YOLOv8-OBB on DOTA.")
    ap.add_argument("--data", default="data/yolo/dota-obb.yaml",
                    help="dataset YAML from Stage 3 (new layout: data/yolo/)")
   
    ap.add_argument("--model", default="yolov8n-obb.pt")
    ap.add_argument("--epochs", type=int, default=100)
   
    ap.add_argument("--imgsz", type=int, default=1024)
    
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0", help="'0' = first GPU, 'cpu' to force CPU")
    # patience: stop early if val mAP hasn't improved for this many epochs.
    ap.add_argument("--patience", type=int, default=50)
    # fraction: train on this fraction of the data (use <1.0 for smoke tests).
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--name", default="dota_obb", help="run name under runs/obb/")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true",
                    help="resume an interrupted run from its last.pt")
    args = ap.parse_args()

    
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics is not installed. In your (3.12) venv:\n"
            "    pip install ultralytics\n"
            "then re-run check_setup.py to confirm torch sees your GPU.")

    model = YOLO(args.model)

    
    import gc
    import torch

    def _free_cuda(trainer):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.add_callback("on_fit_epoch_end", _free_cuda)  # fires after train+val each epoch

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
    print("\nDone. Best weights: runs/obb/%s/weights/best.pt" % args.name)


if __name__ == "__main__":
    main()