"""
Stage 0 - Environment check.

Run this FIRST. It confirms three things before you waste time on data:
  1. PyTorch is installed and can see your NVIDIA GPU.
  2. Ultralytics is installed.
  3. The GPU has enough memory to train comfortably.

Usage:
    python check_setup.py
"""

import sys


def main() -> None:
    ok = True

    # --- PyTorch + CUDA -------------------------------------------------
    try:
        import torch
    except ImportError:
        print("[FAIL] torch is not installed. Run: pip install ultralytics")
        sys.exit(1)

    print(f"torch version      : {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available     : {cuda_ok}")

    if cuda_ok:
        # torch.version.cuda is the CUDA toolkit the wheel was built against.
        print(f"CUDA (torch build) : {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            total_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}            : {name} ({total_gb:.1f} GB)")
            if total_gb < 6:
                print("    [WARN] <6 GB VRAM: use the 'n' (nano) model and a small "
                      "batch size, or expect out-of-memory errors.")
    else:
        ok = False
        print("[FAIL] torch cannot see a GPU. You can still run on CPU but training "
              "will be painfully slow. To fix, install a CUDA torch build, e.g.:")
        print("       pip install torch torchvision "
              "--index-url https://download.pytorch.org/whl/cu121")

    # --- Ultralytics ----------------------------------------------------
    try:
        import ultralytics
        print(f"ultralytics version: {ultralytics.__version__}")
    except ImportError:
        ok = False
        print("[FAIL] ultralytics is not installed. Run: pip install ultralytics")

    print()
    print("All good - ready for Stage 1." if ok
          else "Fix the [FAIL] items above before continuing.")


if __name__ == "__main__":
    main()
