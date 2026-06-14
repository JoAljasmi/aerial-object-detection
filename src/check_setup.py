"""Check the training environment: PyTorch build, CUDA/GPU, and Ultralytics."""

import sys


def main():
    try:
        import torch
    except ImportError:
        sys.exit("torch not installed. Run: pip install ultralytics")

    print(f"torch        : {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"CUDA         : {cuda}")

    if cuda:
        print(f"CUDA (build) : {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}        : {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    else:
        print("No GPU visible to torch. For an NVIDIA GPU install a CUDA build, e.g.:")
        print("  pip install torch torchvision "
              "--index-url https://download.pytorch.org/whl/cu128")

    try:
        import ultralytics
        print(f"ultralytics  : {ultralytics.__version__}")
    except ImportError:
        print("ultralytics not installed. Run: pip install ultralytics")


if __name__ == "__main__":
    main()