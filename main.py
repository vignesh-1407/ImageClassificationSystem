"""
main.py - Unified CLI entry point for the AI Image Classification System

Usage:
    python main.py --mode train
    python main.py --mode evaluate
    python main.py --mode predict --image path/to/image.jpg
    python main.py --mode predict --webcam
    python main.py --mode summary
"""

import argparse
import sys
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Advanced AI-Based Image Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train
  python main.py --mode evaluate
  python main.py --mode predict --image sample_images/cat.jpg
  python main.py --mode predict --webcam
  python main.py --mode summary
        """
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["train", "evaluate", "predict", "summary"],
        help="Mode to run: train | evaluate | predict | summary"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to image for prediction (used with --mode predict)"
    )
    parser.add_argument(
        "--webcam", action="store_true",
        help="Use webcam for real-time prediction (used with --mode predict)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to a specific model file (optional, uses default if not set)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs for training"
    )
    return parser.parse_args()


def print_banner():
    banner = r"""
  ╔══════════════════════════════════════════════════════════╗
  ║   Advanced AI-Based Image Classification System         ║
  ║   CNN  |  TensorFlow  |  Keras  |  OpenCV  |  NumPy     ║
  ║   Built with Antigravity Platform                       ║
  ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    print_banner()
    args = parse_args()

    # Override epochs if provided
    if args.epochs is not None:
        import config
        config.EPOCHS = args.epochs
        print(f"[CONFIG] Epochs overridden to: {config.EPOCHS}")

    if args.mode == "train":
        from src.train import run_training
        run_training()

    elif args.mode == "evaluate":
        from src.evaluate import run_evaluation
        run_evaluation(model_path=args.model)

    elif args.mode == "predict":
        if args.webcam:
            from src.predict import predict_webcam
            predict_webcam(model_path=args.model)
        elif args.image:
            if not os.path.exists(args.image):
                print(f"[ERROR] Image not found: {args.image}")
                sys.exit(1)
            from src.predict import predict_single
            predict_single(args.image, model_path=args.model)
        else:
            print("[ERROR] Provide --image <path> or --webcam with --mode predict")
            sys.exit(1)

    elif args.mode == "summary":
        from src.model import build_cnn, model_summary
        model = build_cnn()
        model_summary(model)
        print("\n[INFO] Model built successfully. Run --mode train to begin training.")


if __name__ == "__main__":
    main()
