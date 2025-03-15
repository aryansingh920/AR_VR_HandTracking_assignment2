"""
Created on 15/03/2025

@author: Aryan

Filename: main.py

Relative Path: main.py
"""

import argparse
import sys
import faceHandTrack
import gl


def main():
    # Use a parser that ignores unknown arguments for our custom mode.
    parser = argparse.ArgumentParser(
        description="Augmented Reality with Hand Tracking", add_help=False)
    parser.add_argument(
        "--mode",
        type=str,
        default="opencv",
        choices=["opencv", "opengl"],
        help="Choose 'opencv' for 2D/3D debugging, 'opengl' for full AR rendering."
    )
    # Parse only our custom arguments
    args, remaining_args = parser.parse_known_args()

    if args.mode == "opencv":
        print("[INFO] Running OpenCV-only pipeline (Tasks 1 & 2)...")
        faceHandTrack.run_opencv_hand_tracking()
    else:
        print("[INFO] Running ModernGL AR pipeline (Tasks 3, 4 & 5)...")
        # Remove our custom '--mode' argument so moderngl_window doesn't see it.
        sys.argv = [sys.argv[0]] + remaining_args
        gl.CameraAR.run()


if __name__ == "__main__":
    main()
