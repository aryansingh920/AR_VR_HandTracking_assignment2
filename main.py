"""
Created on 15/03/2025

@author: Aryan

Filename: main.py

Relative Path: main.py
"""


import argparse
import faceHandTrack
import gl


def main():
    parser = argparse.ArgumentParser(
        description="Augmented Reality with Hand Tracking")
    parser.add_argument(
        "--mode",
        type=str,
        default="opencv",
        choices=["opencv", "opengl"],
        help="Choose 'opencv' for 2D/3D debugging, 'opengl' for full AR rendering."
    )
    args = parser.parse_args()

    if args.mode == "opencv":
        print("[INFO] Running OpenCV-only pipeline (Tasks 1 & 2)...")
        faceHandTrack.run_opencv_hand_tracking()
    else:
        print("[INFO] Running ModernGL AR pipeline (Tasks 3, 4 & 5)...")
        gl.CameraAR.run()


if __name__ == "__main__":
    main()
