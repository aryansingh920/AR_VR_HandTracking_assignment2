"""
Created on 15/03/2025

@author: Aryan

Filename: faceHandTrack.py

Relative Path: faceHandTrack.py
"""

import cv2
import time
import prediction
import numpy as np


def check_pinch_gesture(hand_landmarks):
    """
    Detects if the user is performing a pinch gesture.
    
    A pinch occurs when the thumb tip is close to the index finger tip.
    """
    if len(hand_landmarks) < 2:
        return False  # Not enough landmarks detected

    thumb_tip = np.array(
        [hand_landmarks[4].x, hand_landmarks[4].y])  # Thumb tip
    # Index finger tip
    index_tip = np.array([hand_landmarks[8].x, hand_landmarks[8].y])

    distance = np.linalg.norm(thumb_tip - index_tip)

    # Threshold for pinch detection (adjust as necessary)
    return distance < 0.05


def run_opencv_hand_tracking():
    """
    Displays real-time hand tracking using OpenCV (Tasks 1 & 2).
    Shows:
      - 2D landmarks from MediaPipe
      - 3D landmarks reprojected via solvePnP
      - Basic pinch detection feedback
    """
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("[ERROR] Could not open camera.")
        return

    previousTime = time.time()

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # Flip horizontally (mirror effect)
        frame = cv2.flip(frame, 1)
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1) Get hand landmarks using prediction module
        detection_result = prediction.predict(frame_rgb)

        # Draw the 2D landmarks on the frame
        frame_bgr = prediction.draw_landmarks_on_image(
            cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
            detection_result
        )

        # 2) Use solvePnP to align 3D landmarks to the image
        frame_height, frame_width = frame_bgr.shape[:2]
        camera_matrix = prediction.get_camera_matrix(frame_width, frame_height)

        # Ensure hand_world_landmarks exist
        if hasattr(detection_result, 'hand_world_landmarks') and detection_result.hand_world_landmarks:
            world_landmarks_list = prediction.solvepnp(
                model_landmarks_list=detection_result.hand_world_landmarks,
                image_landmarks_list=detection_result.hand_landmarks,
                camera_matrix=camera_matrix,
                frame_width=frame_width,
                frame_height=frame_height
            )

            # 3) Reproject the 3D landmarks back to 2D to check alignment
            repro_error, repro_points_list = prediction.reproject(
                world_landmarks_list,
                detection_result.hand_landmarks,
                camera_matrix,
                frame_width,
                frame_height
            )

            # Draw the reprojected landmarks in red
            for pts2d in repro_points_list:
                for (px, py) in pts2d:
                    cv2.circle(frame_bgr, (int(px), int(py)),
                               5, (0, 0, 255), -1)

        # (Optional) Basic pinch detection on the first hand if available
        if hasattr(detection_result, 'hand_landmarks') and detection_result.hand_landmarks:
            is_pinching = check_pinch_gesture(
                detection_result.hand_landmarks[0])
            gesture_text = "Pinch" if is_pinching else "No Pinch"
            cv2.putText(
                frame_bgr,
                gesture_text,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # Calculate and display FPS
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        cv2.putText(
            frame_bgr,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("OpenCV Hand Tracking (PnP)", frame_bgr)

        # Exit on 'ESC'
        if cv2.waitKey(5) & 0xFF == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
