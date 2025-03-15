"""
Created on 15/03/2025

@author: Aryan

Filename: prediction.py

Relative Path: prediction.py
"""


import cv2
import mediapipe as mp
import numpy as np
import time

from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions

# ---------------------------
# Create HandLandmarker
# ---------------------------
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,  # detect up to 2 hands
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# For drawing
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


class DetectionResult:
    """
    Container for hand detection results:
      - hand_landmarks: list of 21 normalized 2D landmarks per hand.
      - model_landmarks_list: list of 21 3D landmarks (model space in meters) per hand.
    """

    def __init__(self):
        self.hand_landmarks = []         # List of 2D landmark lists
        self.model_landmarks_list = []   # List of 3D landmark lists


def predict(frame_rgb):
    """
    Task 1: Predict hand landmarks using MediaPipe.
    Wrap the frame into an mp.Image with proper image_format.
    """
    frame_rgb = frame_rgb.astype(np.uint8)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    results = detector.detect(mp_image)
    detection_result = DetectionResult()

    # Updated to use 'hand_landmarks' and 'hand_world_landmarks'
    if results.hand_landmarks:
        for landm_2d in results.hand_landmarks:
            detection_result.hand_landmarks.append(landm_2d)
        for landm_3d in results.hand_world_landmarks:
            detection_result.model_landmarks_list.append(landm_3d)

    return detection_result


def draw_landmarks_on_image(image_bgr, detection_result):
    """
    Draw the 2D hand landmarks onto the BGR image.
    """
    if not detection_result or not detection_result.hand_landmarks:
        return image_bgr

    for hand_landmarks in detection_result.hand_landmarks:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=l.x, y=l.y, z=l.z
            ) for l in hand_landmarks
        ])
        mp_drawing.draw_landmarks(
            image_bgr,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style()
        )
    return image_bgr


def get_camera_matrix(frame_width, frame_height, scale=1.0):
    """
    Estimate an intrinsic camera matrix.
    """
    focal_length = frame_width * scale
    center = (frame_width / 2.0, frame_height / 2.0)
    camera_matrix = np.array(
        [[focal_length,       0, center[0]],
         [0,           focal_length, center[1]],
         [0,                   0,      1]],
        dtype=np.float64
    )
    return camera_matrix


def get_fov_y(camera_matrix, frame_height):
    """
    Compute vertical field-of-view for OpenGL.
    """
    focal_length_y = camera_matrix[1, 1]
    fov_y = np.rad2deg(2.0 * np.arctan2(frame_height / 2.0, focal_length_y))
    return fov_y


def get_matrix44(rvec, tvec):
    """
    Convert rotation (rvec) and translation (tvec) to a 4x4 transformation matrix.
    """
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def solvepnp(model_landmarks_list, image_landmarks_list, camera_matrix, frame_width, frame_height):
    """
    Task 2: Use solvePnP to align 3D model landmarks to 2D image coordinates.
    Returns a list of transformed 3D landmarks (in camera space, scaled to centimeters).
    """
    if not model_landmarks_list or not image_landmarks_list:
        return []

    world_landmarks_list = []
    for (model3d, image2d) in zip(model_landmarks_list, image_landmarks_list):
        model_points = np.float32([[lm.x, lm.y, lm.z] for lm in model3d])
        image_points = np.float32(
            [[lm2d.x * frame_width, lm2d.y * frame_height] for lm2d in image2d])

        dist_coeffs = None
        success, rvec, tvec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            R, _ = cv2.Rodrigues(rvec)
            new_world = []
            for p in model_points:
                p_cm = p * 100.0  # convert meters to centimeters
                p_cam = R @ p_cm + tvec.reshape(3)
                new_world.append(p_cam)
            new_world = np.array(new_world, dtype=np.float32)
            world_landmarks_list.append(new_world)
        else:
            world_landmarks_list.append(model_points)

    return world_landmarks_list


def reproject(world_landmarks_list, image_landmarks_list, camera_matrix, frame_width, frame_height):
    """
    Reproject the transformed 3D landmarks back to 2D.
    Returns the average reprojection error and a list of 2D points.
    """
    reprojection_points_list = []
    total_error = 0.0
    count = 0

    for (world_pts, image_lms) in zip(world_landmarks_list, image_landmarks_list):
        if len(world_pts) == 0:
            reprojection_points_list.append([])
            continue

        world_pts_3f = world_pts.reshape((-1, 1, 3))
        dist_coeffs = None
        image_points_2f = np.float32(
            [[lm2d.x * frame_width, lm2d.y * frame_height] for lm2d in image_lms]).reshape((-1, 1, 2))

        success, rvec, tvec = cv2.solvePnP(
            world_pts, image_points_2f, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            reprojection_points_list.append([])
            continue

        reprojected, _ = cv2.projectPoints(
            world_pts_3f, rvec, tvec, camera_matrix, dist_coeffs)
        reprojected = reprojected.reshape(-1, 2)
        reprojection_points_list.append(reprojected)

        gt_2d = image_points_2f.reshape(-1, 2)
        error = np.mean(np.linalg.norm(reprojected - gt_2d, axis=1))
        total_error += error
        count += 1

    avg_error = total_error / max(count, 1)
    return avg_error, reprojection_points_list


def check_pinch_gesture(hand_landmarks_2d):
    """
    Simple pinch detection: if the thumb tip and index fingertip are close.
    Assumes hand_landmarks_2d is a list of 21 normalized landmarks.
    """
    if len(hand_landmarks_2d) != 21:
        return False
    thumb = hand_landmarks_2d[4]
    index = hand_landmarks_2d[8]
    dx = thumb.x - index.x
    dy = thumb.y - index.y
    dist = (dx**2 + dy**2) ** 0.5
    return dist < 0.04
