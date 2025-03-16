"""
Created on 16/03/2025

@author: Aryan

Filename: prediction.py

Relative Path: assignment2/prediction.py
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
# from mediapipe.framework.formats import image_frame  # Import ImageFrame

import numpy as np
import cv2
import time

# Create a MediaPipe HandLandmarker detector.
# Note: To support both hands detection, you may change num_hands to 2.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)


def predict(frame):
    """
    Task 1: Implement hand landmark prediction.
    Convert the NumPy array to a MediaPipe Image and pass it to the detector.
    """
    # Convert the frame to a MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    results = detector.detect(mp_image)
    return results


def draw_landmarks_on_image(image, detection_result):
    """
    A helper function to draw the detected 2D landmarks on an image.
    """
    if not detection_result:
        return image

    hand_landmarks_list = detection_result.hand_landmarks
    # Loop through detected hands and draw landmarks on the image.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())
    return image


def get_camera_matrix(frame_width, frame_height, scale=1.0):
    """
    Compute the 3x3 camera intrinsic matrix.
    """
    focal_length = frame_width * scale
    center = (frame_width / 2.0, frame_height / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    return camera_matrix


def get_fov_y(camera_matrix, frame_height):
    """
    Compute the vertical field of view from the camera matrix.
    """
    focal_length_y = camera_matrix[1][1]
    fov_y = np.rad2deg(2 * np.arctan2(frame_height, 2 * focal_length_y))
    return fov_y


def get_matrix44(rvec, tvec):
    """
    Convert rotation and translation vectors into a 4x4 transformation matrix.
    """
    rvec = np.asarray(rvec)
    tvec = np.asarray(tvec)
    T = np.eye(4)
    R, _ = cv2.Rodrigues(rvec)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def solvepnp(model_landmarks_list, image_landmarks_list, camera_matrix, frame_width, frame_height):
    """
    Solve for global rotation and translation to map hand model points to camera space.
    """
    if not model_landmarks_list:
        return []

    world_landmarks_list = []

    for (model_landmarks, image_landmarks) in zip(model_landmarks_list, image_landmarks_list):
        # Convert landmarks to NumPy arrays.
        model_points = np.float32([[l.x, l.y, l.z] for l in model_landmarks])
        image_points = np.float32(
            [[l.x * frame_width, l.y * frame_height] for l in image_landmarks])

        # Solve PnP: estimate rvec and tvec.
        retval, rvec, tvec = cv2.solvePnP(
            model_points, image_points, camera_matrix, None)
        if retval:
            T = get_matrix44(rvec, tvec)
            ones = np.ones((model_points.shape[0], 1), dtype=np.float32)
            model_points_hom = np.hstack([model_points, ones])
            world_points_hom = (T @ model_points_hom.T).T
            # Convert from meters to centimeters.
            world_points = world_points_hom[:, :3] * 100.0
            world_landmarks_list.append(world_points)

    return world_landmarks_list


def reproject(world_landmarks_list, image_landmarks_list, camera_matrix, frame_width, frame_height):
    """
    Reproject 3D world landmarks to the image plane for visualization.
    """
    reprojection_points_list = []
    reprojection_error = 0.0
    for (world_landmarks, image_landmarks) in zip(world_landmarks_list, image_landmarks_list):
        output = world_landmarks.dot(camera_matrix.T)
        output[:, 0] /= output[:, 2]
        output[:, 1] /= output[:, 2]
        reprojection_points_list.append(output[:, :2])
        image_points = np.float32(
            [[l.x * frame_width, l.y * frame_height] for l in image_landmarks])
        reprojection_error += np.linalg.norm(output[:, :2] - image_points) / len(
            output) / len(world_landmarks_list)
    return reprojection_error, reprojection_points_list


if __name__ == '__main__':
    # Example main function to display video and hand landmarks using OpenCV.
    capture = cv2.VideoCapture(0)
    previousTime = 0
    currentTime = 0

    while capture.isOpened():
        ret, frame = capture.read()
        aspect_ratio = frame.shape[1] / frame.shape[0]
        frame = cv2.resize(frame, (int(720 * aspect_ratio), 720))
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_result = predict(frame_rgb)
        frame = draw_landmarks_on_image(frame, detection_result)

        # (Optional) SolvePnP and reproject landmarks here for debugging.

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        cv2.putText(frame, str(int(fps)) + " FPS", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
