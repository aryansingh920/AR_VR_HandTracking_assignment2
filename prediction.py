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
    
    Args:
        frame: A NumPy array containing the RGB image data
        
    Returns:
        MediaPipe detection results with hand landmarks
    """
    # Convert the frame to a MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    results = detector.detect(mp_image)
    return results


def draw_landmarks_on_image(image, detection_result):
    """
    A helper function to draw the detected 2D landmarks on an image.
    
    Args:
        image: A NumPy array containing the BGR image data
        detection_result: MediaPipe detection results with hand landmarks
        
    Returns:
        Image with landmarks drawn on it
    """
    if not detection_result or not detection_result.hand_landmarks:
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
    
    Args:
        frame_width: Width of the image frame
        frame_height: Height of the image frame
        scale: Scale factor for focal length calculation (default: 1.0)
        
    Returns:
        3x3 camera intrinsic matrix as a NumPy array
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
    
    Args:
        camera_matrix: 3x3 camera intrinsic matrix
        frame_height: Height of the image frame
        
    Returns:
        Vertical field of view in degrees
    """
    focal_length_y = camera_matrix[1][1]
    fov_y = np.rad2deg(2 * np.arctan2(frame_height, 2 * focal_length_y))
    return fov_y


def get_matrix44(rvec, tvec):
    """
    Convert rotation and translation vectors into a 4x4 transformation matrix.
    
    Args:
        rvec: Rotation vector
        tvec: Translation vector
        
    Returns:
        4x4 transformation matrix
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
    
    Args:
        model_landmarks_list: List of 3D model landmarks
        image_landmarks_list: List of 2D image landmarks
        camera_matrix: 3x3 camera intrinsic matrix
        frame_width: Width of the image frame
        frame_height: Height of the image frame
        
    Returns:
        List of 3D world landmarks for each detected hand
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
    
    Args:
        world_landmarks_list: List of 3D world landmarks
        image_landmarks_list: List of 2D image landmarks
        camera_matrix: 3x3 camera intrinsic matrix
        frame_width: Width of the image frame
        frame_height: Height of the image frame
        
    Returns:
        Tuple of (reprojection_error, reprojection_points_list)
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


def check_pinch_gesture(hand_landmarks):
    """
    Detects if the user is performing a pinch gesture.
    
    A pinch occurs when the thumb tip is close to the index finger tip.
    
    Args:
        hand_landmarks: List of hand landmarks from MediaPipe
        
    Returns:
        Boolean indicating if a pinch is detected
    """
    if not hand_landmarks or len(hand_landmarks) < 9:  # Need at least thumb tip (4) and index tip (8)
        return False

    thumb_tip = np.array(
        [hand_landmarks[4].x, hand_landmarks[4].y])  # Thumb tip
    # Index finger tip
    index_tip = np.array([hand_landmarks[8].x, hand_landmarks[8].y])

    distance = np.linalg.norm(thumb_tip - index_tip)

    # Threshold for pinch detection (adjust as necessary)
    return distance < 0.05


def get_distance_between_points(point1, point2):
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point as NumPy array or list
        point2: Second point as NumPy array or list
        
    Returns:
        Euclidean distance between points
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


if __name__ == '__main__':
    # Example main function to display video and hand landmarks using OpenCV.
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("[ERROR] Could not open camera.")
        exit()

    previousTime = 0
    currentTime = 0

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        # Flip horizontally (mirror effect)
        frame = cv2.flip(frame, 1)

        # Resize and convert to RGB for MediaPipe
        aspect_ratio = frame.shape[1] / frame.shape[0]
        frame = cv2.resize(frame, (int(720 * aspect_ratio), 720))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmarks
        detection_result = predict(frame_rgb)

        # Draw the 2D landmarks on the frame
        frame_bgr = draw_landmarks_on_image(
            cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
            detection_result
        )

        # Calculate camera matrix for PnP
        frame_height, frame_width = frame_bgr.shape[:2]
        camera_matrix = get_camera_matrix(frame_width, frame_height)

        # Apply solvePnP if we have hand landmarks
        if hasattr(detection_result, 'hand_world_landmarks') and detection_result.hand_world_landmarks:
            world_landmarks_list = solvepnp(
                model_landmarks_list=detection_result.hand_world_landmarks,
                image_landmarks_list=detection_result.hand_landmarks,
                camera_matrix=camera_matrix,
                frame_width=frame_width,
                frame_height=frame_height
            )

            # Reproject 3D landmarks back to 2D and draw them in red
            repro_error, repro_points_list = reproject(
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

        # Add pinch detection for the first hand if available
        if hasattr(detection_result, 'hand_landmarks') and detection_result.hand_landmarks:
            is_pinching = check_pinch_gesture(
                detection_result.hand_landmarks[0]
            )
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

        # Display the frame
        cv2.imshow("OpenCV Hand Tracking (PnP)", frame_bgr)

        # Exit on 'ESC'
        if cv2.waitKey(5) & 0xFF == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
