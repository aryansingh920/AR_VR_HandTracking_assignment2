import cv2
import mediapipe as mp
import time

# Initialize MediaPipe modules for Face Mesh and Hand Tracking
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Face Mesh Configuration
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_drawing_spec = mp_drawing.DrawingSpec(
    color=(255, 255, 255), thickness=1, circle_radius=1)

# Hand Tracking Configuration
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# input_video_path = "input.mp4"  # or None
input_video_path = None
cap = cv2.VideoCapture(input_video_path if input_video_path else 0)

# Now, save the output as .mov instead of .avi
# output_video_path = "output.mov"
output_video_path = "output.mp4"
# 'mp4v' works for .mov on many platforms
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# Better for MP4, requires H.264 codec support
fourcc = cv2.VideoWriter_fourcc(*'H264')

out = None
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

if output_video_path:
    out = cv2.VideoWriter(output_video_path, fourcc, fps,
                          (frame_width, frame_height))
##########################################################################

# FPS Calculation
previous_time = 0


def detect_finger_gesture(hand_landmarks):
    """
    Detects if a hand is showing an 'Open Palm' (all fingers extended) or a 'Fist' (all fingers folded).
    Uses the y-coordinates of finger tips relative to their respective lower joints.
    """
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

    index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    middle_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
    ring_base = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
    pinky_base = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y

    # Check if all fingers are extended (Open Palm)
    if (index_tip < index_base and
        middle_tip < middle_base and
        ring_tip < ring_base and
            pinky_tip < pinky_base):
        return "Open Palm"

    # Check if all fingers are curled (Fist)
    if (index_tip > index_base and
        middle_tip > middle_base and
        ring_tip > ring_base and
            pinky_tip > pinky_base):
        return "Fist"

    return "Unknown"


while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip image for a front-facing camera effect
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process Face and Hand Detection
    face_results = face_mesh.process(rgb_image)
    hand_results = hands.process(rgb_image)

    # Draw Face Mesh Landmarks
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=face_drawing_spec,
                connection_drawing_spec=face_drawing_spec
            )

    # Draw Hand Landmarks and Detect Finger Gesture
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS
            )
            gesture = detect_finger_gesture(hand_landmarks)
            cv2.putText(image, gesture, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Calculate and Display FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(image, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show Image
    cv2.imshow('Face & Hand Recognition', image)

    # Save frame to output video
    if out:
        out.write(image)

    # Exit on 'ESC' key
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Cleanup
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
