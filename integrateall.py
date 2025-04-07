import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# Helper function for angle calculation
def angle_between(v1, v2):
    """Calculate angle between two vectors in degrees"""
    if np.linalg.norm(v1) * np.linalg.norm(v2) == 0:
        return 0
    dot_product = np.dot(v1, v2)
    magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    cosine_angle = dot_product / magnitude_product
    # Clamp to avoid numerical errors
    cosine_angle = max(min(cosine_angle, 1.0), -1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start video capture
cap = cv2.VideoCapture(0)

# Auto-detect camera resolution
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Zoom variables
zoom_scale = 1.0
zoom_increment = 0.1

# Store previous positions & timestamps
prev_left_wrist = None
prev_right_wrist = None
still_start_time = None
warning_delay = 2  # Seconds before warning appears

# Posture tracking
posture_buffer = deque(maxlen=15)  # For smoothing posture detection over multiple frames
posture_threshold = 0.018  # More precise threshold
shoulder_alignment_threshold = 0.015  # Threshold for shoulder alignment
neck_angle_threshold = 165  # Degrees - straighter neck is higher angle

# Setup window
cv2.namedWindow("AI Public Speaking Coach", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture video frame.")
        break

    # Zoom handling
    if zoom_scale != 1.0:
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        radius_x, radius_y = int(center_x / zoom_scale), int(center_y / zoom_scale)

        # Safe boundaries
        min_x = max(center_x - radius_x, 0)
        max_x = min(center_x + radius_x, frame.shape[1])
        min_y = max(center_y - radius_y, 0)
        max_y = min(center_y + radius_y, frame.shape[0])

        cropped = frame[min_y:max_y, min_x:max_x]
        frame = cv2.resize(cropped, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

    # Convert to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    hand_warning_displayed = False
    hunching_detected = False
    posture_score = 0  # 0-100 scale where 100 is perfect

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape
        
        # HAND MOVEMENT DETECTION
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        left_wrist_pos = np.array([left_wrist.x, left_wrist.y])
        right_wrist_pos = np.array([right_wrist.x, right_wrist.y])

        if prev_left_wrist is not None and prev_right_wrist is not None:
            left_movement = np.linalg.norm(left_wrist_pos - prev_left_wrist)
            right_movement = np.linalg.norm(right_wrist_pos - prev_right_wrist)

            movement_threshold = 0.01

            if left_movement < movement_threshold and right_movement < movement_threshold:
                if still_start_time is None:
                    still_start_time = time.time()
                elif time.time() - still_start_time > warning_delay:
                    hand_warning_displayed = True
            else:
                still_start_time = None

        prev_left_wrist = left_wrist_pos
        prev_right_wrist = right_wrist_pos

        # IMPROVED POSTURE DETECTION
        # Extract key landmarks
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
        
        # 1. Calculate mid-points
        shoulder_mid = np.array([(left_shoulder.x + right_shoulder.x) / 2, 
                                 (left_shoulder.y + right_shoulder.y) / 2])
        hip_mid = np.array([(left_hip.x + right_hip.x) / 2, 
                            (left_hip.y + right_hip.y) / 2])
        ear_mid = np.array([(left_ear.x + right_ear.x) / 2, 
                           (left_ear.y + right_ear.y) / 2])
        
        # 2. Calculate head forward position (normalized)
        head_forward = nose.x - shoulder_mid[0]
        
        # 3. Calculate shoulder roundness (normalized)
        shoulder_forward = shoulder_mid[0] - hip_mid[0]
        
        # 4. Calculate shoulder levelness
        shoulder_levelness = abs(left_shoulder.y - right_shoulder.y)
        
        # 5. Calculate neck angle
        # Convert to pixel coordinates for angle calculation
        ear_mid_px = (int(ear_mid[0] * w), int(ear_mid[1] * h))
        shoulder_mid_px = (int(shoulder_mid[0] * w), int(shoulder_mid[1] * h))
        hip_mid_px = (int(hip_mid[0] * w), int(hip_mid[1] * h))
        
        # Calculate vectors
        neck_vector = np.array([shoulder_mid_px[0] - ear_mid_px[0], 
                                shoulder_mid_px[1] - ear_mid_px[1]])
        spine_vector = np.array([hip_mid_px[0] - shoulder_mid_px[0], 
                                hip_mid_px[1] - shoulder_mid_px[1]])
        
        # Calculate neck angle (in degrees)
        neck_angle = 180 - angle_between(neck_vector, spine_vector)
        
        # Calculate overall posture score
        posture_metrics = {
            'head_forward': head_forward < posture_threshold,
            'shoulder_forward': shoulder_forward < posture_threshold,
            'shoulder_level': shoulder_levelness < shoulder_alignment_threshold,
            'neck_straight': neck_angle > neck_angle_threshold
        }
        
        # Overall posture score (0-100)
        posture_score = sum(posture_metrics.values()) * 25
        
        # Add current posture score to buffer for smoothing
        posture_buffer.append(posture_score)
        avg_posture_score = sum(posture_buffer) / len(posture_buffer)
        
        # Use smoothed score to determine if hunching
        hunching_detected = avg_posture_score < 75
        
        # Debug visualization - draw the spine and neck
        cv2.line(frame, ear_mid_px, shoulder_mid_px, (0, 255, 0), 2)
        cv2.line(frame, shoulder_mid_px, hip_mid_px, (0, 255, 0), 2)
        
        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    # Add coaching title banner
    title = "AI Public Speaking Coach"
    title_font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.7
    title_thickness = 2
    title_size = cv2.getTextSize(title, title_font, title_scale, title_thickness)[0]
    title_x = 20
    title_y = 40
    
    # Add translucent background for title
    cv2.rectangle(frame, (10, 10), (title_x + title_size[0] + 10, title_y + 10), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, 10), (title_x + title_size[0] + 10, title_y + 10), (0, 140, 255), 2)
    cv2.putText(frame, title, (title_x, title_y), title_font, title_scale, (255, 255, 255), title_thickness, cv2.LINE_AA)

    # Display warning messages directly on the video frame
    warning_y_position = frame_height - 50  # Start position from bottom

    # Function to display warning with background
    def display_warning(text, y_pos, color=(0, 255, 255)):
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.8
        thickness = 2
        
        # Get text size for centering
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (frame_width - text_size[0]) // 2
        
        # Create semi-transparent background for better readability
        overlay = frame.copy()
        bg_padding = 10
        bg_top_left = (text_x - bg_padding, y_pos - text_size[1] - bg_padding)
        bg_bottom_right = (text_x + text_size[0] + bg_padding, y_pos + bg_padding)
        cv2.rectangle(overlay, bg_top_left, bg_bottom_right, (40, 40, 40), -1)
        
        # Apply transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw text with shadow for better visibility
        cv2.putText(frame, text, (text_x + 2, y_pos + 2), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, text, (text_x, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)
        
        return y_pos - 60  # Return position for next warning (if any)

    # Display posture score
    if results.pose_landmarks:
        score_text = f"Posture Score: {int(posture_score)}/100"
        score_color = (0, 255, 0) if posture_score >= 75 else (0, 165, 255) if posture_score >= 50 else (0, 0, 255)
        cv2.putText(frame, score_text, (frame_width - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, score_color, 2, cv2.LINE_AA)

    # Display warnings based on detected issues
    if hand_warning_displayed:
        warning_y_position = display_warning("⚠️ Use hand gestures to engage your audience!", warning_y_position)
        
    if hunching_detected and results.pose_landmarks:
        display_warning("⚠️ Straighten your back and align your posture!", warning_y_position, color=(0, 0, 255))

    # Show the video feed
    cv2.imshow("AI Public Speaking Coach", frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord('+'), ord('=')]:
        zoom_scale = min(zoom_scale + zoom_increment, 3.0)  # Max zoom x3
    elif key in [ord('-'), ord('_')]:
        zoom_scale = max(zoom_scale - zoom_increment, 1.0)  # Min zoom x1

cap.release()
cv2.destroyAllWindows()