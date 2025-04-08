import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp

class PostureAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(model_complexity=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

        self.prev_left_wrist = None
        self.prev_right_wrist = None
        self.still_start_time = None
        self.warning_delay = 2

        self.posture_buffer = deque(maxlen=15)
        self.posture_threshold = 0.018
        self.shoulder_alignment_threshold = 0.015
        self.neck_angle_threshold = 165

        self.hand_warning_displayed = False
        self.hunching_detected = False
        self.posture_score = 0

def analyze_frame(self, frame):
    if frame is None:
        return frame, {}
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = self.pose.process(frame_rgb)

    self.hand_warning_displayed = False
    self.hunching_detected = False
    self.posture_score = 100  # Start perfect, deduct points
    posture_metrics = {}

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        # Extract relevant points
        left_wrist = np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                               landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y,
                               landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].z])
        right_wrist = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].x,
                                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y,
                                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].z])
        left_shoulder = np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                  landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                                  landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].z])
        right_shoulder = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                   landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                                   landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z])
        nose = np.array([landmarks[self.mp_pose.PoseLandmark.NOSE].x,
                         landmarks[self.mp_pose.PoseLandmark.NOSE].y,
                         landmarks[self.mp_pose.PoseLandmark.NOSE].z])

        left_ear = np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].x,
                             landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].y,
                             landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].z])

        # Improved hand position detection
        left_hand_to_shoulder_dist = np.linalg.norm(left_wrist - left_shoulder)
        right_hand_to_shoulder_dist = np.linalg.norm(right_wrist - right_shoulder)

        hand_threshold = 0.2  # Adjust based on camera distance

        if left_hand_to_shoulder_dist > hand_threshold or right_hand_to_shoulder_dist > hand_threshold:
            self.hand_warning_displayed = True
            self.posture_score -= 20

        # Hunchback detection — using angle between shoulder and ear
        shoulder_to_ear = left_ear - left_shoulder
        horizontal_vector = np.array([1, 0, 0])  # Reference for upright

        angle_rad = np.arccos(np.clip(np.dot(shoulder_to_ear[:2], horizontal_vector[:2]) /
                                      (np.linalg.norm(shoulder_to_ear[:2]) * np.linalg.norm(horizontal_vector[:2])), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        if angle_deg < 60:  # Less than 60 degrees means hunching
            self.hunching_detected = True
            self.posture_score -= 30

        # Prepare posture metrics
        posture_metrics = {
            'hand_distance_left': left_hand_to_shoulder_dist,
            'hand_distance_right': right_hand_to_shoulder_dist,
            'shoulder_ear_angle': angle_deg,
            'posture_score': self.posture_score
        }

        # Drawing utilities (optional but nice for visual debugging)
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

    return frame, posture_metrics

