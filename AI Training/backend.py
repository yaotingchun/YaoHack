from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import mediapipe as mp
import io
from google.cloud import vision

app = FastAPI()

# Initialize Google Cloud Vision
client = vision.ImageAnnotatorClient()

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

@app.post("/analyze_posture/")
async def analyze_posture(file: UploadFile = File(...)):
    image_data = await file.read()
    np_image = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Convert frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    posture_feedback = "Good posture"
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

        if abs(left_shoulder - right_shoulder) > 0.1:
            posture_feedback = "Hunching detected. Stand straight!"

    return {"feedback": posture_feedback}

@app.post("/analyze_expression/")
async def analyze_expression(file: UploadFile = File(...)):
    image_data = await file.read()
    image = vision.Image(content=image_data)

    response = client.face_detection(image=image)
    faces = response.face_annotations

    expression_feedback = "Confident expression detected!"
    if faces:
        face = faces[0]
        if face.sorrow_likelihood >= 3 or face.anger_likelihood >= 3:
            expression_feedback = "You look nervous. Try to relax!"

    return {"feedback": expression_feedback}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)