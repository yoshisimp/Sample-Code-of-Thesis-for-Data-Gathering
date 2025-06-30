import cv2
import torch
from face_recognition_module import recognize_faces
from emotion_detector import EmotionRecognizer
from speech_to_text import transcribe_speech
from logger import Logger
from summary_report import generate_emotion_chart
from datetime import datetime
import os

# Load YOLOv10 model
model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov10s.pt')

# Load Emotion Recognizer
emotion_model = EmotionRecognizer('models/emotion_model.h5')

# Logger
logger = Logger()

# Distraction objects
DISTRACTION_OBJECTS = ["cell phone", "laptop", "tv", "bottle", "cup", "bowl", "sandwich", "remote", "book"]

# Start Webcam
cap = cv2.VideoCapture(0)

frame_count = 0
session_emotions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Detect objects with YOLOv10
    results = model(frame)
    detected = results.pandas().xyxy[0]
    distraction_found = any(obj in DISTRACTION_OBJECTS for obj in detected['name'])

    # Face recognition
    face_names = recognize_faces(frame)

    # Emotion recognition
    emotion = emotion_model.detect_emotion(frame)
    session_emotions.append(emotion)

    # Log results
    logger.log(frame, emotion, face_names, distraction_found)

    # Display
    for i, name in enumerate(face_names):
        cv2.putText(frame, f"{name}", (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Emotion: {emotion}", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imshow("AI Counseling", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save report
generate_emotion_chart(session_emotions)
