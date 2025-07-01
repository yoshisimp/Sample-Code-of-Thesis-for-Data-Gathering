# emotion_module/emotion_detection_module.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model

class EmotionDetector:
    def __init__(self, model_path="emotion_model.h5"):
        self.model = load_model(model_path)
        self.emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def detect_emotion(self, face_image):
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))
        predictions = self.model.predict(reshaped, verbose=0)
        max_index = int(np.argmax(predictions))
        emotion = self.emotion_labels[max_index]
        return emotion, predictions[0]
