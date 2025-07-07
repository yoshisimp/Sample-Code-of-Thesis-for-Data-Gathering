# emotion_detector.py

from keras.models import load_model
import numpy as np
import cv2

class EmotionDetector:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    def predict_emotion(self, face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48)) / 255.0
        reshaped = resized.reshape(1, 48, 48, 1)

        predictions = self.model.predict(reshaped)
        label_index = np.argmax(predictions)
        return self.emotion_labels[label_index], float(np.max(predictions))
