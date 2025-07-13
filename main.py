import cv2
from yolov10.detector import YOLOv10Detector
from face_module.face_recognition_module import FaceRecognizer
from emotion_module.emotion_detector import EmotionDetector
from speech_module.speech_to_text import transcribe_speech
from logger.logger import BehaviorLogger
import datetime
import os

def log_emotion(name, emotion):
    os.makedirs("logs", exist_ok=True)
    with open("logs/emotion_log.csv", "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp},{name},{emotion}\n")

def main():
    print("[🚀 System Starting...]")

    # 🎥 Initialize webcam
    cap = cv2.VideoCapture(0)

    # ✅ Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Error: Cannot access webcam")
        return

    # 🧠 Initialize modules
    detector = YOLOv10Detector(model_path="yolov10/yolov10s.pt")
    recognizer = FaceRecognizer(known_faces_dir="faces")
    emotion_model = EmotionDetector("emotion_module/custom_emotion_model.h5")
    logger = BehaviorLogger()

    frame_count = 0
    print("[🟢 Running Real-Time Analysis. Press 'q' to quit.]")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[❌ Error] Failed to capture frame.")
            break

        frame_count += 1

        # 🧑 Face recognition
        names, face_locations = recognizer.recognize(frame)
        print(f"[🎯 Detected Faces] {len(face_locations)}")
        emotion = "Unknown"

        # For each recognized face
        for (top, right, bottom, left), name in zip(face_locations, names):
            # Crop the face
            face_image = frame[top:bottom, left:right]
            if face_image.size > 0:
                try:
                    # 🧠 Detect emotion from the cropped face
                    emotion, emotion_probs = emotion_model.detect_emotion(face_image)
                    print(f"[🎭 Emotion] {name}: {emotion}")
                    log_emotion(name, emotion)
                except Exception as e:
                    print(f"[⚠️ Emotion Detection Error] {e}")

            # 🖼️ Draw rectangle and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}, {emotion}", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 🧠 YOLOv10: Distraction detection
        detections = detector.detect(frame)
        distractions = [d["name"] for d in detections if d["name"] in detector.distraction_list]

        # 🗣️ Speech transcription every 150 frames
        speech = ""
        if frame_count % 150 == 0:
            speech = transcribe_speech()

        # 📋 Display summary (show first name if available, or 'Unknown')
        summary_name = names[0] if names else "Unknown"
        summary_emotion = emotion if names else "Unknown"

        cv2.putText(frame, f"Name: {summary_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Emotion: {summary_emotion}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if distractions:
            cv2.putText(frame, f"Distractions: {', '.join(distractions)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)


        cv2.imshow("Real-Time Behavior Analysis", frame)

        # 📝 Log behavior every 30 frames
        if frame_count % 30 == 0:
            logger.log(name=summary_name, emotion=summary_emotion, speech=speech, distractions=distractions)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[🛑 System Stopped]")

if __name__ == "__main__":
    main()
