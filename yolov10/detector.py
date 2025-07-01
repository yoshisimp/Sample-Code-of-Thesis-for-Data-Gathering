
import os
from ultralytics import YOLO
import cv2

class YOLOv10Detector:
    def __init__(self, model_path='yolov10/yolov10s.pt', conf_threshold=0.5):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[ERROR] YOLOv10 model not found at {model_path}")
        self.model = YOLO(model_path)

        # Customize this list based on your thesis
        self.distraction_list = [
            "cell phone", "laptop", "tv", "bottle", "cup",
            "bowl", "sandwich", "remote", "book"
        ]

    def detect(self, frame):
        """Run YOLOv10 detection on the frame and return list of dicts with 'name' and 'confidence'."""
        results = self.model.predict(source=frame, verbose=False, stream=False)
        detections = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                confidence = float(box.conf[0])

                detections.append({
                    "name": class_name,
                    "confidence": confidence
                })

        return detections
