import face_recognition
import cv2
import os
import numpy as np

class FaceRecognizer:
    def __init__(self, known_faces_dir="faces"):
        self.known_faces_dir = known_faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()


    def load_known_faces(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        faces_dir = os.path.join(base_dir, self.known_faces_dir)
        for name in os.listdir(faces_dir):
            person_dir = os.path.join(faces_dir, name)
            if os.path.isdir(person_dir):
                for img_name in os.listdir(person_dir):
                    img_path = os.path.join(person_dir, img_name)
                    image = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(name)
                        print(f"[✅ Loaded] {name} from {img_path}")
                    else:
                        print(f"[❌ No Face Found] in {img_path}")


    def recognize(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        valid_names = []
        valid_locations = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            if not isinstance(face_location, (tuple, list)) or len(face_location) != 4:
                print(f"⚠️ Skipping invalid face_location: {face_location}")
                continue

            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            valid_names.append(name)
            valid_locations.append(face_location)

        return valid_names, valid_locations
