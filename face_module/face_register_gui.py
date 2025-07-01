import cv2
import os

def register_face(name, save_dir="face_module/faces"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cap = cv2.VideoCapture(0)
    print(f"[INFO] Capturing face for '{name}'. Press 's' to save, or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            img_path = os.path.join(save_dir, f"{name}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"[INFO] Face saved to {img_path}")
            break
        elif key == ord('q'):
            print("[INFO] Registration canceled.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    user_name = input("Enter your name: ").strip()
    if user_name:
        register_face(user_name)

