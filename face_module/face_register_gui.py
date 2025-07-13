import cv2
import os

name = input("Enter your name: ")
save_dir = f"face_module/faces/{name}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0
print("Press SPACE to capture face image. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Register Face", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC to quit
        break
    elif key == 32:  # SPACE to capture
        img_path = os.path.join(save_dir, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        count += 1

cap.release()
cv2.destroyAllWindows()
