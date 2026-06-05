import cv2
from deepface import DeepFace
import time

cap = cv2.VideoCapture(0)

prev_time = 0

print("Emotion Detection Started...")
print("Press Q to Quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    frame = cv2.flip(frame, 1)

    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )

        emotion = result[0]['dominant_emotion']

        cv2.putText(
            frame,
            f"Emotion: {emotion}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    except Exception as e:
        print("Error:", e)

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()