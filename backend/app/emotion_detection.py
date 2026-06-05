import cv2
import mediapipe as mp
import tensorflow.keras
from deepface import DeepFace
import time

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to access webcam.")
    exit()

prev_time = 0
current_emotion = "Detecting..."
current_emotion_confidence = 0
last_emotion_time = 0
emotion_interval = 1
emotion_history = []

print("Emotion Detection Started...")
print("Press Q to Quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time
    fps = min(int(fps), 60)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + bw + padding)
            y2 = min(h, y + bh + padding)

            face_img = frame[y1:y2, x1:x2]

            if time.time() - last_emotion_time > emotion_interval and face_img.size != 0:
                try:
                    result = DeepFace.analyze(
                        face_img,
                        actions=["emotion"],
                        enforce_detection=False,
                        silent=True
                    )

                    emotions = result[0]["emotion"]
                    detected_emotion = result[0]["dominant_emotion"]
                    confidence = emotions[detected_emotion]

                    if confidence < 55:
                        detected_emotion = "neutral"

                    emotion_history.append(detected_emotion)

                    if len(emotion_history) > 3:
                        emotion_history.pop(0)

                    current_emotion = max(
                        set(emotion_history),
                        key=emotion_history.count
                    )

                    current_emotion_confidence = confidence
                    last_emotion_time = time.time()

                except Exception as e:
                    print("Emotion Error:", e)
                    current_emotion = "Unknown"
                    current_emotion_confidence = 0

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    cv2.putText(
        frame,
        f"Emotion: {current_emotion} ({current_emotion_confidence:.1f}%)",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 0),
        2
    )

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()