import cv2
import mediapipe as mp
from deepface import DeepFace
import time
import os

# -----------------------------
# INIT
# -----------------------------
mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to access webcam.")
    exit()

frame_count = 0
prev_time = 0

# Last recognized person
current_name = "Unknown"

print("Face Recognition Started...")
print("Press Q to Quit")

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:

    success, frame = cap.read()

    if not success:
        print("Failed to read frame.")
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb_frame)

    # -----------------------------
    # FPS
    # -----------------------------
    current_time = time.time()

    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0

    prev_time = current_time

    frame_count += 1

    # -----------------------------
    # FACE DETECTION
    # -----------------------------
    if results.detections:

        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)

            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            x = max(0, x)
            y = max(0, y)

            face_img = frame[y:y + bh, x:x + bw]

            # -----------------------------
            # RECOGNITION EVERY 10 FRAMES
            # -----------------------------
            if frame_count % 10 == 0 and face_img.size != 0:

                try:

                    result = DeepFace.find(
                        img_path=face_img,
                        db_path="dataset",
                        model_name="Facenet512",
                        distance_metric="cosine",
                        enforce_detection=False,
                        silent=True
                    )

                    if len(result) > 0 and len(result[0]) > 0:

                        matched_path = result[0].identity.iloc[0]

                        current_name = os.path.basename(
                            os.path.dirname(matched_path)
                        )

                    else:
                        current_name = "Unknown"

                except Exception as e:

                    print("Recognition Error:", e)

                    current_name = "Unknown"

            # -----------------------------
            # DRAW FACE BOX
            # -----------------------------
            cv2.rectangle(
                frame,
                (x, y),
                (x + bw, y + bh),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                current_name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    # -----------------------------
    # FPS DISPLAY
    # -----------------------------
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow("Face Recognition System", frame)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()