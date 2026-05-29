import cv2
import mediapipe as mp
import time

# MediaPipe setup
mp_face_detection = mp.solutions.face_detection

# Open webcam
cap = cv2.VideoCapture(0)

# FPS variables
prev_time = 0

# Face Detection Model
with mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
) as face_detection:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("Failed to access webcam.")
            break

        # Mirror effect
        frame = cv2.flip(frame, 1)

        # Get frame dimensions
        h, w, c = frame.shape

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = face_detection.process(rgb_frame)

        # Face detections
        if results.detections:

            for detection in results.detections:

                # Bounding box
                bbox = detection.location_data.relative_bounding_box

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Confidence score
                confidence = int(detection.score[0] * 100)

                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2
                )

                # Display confidence
                cv2.putText(
                    frame,
                    f"Face {confidence}%",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        # FPS calculation
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Display FPS
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        # Show output
        cv2.imshow("AI Face Detection System", frame)

        # Exit on Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
