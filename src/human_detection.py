import cv2
from ultralytics import YOLO
import time

# Load the YOLOv8 model
model = YOLO('../models/yolov8s.pt')

# Capture real time webcam footage, the index 0 indicates the default laptop camera is used
cap = cv2.VideoCapture(0)

# Loop through video frames
while cap.isOpened():
    #Read a frame from camera
    success, frame = cap.read()

    if success:
        start = time.perf_counter()
        # Run YOLOv8 inference on the frame
        results = model(frame)

        end = time.perf_counter()
        total_time = end - start
        fps = 1 / total_time

        # Visualize and display the results on the frame
        annotated_frame = results[0].plot()
        cv2.putText(
            annotated_frame,
            f"FPS: {int(fps)}",
            (20, 40),  # position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,  # font
            1,  # font scale
            (0, 255, 0),  # color (BGR)
            2  # thickness
        )
        cv2.imshow(
            "YOLOv8 Inference",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break   #Explicitly exited/stopped detection
    else:
        break   #Detection stopped because camera is not on


cap.release()
cv2.destroyAllWindows()
