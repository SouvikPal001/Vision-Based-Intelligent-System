import cv2
import time
from human_detection import HumanDetector

def main():
    # -------------------- CONFIG --------------------
    MODEL_PATH = "../models/yolov8s.pt"
    CAMERA_INDEX = 0
    # ------------------------------------------------

    # Initialize detector
    detector = HumanDetector(
        model_path=MODEL_PATH,
        confidence_threshold=0.6,
        stable_frames_required=5
    )

    # Capture real time webcam footage, the index 0 indicates the default laptop camera is used
    cap = cv2.VideoCapture(CAMERA_INDEX)

    # Loop through video frames
    while cap.isOpened():
        # Read a frame from camera
        success, frame = cap.read()  # (bool,img)

        if not success: break  # Detection stopped because camera is not ON/frames not detected/corrupted frames

        # FPS Calculation
        start = time.perf_counter()
        fps = int(1 / (time.perf_counter() - start))
        HUMAN_PRESENT, results = detector.process_frame(frame)

        # Visualize and display the results on the frame
        annotated_frame = results[0].plot()
        status_text = "HUMAN PRESENT" if HUMAN_PRESENT else "NO HUMAN"
        status_color = (0, 255, 0) if HUMAN_PRESENT else (0, 0, 255)

        cv2.putText(
            annotated_frame,
            f"{status_text} | FPS: {int(fps)}",
            (20, 40),  # position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,  # font
            1,  # font scale
            status_color,  # color (BGR)
            2  # thickness
        )
        cv2.imshow(
            "Vision-Based Intelligent System",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break  # Explicitly exited/stopped detection
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
