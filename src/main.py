import cv2
import time
import serial

from human_detection import HumanDetector
import config


def main():
    # -------- Serial Initialization --------
    arduino = serial.Serial(
        config.SERIAL_PORT,
        config.BAUD_RATE,
        timeout=1
    )
    time.sleep(2)
    last_sent = None

    # -------- Detector Initialization -------
    detector = HumanDetector(
        model_path=config.MODEL_PATH,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        stable_frames_required=config.STABLE_FRAMES_REQUIRED
    )

    # -------- Camera Initialization ---------
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    prev_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # FPS calculation
        current_time = time.time()
        fps = int(1 / (current_time - prev_time))
        prev_time = current_time

        # Human detection
        human_present, results = detector.process_frame(frame)

        # Serial communication (send only on change)
        if human_present and last_sent != '1':
            arduino.write(b'1')
            last_sent = '1'

        elif not human_present and last_sent != '0':
            arduino.write(b'0')
            last_sent = '0'

        # Visualization
        annotated_frame = results[0].plot()
        status_text = "HUMAN PRESENT" if human_present else "NO HUMAN"
        status_color = (0, 255, 0) if human_present else (0, 0, 255)

        cv2.putText(
            annotated_frame,
            f"{status_text} | FPS: {fps}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            status_color,
            2
        )

        cv2.imshow(config.WINDOW_NAME, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    arduino.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
