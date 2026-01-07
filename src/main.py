import cv2
import time
import serial
from human_detection import HumanDetector
import config


def main():
    # -------- Serial --------
    arduino = serial.Serial(
        config.SERIAL_PORT,
        config.BAUD_RATE,
        timeout=1
    )
    time.sleep(2)

    # FORCE SAFE START
    arduino.write(b'0')
    last_sent = '0'

    # -------- Detector -------
    detector = HumanDetector(
        model_path=config.MODEL_PATH,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        stable_frames_required=config.STABLE_FRAMES_REQUIRED
    )

    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        delta = current_time - prev_time
        fps = int(1 / delta) if delta > 0 else 0
        prev_time = current_time

        human_present, results = detector.process_frame(frame)

        # SERIAL: SEND ONLY ON CHANGE
        if human_present and last_sent != '1':
            arduino.write(b'1')
            last_sent = '1'
        elif not human_present and last_sent != '0':
            arduino.write(b'0')
            last_sent = '0'

        annotated = results[0].plot()
        label = "HUMAN PRESENT" if human_present else "NO HUMAN"
        color = (0, 255, 0) if human_present else (0, 0, 255)

        cv2.putText(
            annotated,
            f"{label} | FPS: {fps}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        cv2.imshow(config.WINDOW_NAME, annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    arduino.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
