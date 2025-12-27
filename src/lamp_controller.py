import serial
import time
import cv2
from ultralytics import YOLO

# Arduino serial connection
arduino = serial.Serial('COM3', 9600, timeout=1)  # change COM port
time.sleep(2)

# Load YOLO model
model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

HUMAN_CLASS_ID = 0  # COCO: person
CONF_THRESHOLD = 0.5

human_present = False
last_sent = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)
    human_present = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == HUMAN_CLASS_ID and conf > CONF_THRESHOLD:
                human_present = True

    # Send command only if state changes
    if human_present and last_sent != '1':
        arduino.write(b'1')
        last_sent = '1'
        print("Human detected → Lamp logic active")

    elif not human_present and last_sent != '0':
        arduino.write(b'0')
        last_sent = '0'
        print("No human → Lamp OFF")

    cv2.imshow("Vision", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
