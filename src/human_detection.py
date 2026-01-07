from ultralytics import YOLO

class HumanDetector:
    def __init__(
        self,
        model_path,
        confidence_threshold=0.6,
        stable_frames_required=5
    ):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.stable_frames_required = stable_frames_required

        self.frame_count = 0
        self.human_present = False

    def process_frame(self, frame):
        results = self.model(frame, verbose=False)
        detections = results[0].boxes

        detected = False

        if detections is not None:
            for box in detections:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if self.model.names[cls_id] == "person" and conf >= self.confidence_threshold:
                    detected = True
                    break

        if detected:
            self.frame_count += 1
        else:
            self.frame_count = 0
            self.human_present = False

        if self.frame_count >= self.stable_frames_required:
            self.human_present = True

        return self.human_present, results