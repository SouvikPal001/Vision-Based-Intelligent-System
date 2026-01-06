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

        self.person_frame_count = 0
        self.human_present = False

    def process_frame(self, frame):
        results = self.model(frame, verbose=False)
        detections = results[0].boxes

        person_detected_this_frame = False

        if detections is not None:
            for box in detections:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[cls_id]

                if class_name == "person" and confidence >= self.confidence_threshold:
                    person_detected_this_frame = True
                    break

        # Stability logic
        if person_detected_this_frame:
            self.person_frame_count += 1
        else:
            self.person_frame_count = 0
            self.human_present = False

        if self.person_frame_count >= self.stable_frames_required:
            self.human_present = True

        return self.human_present, results
