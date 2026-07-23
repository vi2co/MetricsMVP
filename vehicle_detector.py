import cv2
from ultralytics import YOLO

VEHICLE_CLASSES = [2, 3, 5, 7]


class VehicleDetector:
    def __init__(self, weights="yolo11s.pt"):
        self.model = YOLO(weights)

    def detect(self, frame):
        return self.model(
            frame,
            classes=VEHICLE_CLASSES,
            conf=0.15,
            imgsz=960,
            verbose=False,
        )[0]

    def track(self, frame):
        """
        Detect and track vehicles across frames.

        Uses the built-in ByteTrack tracker so each vehicle keeps a
        stable ID for its lifetime in the video.
        """
        return self.model.track(
            frame,
            classes=VEHICLE_CLASSES,
            conf=0.15,
            imgsz=960,
            persist=True,
            verbose=False,
        )[0]

    def class_name(self, class_id):
        return self.model.names[class_id]