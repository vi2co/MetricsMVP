import cv2
import numpy as np
from ultralytics import YOLO

TRAFFIC_LIGHT_CLASS = 9

RED_RANGE1 = ((0, 50, 50), (10, 255, 255))
RED_RANGE2 = ((170, 50, 50), (180, 255, 255))
YELLOW_RANGE = ((15, 50, 50), (35, 255, 255))
GREEN_RANGE = ((40, 50, 50), (90, 255, 255))

STATE_COLORS = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "unknown": (128, 128, 128),
}


class TrafficSignalDetector:
    def __init__(self, weights="yolo11s.pt"):
        self.model = YOLO(weights)

    def detect(self, frame):
        results = self.model(
            frame,
            classes=[TRAFFIC_LIGHT_CLASS],
            conf=0.15,
            imgsz=640,
            verbose=False,
        )[0]

        signals = []
        for detection in results.boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
            conf = float(detection.conf[0])
            crop = frame[y1:y2, x1:x2]
            state = self._analyze_state(crop)
            signals.append({
                "box": (x1, y1, x2, y2),
                "confidence": conf,
                "state": state,
            })

        return signals

    def _analyze_state(self, crop):
        if crop is None or crop.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        red_mask1 = cv2.inRange(hsv, *RED_RANGE1)
        red_mask2 = cv2.inRange(hsv, *RED_RANGE2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        yellow_mask = cv2.inRange(hsv, *YELLOW_RANGE)
        green_mask = cv2.inRange(hsv, *GREEN_RANGE)

        red_px = cv2.countNonZero(red_mask)
        yellow_px = cv2.countNonZero(yellow_mask)
        green_px = cv2.countNonZero(green_mask)

        total = red_px + yellow_px + green_px
        if total == 0:
            return "unknown"

        if red_px > yellow_px and red_px > green_px:
            return "red"
        if yellow_px > red_px and yellow_px > green_px:
            return "yellow"
        return "green"
