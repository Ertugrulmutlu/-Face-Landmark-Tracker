import cv2 as cv
import numpy as np
class Detector:
    def __init__(self,proto_path,weights_path):
        self.proto_path   = proto_path
        self.weights_path = weights_path
        self.net = None
    def load_face_detector(self):
        self.net = cv.dnn.readNetFromCaffe(self.proto_path, self.weights_path)

    def detect_faces_dnn(self, frame, conf_thr=0.5):
        """Return list of (x, y, w, h) boxes."""
        h, w = frame.shape[:2]
        blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0, (300, 300),
                                    (104.0, 177.0, 123.0), False, False)
        self.net.setInput(blob)
        detections = self.net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence >= conf_thr:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w-1, x2), min(h-1, y2)
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes