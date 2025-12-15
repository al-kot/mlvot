import cv2
import numpy as np


class ReID:
    def __init__(self, model_path):
        self.net = cv2.dnn.readNet(model_path)

    def extract(self, img, box):
        x, y, w, h = map(int, box)
        hi, wi = img.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, wi - x)
        h = min(h, hi - y)

        if w <= 0 or h <= 0:
            return None

        crop = img[y : y + h, x : x + w]

        crop = cv2.resize(crop, (64, 128))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = crop.astype(np.float32) / 255.0
        crop = (crop - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225]
        )

        blob = np.transpose(crop, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        self.net.setInput(blob)
        feat = self.net.forward()

        feat = feat.flatten()
        n = np.linalg.norm(feat)
        if n > 0:
            feat /= n
        return feat
