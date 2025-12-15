import numpy as np
from KalmanFilter import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou_calc(b1, b2):
    xx1 = np.maximum(b1[0], b2[0])
    yy1 = np.maximum(b1[1], b2[1])
    xx2 = np.minimum(b1[0] + b1[2], b2[0] + b2[2])
    yy2 = np.minimum(b1[1] + b1[3], b2[1] + b2[3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    union = (b1[2] * b1[3]) + (b2[2] * b2[3]) - inter
    return inter / union if union > 0 else 0


class Tracker:
    def __init__(self, miss_max=5, iou_th=0.3):
        self.trks = []
        self.nid = 1
        self.miss_max = miss_max
        self.iou_th = iou_th

    def update(self, dets):
        # Predict
        for trk in self.trks:
            trk["kf"].predict()
            px, py = trk["kf"].state[0], trk["kf"].state[1]
            w, h = trk["box"][2], trk["box"][3]
            trk["pred_box"] = [px - w / 2, py - h / 2, w, h]

        n_trks = len(self.trks)
        n_dets = len(dets)

        iou_mat = np.zeros((n_trks, n_dets))
        for i, trk in enumerate(self.trks):
            pbox = trk.get("pred_box", trk["box"])
            for j, det in enumerate(dets):
                iou_mat[i, j] = iou_calc(pbox, det[:4])

        if n_trks > 0 and n_dets > 0:
            r_idx, c_idx = linear_sum_assignment(-iou_mat)
        else:
            r_idx, c_idx = [], []

        unm_trks = set(range(n_trks))
        unm_dets = set(range(n_dets))

        matched = []
        for r, c in zip(r_idx, c_idx):
            if iou_mat[r, c] >= self.iou_th:
                unm_trks.discard(r)
                unm_dets.discard(c)

                det_box = dets[c][:4]
                cx = det_box[0] + det_box[2] / 2
                cy = det_box[1] + det_box[3] / 2

                self.trks[r]["kf"].update(np.array([cx, cy]))
                self.trks[r]["box"] = det_box
                self.trks[r]["conf"] = dets[c][4]
                self.trks[r]["miss"] = 0
                matched.append(self.trks[r])

        new_trks = []
        for r in unm_trks:
            self.trks[r]["miss"] += 1
            if self.trks[r]["miss"] <= self.miss_max:
                new_trks.append(self.trks[r])
        new_trks.extend(matched)

        for c in unm_dets:
            det_box = dets[c][:4]
            cx = det_box[0] + det_box[2] / 2
            cy = det_box[1] + det_box[3] / 2

            kf = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)
            kf.state = np.array([cx, cy, 0.0, 0.0])

            new_trks.append(
                {
                    "id": self.nid,
                    "box": det_box,
                    "kf": kf,
                    "conf": dets[c][4],
                    "miss": 0,
                }
            )
            self.nid += 1

        self.trks = new_trks
        return self.trks
