import numpy as np
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
        n_trks = len(self.trks)
        n_dets = len(dets)

        iou_mat = np.zeros((n_trks, n_dets))
        for i, trk in enumerate(self.trks):
            for j, det in enumerate(dets):
                iou_mat[i, j] = iou_calc(trk["box"], det[:4])

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
                self.trks[r]["box"] = dets[c][:4]
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
            new_trks.append(
                {"id": self.nid, "box": dets[c][:4], "conf": dets[c][4], "miss": 0}
            )
            self.nid += 1

        self.trks = new_trks
        return self.trks
