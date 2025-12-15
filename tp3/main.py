import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tracker import Tracker


def parse_dets(path):
    d = {}
    with open(path, "r") as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            parts = l.split(",") if "," in l else l.split()
            fr = int(parts[0])
            box = [
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
                float(parts[6]),
            ]
            if fr not in d:
                d[fr] = []
            d[fr].append(box)
    return d


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d_path = os.path.join(base, "ADL-Rundle-6", "det", "Yolov5s", "det.txt")
    i_dir = os.path.join(base, "ADL-Rundle-6", "img1")
    o_txt = os.path.join(base, "tp3", "ADL-Rundle-6.txt")
    o_vid = os.path.join(base, "tp3", "ADL-Rundle-6.mp4")

    dets = parse_dets(d_path)
    trk = Tracker()

    fns = sorted([f for f in os.listdir(i_dir) if f.endswith(".jpg")])
    if not fns:
        return

    im = cv2.imread(os.path.join(i_dir, fns[0]))
    h, w, _ = im.shape
    v_out = cv2.VideoWriter(o_vid, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    t_out = open(o_txt, "w")

    for i, fn in enumerate(tqdm(fns)):
        fid = i + 1
        im = cv2.imread(os.path.join(i_dir, fn))
        res = trk.update(dets.get(fid, []))

        for r in res:
            tid, b = r["id"], r["box"]
            # Draw predicted box (blue)
            if "pred_box" in r:
                pb = r["pred_box"]
                px, py, pw, ph = map(int, pb)
                cv2.rectangle(im, (px, py), (px + pw, py + ph), (255, 0, 0), 2)

            # Draw tracked box (green)
            t_out.write(
                f"{fid},{tid},{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f},1,-1,-1,-1\n"
            )
            ix, iy, iw, ih = map(int, b)
            cv2.rectangle(im, (ix, iy), (ix + iw, iy + ih), (0, 255, 0), 2)
            cv2.putText(
                im,
                str(tid),
                (ix, iy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        v_out.write(im)

    t_out.close()
    v_out.release()


if __name__ == "__main__":
    main()
