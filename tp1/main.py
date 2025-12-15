import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Detector import detect
from tracker import Tracker


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, "video", "randomball.avi")
    out_vid_path = os.path.join(base_dir, "output.mp4")

    trk = Tracker()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    v_out = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print("Running ...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        centers = detect(frame)

        pred_state = trk.predict()
        px, py = int(pred_state[0]), int(pred_state[1])

        cv2.rectangle(frame, (px - 5, py - 5), (px + 5, py + 5), (0, 0, 255), 2)

        if len(centers) > 0:
            c = (int(centers[0][0][0]), int(centers[0][1][0]))
            cv2.circle(frame, center=c, color=(0, 255, 0), radius=3, thickness=2)
            est_state = trk.update(c)
            ex, ey = int(est_state[0]), int(est_state[1])
            cv2.rectangle(frame, (ex - 5, ey - 5), (ex + 5, ey + 5), (255, 0, 0), 2)

        v_out.write(frame)

    cap.release()
    v_out.release()


if __name__ == "__main__":
    main()
