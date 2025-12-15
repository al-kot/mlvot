# Multi-Object Tracking

This project implements various Multi-Object Tracking (MOT) algorithms.

## Setup

Place the extracted `ADL-Rundle-6` folder and `reid_osnet_x025_market1501.onnx`
model in the root of the project:
[link](https://www.swisstransfer.com/d/3cbb37ea-8da0-428e-af9b-c58deaba50bf)

```bash
.
├── ADL-Rundle-6
│   ├── det
│   ├── gt
│   ├── img1
│   └── seqinfo.ini
├── pyproject.toml
├── README.md
├── reid_osnet_x025_market1501.onnx
├── report
│   ├── report.pdf
│   └── report.typ
├── tp1
│   ├── Detector.py
│   ├── KalmanFilter.py
...
```

## Structure
- `tp1/`: Single Object Tracking.
- `tp2/`: IoU-based Tracker.
- `tp3/`: Kalman-Guided IoU Tracker.
- `tp4/`: Appearance-Aware (ReID) Tracker.
- `report/`: Project report.

## Usage

Ensure you have the virtual environment set up with [uv](https://docs.astral.sh/uv/getting-started/installation/):
```bash
uv sync
```

### TP 1: Single Object Tracker
```bash
uv run tp1/main.py
```
Results: `tp1/output.mp4`

### TP 2: IoU Tracker
```bash
uv run tp2/main.py
```
Results: `tp2/ADL-Rundle-6.mp4`, `tp2/ADL-Rundle-6.txt`

### TP 3: Kalman Tracker
```bash
uv run tp3/main.py
```
Results: `tp3/ADL-Rundle-6.mp4`, `tp3/ADL-Rundle-6.txt`

### TP 4: ReID Tracker
```bash
uv run tp4/main.py
```
Results: `tp4/ADL-Rundle-6.mp4`, `tp4/ADL-Rundle-6.txt`
