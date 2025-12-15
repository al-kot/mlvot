import numpy as np
from KalmanFilter import KalmanFilter


class Tracker:
    def __init__(self):
        self.kf = KalmanFilter(0.1, 1.0, 1.0, 1.0, 0.1, 0.1)

    def predict(self):
        self.kf.predict()
        return self.kf.state

    def update(self, center):
        self.kf.update(np.array(center))
        return self.kf.state
