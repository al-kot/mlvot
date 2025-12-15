import numpy as np
from numpy._typing import NDArray


class KalmanFilter:

    def __init__(
        self,
        dt: float,
        u_x: float,
        u_y: float,
        std_acc: float,
        x_std_meas: float,
        y_std_meas: float,
    ):
        self.input = np.array([u_x, u_y])
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        self.A = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.B = np.array(
            [
                [(dt**2) / 2, 0.0],
                [0.0, (dt**2) / 2],
                [dt, 0.0],
                [0.0, dt],
            ]
        )
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        self.Q = (
            np.array(
                [
                    [(dt**4) / 4, 0.0, (dt**3) / 2, 0.0],
                    [0.0, (dt**4) / 4, 0.0, (dt**3) / 2],
                    [(dt**3) / 2, 0.0, dt**2, 0.0],
                    [0.0, (dt**3) / 2, 0.0, dt**2],
                ]
            )
            * std_acc**2
        )
        self.R = np.array(
            [
                [x_std_meas**2, 0.0],
                [0.0, y_std_meas**2],
            ]
        )
        self.P = np.identity(4)

    def predict(self):
        self.state = self.A @ self.state + self.B @ self.input
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, zk: NDArray):
        Sk = self.H @ self.P @ self.H.T + self.R
        Kk = self.P @ self.H.T @ np.linalg.inv(Sk)
        self.state = self.state + Kk @ (zk - self.H @ self.state)
        KkH = Kk @ self.H
        self.P = (np.identity(np.max(KkH.shape)) - KkH) @ self.P
