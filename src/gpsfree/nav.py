"""Horizontal navigation filter: accelerometer prediction, VO velocity updates.

State [px, py, vx, vy, bx, by]: position, velocity, accelerometer bias. With
no VO updates this is pure inertial dead reckoning, and a 0.05 m/s^2 bias
(a fraction of a degree of levelling error) is hundreds of metres over a
two-minute flight. That is the baseline every camera method is measured against.
"""
from __future__ import annotations

import numpy as np

CHI2_2_99 = 9.21


class NavEKF:
    def __init__(self, p0, sigma_acc=0.15, sigma_bias_walk=0.002, sigma_vo=0.25, gate=True):
        self.x = np.r_[p0, 0.0, 0.0, 0.0, 0.0].astype(float)
        self.P = np.diag([0.01, 0.01, 0.01, 0.01, 0.1 ** 2, 0.1 ** 2])
        self.sa, self.sb, self.svo, self.gate = sigma_acc, sigma_bias_walk, sigma_vo, gate
        self.accepted = self.rejected = 0

    @property
    def pos(self):
        return self.x[:2]

    @property
    def vel(self):
        return self.x[2:4]

    def predict(self, acc_meas, dt):
        """acc_meas: horizontal world-frame acceleration from the IMU, bias included."""
        F = np.eye(6)
        F[0, 2] = F[1, 3] = dt
        F[0, 4] = F[1, 5] = -0.5 * dt * dt
        F[2, 4] = F[3, 5] = -dt
        a = acc_meas - self.x[4:]
        self.x[:2] += self.x[2:4] * dt + 0.5 * a * dt * dt
        self.x[2:4] += a * dt
        G = np.zeros((6, 2))
        G[0, 0] = G[1, 1] = 0.5 * dt * dt
        G[2, 0] = G[3, 1] = dt
        Q = G @ G.T * self.sa ** 2
        Q[4, 4] = Q[5, 5] = (self.sb ** 2) * dt
        self.P = F @ self.P @ F.T + Q

    def update_velocity(self, v_meas):
        H = np.zeros((2, 6))
        H[0, 2] = H[1, 3] = 1.0
        r = v_meas - self.x[2:4]
        S = H @ self.P @ H.T + np.eye(2) * self.svo ** 2
        if self.gate and float(r @ np.linalg.solve(S, r)) > CHI2_2_99:
            self.rejected += 1
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ r
        self.P = (np.eye(6) - K @ H) @ self.P
        self.accepted += 1
        return True
