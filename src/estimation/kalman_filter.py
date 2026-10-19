
import numpy as np


class KalmanFilter:
    """12-state linear Kalman Filter for a 6-DOF robot arm."""

    def __init__(self, dt, encoder_std=0.005, accel_std=0.5):
        self.n = 6
        self.dt = float(dt)

        if self.dt <= 0:
            raise ValueError("dt must be positive")

        # State: [q1..q6, dq1..dq6]
        self.x = np.zeros(12)

        # Initial estimation uncertainty
        self.P = np.eye(12)
        self.P[:6, :6] *= 0.01
        self.P[6:, 6:] *= 1.0

        # Discrete constant-velocity motion model
        self.A = np.eye(12)
        self.A[:6, 6:] = self.dt * np.eye(6)

        # Encoder measures joint positions only
        self.C = np.hstack((np.eye(6), np.zeros((6, 6))))

        # Measurement noise covariance
        self.R = (encoder_std ** 2) * np.eye(6)

        # Process noise from unmodelled joint acceleration
        G = np.vstack((
            0.5 * self.dt**2 * np.eye(6),
            self.dt * np.eye(6)
        ))
        self.Q = (accel_std ** 2) * (G @ G.T)

        self.initialized = False

    def initialize(self, q_measured):
        """Start the filter at the first encoder measurement."""
        q = np.asarray(q_measured, dtype=float).reshape(6)
        self.x[:6] = q
        self.x[6:] = 0.0
        self.initialized = True

    def predict(self):
        """Predict state and uncertainty before new measurements."""
        if not self.initialized:
            raise RuntimeError("Initialize KF before prediction")

        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, q_measured):
        """Correct the estimate using noisy joint encoder readings."""
        if not self.initialized:
            self.initialize(q_measured)
            return self.get_state()

        y = np.asarray(q_measured, dtype=float).reshape(6)

        # Innovation: measurement - expected measurement
        innovation = y - self.C @ self.x

        # Innovation covariance
        S = self.C @ self.P @ self.C.T + self.R

        # Kalman gain (solve is preferred to explicit inverse)
        K = np.linalg.solve(S, self.C @ self.P).T

        # Correct estimated state
        self.x = self.x + K @ innovation

        # Joseph-form covariance update for numerical stability
        I = np.eye(12)
        IKC = I - K @ self.C
        self.P = IKC @ self.P @ IKC.T + K @ self.R @ K.T

        return self.get_state()

    def step(self, q_measured):
        """Run one prediction + correction cycle."""
        if not self.initialized:
            return self.update(q_measured)

        self.predict()
        return self.update(q_measured)

    def get_state(self):
        """Return estimated joint positions and velocities."""
        q_hat = self.x[:6].copy()
        qd_hat = self.x[6:].copy()
        return q_hat, qd_hat
