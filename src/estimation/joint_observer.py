import numpy as np


class JointObserver:
    """Kalman filter for the 6 joints. State x = [q (6); qd (6)].  Encoders measure q only.

    Model (after the controller's computed-torque cancels the dynamics, qdd = a_cmd + noise):
        x_next = A x + B a_cmd + w      A = [[I, dt I], [0, I]]   B = [[dt^2/2 I], [dt I]]
        z      = C x + v                C = [I, 0]
    """

    def __init__(self, dt, encoder_std, accel_std):
        n = 6
        I, Z = np.eye(n), np.zeros((n, n))
        self.A = np.block([[I, dt * I], [Z, I]])
        self.B = np.vstack([0.5 * dt**2 * I, dt * I])
        self.C = np.hstack([I, Z])
        self.Q = accel_std**2 * (self.B @ self.B.T)          # process noise: unmodelled acceleration
        self.R = encoder_std**2 * I                           # measurement noise: encoder
        self.encoder_std = encoder_std
        self.x, self.P = None, None

    def step(self, z, a_cmd=None):
        """One cycle: predict with the commanded acceleration, correct with the encoder reading z.
        Returns (q_hat, qd_hat)."""
        z = np.asarray(z, float)
        if self.x is None:                                    # start at the first reading
            self.x = np.concatenate([z, np.zeros(6)])
            self.P = np.diag(np.r_[np.full(6, self.encoder_std**2), np.full(6, 0.5**2)])
            return self.x[:6].copy(), self.x[6:].copy()
        a = np.zeros(6) if a_cmd is None else np.asarray(a_cmd, float)
        self.x = self.A @ self.x + self.B @ a                 # predict
        self.P = self.A @ self.P @ self.A.T + self.Q
        S = self.C @ self.P @ self.C.T + self.R               # update
        K = np.linalg.solve(S, self.C @ self.P).T
        self.x = self.x + K @ (z - self.C @ self.x)
        IKC = np.eye(12) - K @ self.C
        self.P = IKC @ self.P @ IKC.T + K @ self.R @ K.T     # Joseph form
        return self.x[:6].copy(), self.x[6:].copy()

    @property
    def sigma(self):
        """1-sigma uncertainty of the 12 states (for the +-3 sigma plots)."""
        return np.sqrt(np.diag(self.P))