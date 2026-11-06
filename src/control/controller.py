import numpy as np
import mujoco
from scipy.linalg import solve_discrete_are


class JointController:
    """Computed-torque + LQR joint controller. Works only from the state ESTIMATE (q_hat, qd_hat)."""

    def __init__(self, m, dt, q_w=2500.0, qd_w=1.0, r=1.0):
        self.m = m
        self.scratch = mujoco.MjData(m)                       # private copy: the model evaluated at the estimate
        self._M = np.zeros((m.nv, m.nv))
        self.limit = m.actuator_ctrlrange[:6, 1].copy()       # torque limits [Nm]

        # one joint after cancelling the dynamics is a double integrator: x = [angle error, speed error]
        self.Ad = np.array([[1.0, dt], [0.0, 1.0]])
        self.Bd = np.array([[0.5 * dt**2], [dt]])
        Q = np.diag([q_w, qd_w])                              # cost on position / velocity error
        R = np.array([[r]])                                   # cost on acceleration effort
        P = solve_discrete_are(self.Ad, self.Bd, Q, R)        # minimum cost-to-go matrix
        self.K = np.linalg.solve(R + self.Bd.T @ P @ self.Bd, self.Bd.T @ P @ self.Ad)   # shape (1, 2)
        self.limit = m.actuator_ctrlrange[:6, 1].copy()       # torque limits [Nm]
        self.last_a = np.zeros(6)                            

    def _mass_matrix(self):
        try:
            mujoco.mj_fullM(self.m, self.scratch, self._M)    # newer MuJoCo
        except TypeError:
            mujoco.mj_fullM(self.m, self._M, self.scratch.qM) # older MuJoCo
        return self._M[:6, :6]

    def compute(self, q_hat, qd_hat, q_ref, qd_ref=None):
        """Returns the 6 joint torques [Nm]."""
        if qd_ref is None:
            qd_ref = np.zeros(6)
        q_hat, qd_hat = np.asarray(q_hat, float), np.asarray(qd_hat, float)

        # 1. LQR: desired joint acceleration from the estimated errors
        a = -(self.K[0, 0] * (q_hat - q_ref) + self.K[0, 1] * (qd_hat - qd_ref))
        self.last_a = a                                       

        # 2. model at the estimated state: mass matrix M and gravity+Coriolis bias h
        self.scratch.qpos[:6] = q_hat
        self.scratch.qvel[:6] = qd_hat
        mujoco.mj_forward(self.m, self.scratch)
        M = self._mass_matrix()
        h = self.scratch.qfrc_bias[:6]

        # 3. computed torque: M*a + h makes qdd = a, then respect the motor limits
        return np.clip(M @ a + h, -self.limit, self.limit)


def make_controller(m, q0=None, dt=None, **kw):
    return JointController(m, dt if dt is not None else m.opt.timestep, **kw)
