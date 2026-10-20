
import numpy as np


class JointController:
    def __init__(self, m, kp=0.2, kd=0.02):
        self.m = m
        self.kp = kp
        self.kd = kd
        self.lo = m.actuator_ctrlrange[:6, 0].copy()
        self.hi = m.actuator_ctrlrange[:6, 1].copy()

    def compute(self, q, qd, q_ref, qd_ref=None):
        if qd_ref is None:
            qd_ref = np.zeros(6)

        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        q_ref = np.asarray(q_ref, dtype=float)
        qd_ref = np.asarray(qd_ref, dtype=float)

        position_error = q_ref - q
        velocity_error = qd_ref - qd

        corrected_ref = q_ref + self.kp * position_error + self.kd * velocity_error

        return np.clip(corrected_ref, self.lo, self.hi)


def make_controller(m, q0=None, Qc=None, Rc=None):
    return JointController(m)
