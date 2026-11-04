"""State-space joint controller: pole-placement feedback + Luenberger observer + computed torque."""
import mujoco
import numpy as np
from config import CTRL_POLES, DT, ENC_NOISE_STD, JOINTS, OBS_POLES, TORQUE_LIMIT


#  CONTROLLER 
class StateSpaceController:
#Reads simulated joint positions with added encoder noise.
#Uses a Luenberger observer to estimate joint positions and velocities.
#Calculates feedback corrections when the estimated motion differs from the reference.
#Uses the MuJoCo mass matrix and dynamic terms to calculate joint torques.
#Clips the commanded torques to configured limits.

    def __init__(self, model, data, rng):
        self.m, self.d, self.rng = model, data, rng
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in JOINTS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jids])
        p1, p2 = CTRL_POLES
        self.k1, self.k2 = p1 * p2, -(p1 + p2)
        o1, o2 = OBS_POLES
        self.l1, self.l2 = -(o1 + o2), o1 * o2
        self.Mfull = np.zeros((model.nv, model.nv))
        self.reset(data.qpos[self.qadr])

    def reset(self, q):
        self.qh = np.array(q, float)
        self.vh = np.zeros(6)
        self.acc = np.zeros(6)

    def step(self, q_ref, v_ref, a_ref):
        d = self.d
        y = d.qpos[self.qadr] + self.rng.normal(0, ENC_NOISE_STD, 6)     # measurement
        innov = y - self.qh
        qh = self.qh + DT * (self.vh + self.l1 * innov)                  # observer
        vh = self.vh + DT * (self.acc + self.l2 * innov)
        self.qh, self.vh = qh, vh
        acc = a_ref - self.k1 * (qh - q_ref) - self.k2 * (vh - v_ref)    # state feedback
        self.acc = acc
        if hasattr(d, "qM"):                                             # MuJoCo <= 3.2
            mujoco.mj_fullM(self.m, self.Mfull, d.qM)
        else:                                                            # newer releases
            mujoco.mj_fullM(self.m, d, self.Mfull)
        M = self.Mfull[np.ix_(self.dadr, self.dadr)]
        tau = M @ acc + d.qfrc_bias[self.dadr] - d.qfrc_passive[self.dadr]
        d.qfrc_applied[self.dadr] = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)
