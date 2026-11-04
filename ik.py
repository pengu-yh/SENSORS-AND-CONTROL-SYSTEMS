"""Damped-least-squares 6-D inverse kinematics on the TCP site (PBVS core)."""
import mujoco
import numpy as np
from config import JOINTS


#IK (PBVS core)
class IK:
    """Damped-least-squares 6-D IK on the TCP site, on a private MjData."""

    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in JOINTS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.lo = np.array([model.jnt_range[j, 0] if model.jnt_limited[j] else -2 * np.pi for j in jids])
        self.hi = np.array([model.jnt_range[j, 1] if model.jnt_limited[j] else 2 * np.pi for j in jids])
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros((3, model.nv))

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def solve(self, pos, R_des, q_seed, iters=300, lam=0.05):
        q = np.array(q_seed, float)
        for _ in range(iters):
            p, R = self.fk(q)
            e_p = pos - p
            e_r = 0.5 * sum(np.cross(R[:, i], R_des[:, i]) for i in range(3))
            if np.linalg.norm(e_p) < 2e-4 and np.linalg.norm(e_r) < 2e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.sid)
            J = np.vstack([self.jp[:, self.dadr], self.jr[:, self.dadr]])
            e = np.concatenate([e_p, e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
            n = np.linalg.norm(dq)
            if n > 0.2:
                dq *= 0.2 / n
            q = np.clip(q + dq, self.lo, self.hi)
        p, R = self.fk(q)
        e_r = 0.5 * sum(np.cross(R[:, i], R_des[:, i]) for i in range(3))
        ok = np.linalg.norm(pos - p) < 1e-3 and np.linalg.norm(e_r) < 2e-2
        return q, ok

    def line(self, p0, p1, R_des, q_seed, n=12):
        """Joint waypoints along a straight Cartesian line (None if unreachable)."""
        q, ok = self.solve(p0, R_des, q_seed)
        if not ok:
            return None
        qs = [q]
        for s in np.linspace(0, 1, n)[1:]:
            q, ok = self.solve(p0 + (p1 - p0) * s, R_des, qs[-1])
            if not ok:
                return None
            qs.append(q)
        return np.array(qs)
