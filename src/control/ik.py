import numpy as np
import mujoco

R_DOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]]) 

def make_ik(m):
    tcp = m.site("tcp").id
    d_ik = mujoco.MjData(m)

    def ik(p_des, q0):
        q = np.array(q0, float)
        for _ in range(300):
            d_ik.qpos[:6] = q
            mujoco.mj_kinematics(m, d_ik); mujoco.mj_comPos(m, d_ik)
            p = d_ik.site_xpos[tcp]; R = d_ik.site_xmat[tcp].reshape(3, 3)
            e = np.r_[p_des - p, 0.5 * sum(np.cross(R[:, i], R_DOWN[:, i]) for i in range(3))]
            if np.linalg.norm(e) < 1e-5:
                break
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d_ik, jp, jr, tcp)
            J = np.vstack([jp[:, :6], jr[:, :6]])
            q = q + J.T @ np.linalg.solve(J @ J.T + 4e-4 * np.eye(6), e)
        if np.linalg.norm(e) > 1e-3:
            print("WARNING: target may be out of reach, IK error", np.linalg.norm(e))
        return q
    return ik