import sys; sys.path.insert(0, ".")
import numpy as np, mujoco
from src.control.controller import make_controller
from src.estimation.joint_observer import JointObserver

m = mujoco.MjModel.from_xml_path("robot_model/pick_scene.xml"); d = mujoco.MjData(m)
dt = m.opt.timestep; ENC_STD = 5e-4; ctrl = make_controller(m)

for accel_std in (0.3, 1.0, 3.0, 10.0):
    rng = np.random.default_rng(0)
    mujoco.mj_resetDataKeyframe(m, d, 0); q0 = d.qpos[:6].copy()
    obs = JointObserver(dt, ENC_STD, accel_std)
    q_ref = q0.copy(); q_ref[1] += 0.4; q_ref[2] -= 0.4
    e_q, e_qd, peak, inside = [], [], [], []
    for k in range(3000):
        z = d.qpos[:6] + rng.normal(0, ENC_STD, 6)            # the ONLY read of true q for control
        q_hat, qd_hat = obs.step(z, ctrl.last_a)              # observer gets the last commanded acceleration
        d.ctrl[:6] = ctrl.compute(q_hat, qd_hat, q_ref)
        mujoco.mj_step(m, d)
        e_q.append(q_hat - d.qpos[:6]); e_qd.append(qd_hat - d.qvel[:6]); peak.append(np.abs(e_qd[-1]).max())
        inside.append(np.abs(np.r_[e_q[-1], e_qd[-1]]) < 3 * obs.sigma)
    rms = lambda a: np.sqrt(np.mean(np.square(a)))
    print(f"accel_std={accel_std:5.1f}: pos {rms(e_q)*1e3:.3f} mrad | vel {rms(e_qd):.4f} rad/s | "
          f"worst vel {max(peak):.3f} | within 3sigma {100*np.mean(inside):.1f}%")