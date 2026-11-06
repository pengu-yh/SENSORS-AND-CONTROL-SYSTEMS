import mujoco, numpy as np
from pathlib import Path
import sys 

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.control.ik import make_ik

XML = Path(__file__).resolve().parents[2] / "robot_model" / "pick_scene.xml"
m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)      
n = 6
q0 = make_ik(m)(np.array([0.27, 0.10, 0.30]), d.qpos[:n])   # scan pose, where the arm spends the run           

def mass(q):
    d.qpos[:n] = q; mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    try:    mujoco.mj_fullM(m, d, M)        # newer MuJoCo
    except TypeError: mujoco.mj_fullM(m, M, d.qM)   # older MuJoCo
    return M[:n, :n]

def gravity(q):
    d.qpos[:n] = q; d.qvel[:] = 0; mujoco.mj_forward(m, d)
    return d.qfrc_bias[:n].copy()           # at zero velocity this is g(q)

# Linearise  M(q) qdd + C qd + g(q) + D qd = tau   about (q0, qd=0, tau0=g(q0))
eps = 1e-5
G = np.column_stack([(gravity(q0 + eps*e) - gravity(q0 - eps*e)) / (2*eps) for e in np.eye(n)])
Minv = np.linalg.inv(mass(q0))
D = np.diag(m.dof_damping[:n])
Z, I = np.zeros((n, n)), np.eye(n)

A = np.block([[Z, I], [-Minv @ G, -Minv @ D]])     # x = [dq ; dqd]  (12 states)
B = np.vstack([Z, Minv])                           # u = d_tau       (6 inputs)
C = np.hstack([I, Z])                              # y = q (encoders: positions only)

N = 2 * n
ctrb = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(N)])
obsv = np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(N)])
print("Linearised UR3e: states 12, inputs 6, outputs 6")
print("controllability rank:", np.linalg.matrix_rank(ctrb), "/ 12")
print("observability rank  :", np.linalg.matrix_rank(obsv), "/ 12")
print("open-loop eigenvalues:", np.round(np.linalg.eigvals(A), 2))

# Per-joint model after computed-torque feedback linearisation: qdd = a
a1 = np.array([[0, 1], [0, 0]]); b1 = np.array([[0], [1]]); c1 = np.array([[1, 0]])
print("per-joint ctrb rank:", np.linalg.matrix_rank(np.hstack([b1, a1 @ b1])), "/ 2")
print("per-joint obsv rank:", np.linalg.matrix_rank(np.vstack([c1, c1 @ a1])), "/ 2")