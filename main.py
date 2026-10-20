# Run: python main.py
# Uses the existing camera, Kalman filter and IK.
# controller still needs to replace the position commands below.
import csv
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from src.perception.sensing import camera
from src.estimation.kalman_filter import KalmanFilter
from src.control.ik import make_ik

ROOT = Path(__file__).resolve().parent
SHOW_VIEWER = '--headless' not in sys.argv
SEED = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 0


# Set up the robot and object
m = mujoco.MjModel.from_xml_path(str(ROOT / 'robot_model/pick_scene.xml'))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
rng = np.random.default_rng(SEED)
# Randomise the object only at reset. Vision supplies the picking target.
box_q = m.jnt_qposadr[m.joint('box_joint').id]
d.qpos[box_q:box_q + 2] = [rng.uniform(0.24, 0.30), rng.uniform(0.07, 0.13)]
ik = make_ik(m)
q_scan = ik(np.array([0.27, 0.10, 0.30]), d.qpos[:6])
# Initialise at the scan pose, following the group's sensing test.
d.qpos[:6] = q_scan
d.ctrl[:] = q_scan
mujoco.mj_forward(m, d)
cam = camera(m, seed=SEED)
kf = KalmanFilter(m.opt.timestep, encoder_std=0.005, accel_std=2.0)
tcp = m.site('tcp').id
cup = m.geom('cup').id
box_geom = m.geom('box_geom').id
tool = m.body('tool').id
box = m.body('box').id
suction = m.equality('suction').id
check = mujoco.MjData(m)
viewer = mujoco.viewer.launch_passive(m, d) if SHOW_VIEWER else None
if viewer is not None:
    viewer.cam.lookat[:] = [0.2, 0, 0.2]
    viewer.cam.distance = 1.2
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -25

# Get joint angles and check the target can be reached
def target(p, q):
    q_goal = ik(p, q)
    check.qpos[:6] = q_goal
    mujoco.mj_forward(m, check)
    error = np.linalg.norm(check.site_xpos[tcp] - p)
    if not np.all(np.isfinite(q_goal)) or error > 0.003:
        raise ValueError('IK target could not be reached')
    bounds = m.actuator_ctrlrange
    if np.any((q_goal < bounds[:, 0]) | (q_goal > bounds[:, 1])):
        raise ValueError('IK target exceeds position command limits')
    return q_goal

# Start with the camera looking at the table
stage = 'scan'
stage_start = 0.0
duration = 2.0
q_start = q_scan.copy()
q_goal = q_scan.copy()
attached = False
success = False
initial_height = d.xpos[box, 2]  # evaluation only
output = ROOT / 'tracking_results.csv'
try:
    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time'] + [f'q_ref{i}' for i in range(1, 7)]
                        + [f'q_actual{i}' for i in range(1, 7)])
        while d.time < 25 and (viewer is None or viewer.is_running()):
            start = time.perf_counter()
            q_measured = d.qpos[:6] + rng.normal(0, 0.005, 6)
            q_hat, qd_hat = kf.step(q_measured)
            elapsed = d.time - stage_start
            fraction = np.clip(elapsed / duration, 0, 1)
            blend = 3 * fraction**2 - 2 * fraction**3
            q_ref = q_start + blend * (q_goal - q_start)

            # Simulated contact switch: only cup-to-object contact can attach.
            contact = False
            for c in d.contact:
                cup_box = c.geom1 == cup and c.geom2 == box_geom
                box_cup = c.geom1 == box_geom and c.geom2 == cup
                if (cup_box or box_cup) and c.dist <= 0:
                    contact = True
            if stage in ('descend', 'touch') and contact:
                # Store the cube position and rotation relative to the tool.
                R = d.xmat[tool].reshape(3, 3)
                m.eq_data[suction, 3:6] = R.T @ (d.xpos[box] - d.xpos[tool])
                inverse = np.empty(4)
                mujoco.mju_negQuat(inverse, d.xquat[tool])
                mujoco.mju_mulQuat(m.eq_data[suction, 6:10], inverse, d.xquat[box])
                d.eq_active[suction] = True
                attached = True
                stage = 'lift'
                q_start = q_ref.copy()
                q_goal = target(p_object + [0, 0, 0.15], q_hat)
                stage_start = d.time
                duration = 4.0
                print('Contact detected: suction on, lifting')
            elif elapsed >= duration + 0.8:
                if stage == 'scan' or stage == 'approach':
                    rgb, depth = cam.take_photo(d)
                    result = cam.locate(rgb, depth, q_hat)
                    if result is None:
                        print('Object not detected')
                        break
                    p_object, detection = result
                    q_start = q_ref.copy()
                    duration = 3.0
                    if stage == 'scan':
                        q_goal = target(p_object + [0, 0, 0.08], q_hat)
                        stage = 'approach'
                        print('Detected object top:', p_object.round(3))
                    else:
                        q_goal = target(p_object, q_hat)
                        stage = 'descend'
                elif stage == 'descend':
                    # Small extra descent allows for depth/servo error.
                    q_start = q_ref.copy()
                    q_goal = target(p_object - [0, 0, 0.008], q_hat)
                    stage, duration = 'touch', 1.5
                elif stage == 'touch':
                    print('No cup contact: pick failed')
                    break
                elif stage == 'lift':
                    # Use the actual cube height only to check whether the lift worked.
                    height = d.xpos[box, 2] - initial_height
                    success = attached and height > 0.10
                    print(f'Lifted {height:.3f} m; success={success}')
                    break
                stage_start = d.time

            # Temporary built-in position servos. Replace with Alzaid's controller.
            d.ctrl[:] = q_ref
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            writer.writerow([d.time] + q_ref.tolist() + d.qpos[:6].tolist())
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0, m.opt.timestep - (time.perf_counter() - start)))
finally:
    cam.ren.close()
    if viewer is not None:
        viewer.close()
    print('Saved:', output.name)

