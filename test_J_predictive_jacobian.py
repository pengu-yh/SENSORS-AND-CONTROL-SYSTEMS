"""Test J: predictive visual tracking + Jacobian velocity feedforward (NO PICK).
Run from repository root: .\\.venv\\Scripts\\python.exe test_H_visual_tracking.py
Use --headless to disable viewer. Needs robot_model/pick_scene.xml and its OBJ meshes.
"""
import csv
import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from src.perception.sensing import camera
from src.estimation.cube_ekf import CubeEKF
from src.control.ik import make_ik
from src.control.controller import make_controller

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--headless', action='store_true')
parser.add_argument('--step-limit', type=float, default=0.016)
parser.add_argument('--tau', type=float, default=0.0, choices=[0.0, 0.05, 0.10, 0.15])
parser.add_argument('--feedforward', action='store_true', help='Use damped Jacobian pseudoinverse for joint velocity feedforward')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
SHOW_VIEWER = not args.headless
m = mujoco.MjModel.from_xml_path(str(ROOT / 'robot_model' / 'pick_scene.xml'))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
ik = make_ik(m)
q_scan = np.asarray(ik(np.array([0.27, 0.10, 0.30]), d.qpos[:6].copy()), float)
if not np.all(np.isfinite(q_scan)):
    raise RuntimeError('Invalid scan IK')
d.qpos[:6] = q_scan
d.ctrl[:6] = q_scan
mujoco.mj_forward(m, d)
cam = camera(m, seed=args.seed)
ekf = CubeEKF(dt=0.1, K=cam.K, top_z=cam.top_z)
controller = make_controller(m, q0=q_scan)
box_dof = int(m.jnt_dofadr[m.joint('box_joint').id])
box_qpos = int(m.jnt_qposadr[m.joint('box_joint').id])
box_body = m.body('box').id
tcp_id = m.site('tcp').id
check = mujoco.MjData(m)
q_ref = q_scan.copy()
qd_ref = np.zeros(6)
prev_q_ref = q_ref.copy()
last_jac_cond = np.nan
max_ff_velocity = 0.0
max_joint_ref_error = 0.0
max_tcp_target_error = 0.0
ff_saturated = 0
accepted_targets = 0
prev_command_time = None
vel_cap = 0.6  # rad/s per joint
damping = 0.03  # damped least squares Jacobian, meters/rad translational

last_vision = None
last_detection = None
last_target = None
next_camera_time = 0.0
last_ik_time = -1.0
sample_dt = 0.1
tracking_height = 0.30  # TCP stays above table; NO DESCENT
max_target_step = args.step_limit  # experimental variable; per 0.1s camera frame
rows = []
q_diagnostics = []
clip_frames = 0
clip_axes = 0
max_joint_tracking_error = 0.0
accepted = 0
seen = 0
lost = 0
ik_fail = 0

viewer = None
try:
    if SHOW_VIEWER:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(m, d)
        viewer.cam.lookat[:] = [0.25, 0.10, 0.15]
        viewer.cam.distance = 1.0
    while d.time < 10.0 and (viewer is None or viewer.is_running()):
        tic = time.perf_counter()
        t = float(d.time)
        # TEST FIXTURE ONLY: prescribe cube planar velocity, NOT given to EKF/controller.
        # Same sinusoidal velocity used in prior isolated moving-cube tests.
        if t < 6.0:
            d.qvel[box_dof] = 0.035 * np.cos(2.0*t)
            d.qvel[box_dof+1] = 0.035 * np.sin(2.0*t)
        else:
            d.qvel[box_dof:box_dof+2] = 0.0
        d.qvel[box_dof+2:box_dof+6] = 0.0
        # Hold cube upright for the known-yaw square model. This is a test fixture,
        # not a measurement or state input to the estimator.
        d.qpos[box_qpos+3:box_qpos+7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)

        if t + 1e-9 >= next_camera_time:
            next_camera_time += sample_dt
            rgb, depth = cam.take_photo(d, noise_scale=1.0)
            q_actual = d.qpos[:6].copy()  # real joint state for camera extrinsics
            cam_pos, cam_R = cam.pose(q_actual)
            result = cam.locate(rgb, depth, q_actual)
            if ekf.initialized and last_vision is not None:
                ekf.predict(max(t-last_vision, 1e-6))
            last_vision = t
            if result is not None:
                seen += 1
                _, det = result
                if ekf.update(det, cam_pos, cam_R):
                    accepted += 1
                    last_detection = t
            else:
                lost += 1
            # Never track stale data indefinitely.
            if ekf.initialized and last_detection is not None and t-last_detection <= 0.35:
                p_target = np.array([*(ekf.position[:2] + args.tau * ekf.velocity), tracking_height])
                # Do not run IK more frequently than camera updates.
                q_candidate = np.asarray(ik(p_target, q_actual), dtype=float)
                if q_candidate.shape == (6,) and np.all(np.isfinite(q_candidate)):
                    check.qpos[:6] = q_candidate
                    mujoco.mj_forward(m, check)
                    pos_err = np.linalg.norm(check.site_xpos[tcp_id]-p_target)
                    bounds = m.actuator_ctrlrange[:6]
                    if pos_err <= 0.003 and np.all(q_candidate >= bounds[:, 0]) and np.all(q_candidate <= bounds[:, 1]):
                        raw_dq = q_candidate-q_ref
                        clipped = np.abs(raw_dq) > max_target_step
                        clip_frames += int(np.any(clipped))
                        clip_axes += int(np.sum(clipped))
                        dq = np.clip(raw_dq, -max_target_step, max_target_step)
                        q_ref = np.clip(q_ref+dq, bounds[:, 0], bounds[:, 1])
                        accepted_targets += 1
                        # Feedforward uses the *estimated* object velocity, never ground truth.
                        # MuJoCo translational Jacobian maps joint velocity -> TCP world velocity.
                        # At the accepted IK configuration, compute damped least-squares qdot.
                        if args.feedforward:
                            jp = np.zeros((3, m.nv))
                            jr = np.zeros((3, m.nv))
                            mujoco.mj_jacSite(m, check, jp, jr, tcp_id)
                            J = jp[:2, :6]
                            # 2D XY task: preserve the IK position/orientation in q_ref.
                            # Only velocity command is fed forward to the existing controller.
                            vxy = ekf.velocity.copy()
                            qdot_raw = J.T @ np.linalg.solve(J @ J.T + damping**2*np.eye(2), vxy)
                            if np.any(np.abs(qdot_raw) > vel_cap):
                                ff_saturated += 1
                            qd_ref = np.clip(qdot_raw, -vel_cap, vel_cap)
                            max_ff_velocity = max(max_ff_velocity, float(np.max(np.abs(qd_ref))))
                            last_jac_cond = float(np.linalg.cond(J))
                        else:
                            qd_ref = np.zeros(6)
                        last_target = p_target
                    else:
                        ik_fail += 1
                else:
                    ik_fail += 1
            true_xy = d.xpos[box_body, :2].copy()  # logging ONLY
            tcp_xy = d.site_xpos[tcp_id, :2].copy()
            est_xy = ekf.position[:2] if ekf.initialized else [np.nan, np.nan]
            speed = ekf.speed if ekf.initialized else np.nan
            q_diagnostics.append([t, *q_ref.copy(), *d.qpos[:6].copy(), *qd_ref.copy(), *d.qvel[:6].copy()])
            max_tcp_target_error = max(max_tcp_target_error, float(np.linalg.norm(tcp_xy - p_target[:2]))) if last_target is not None else max_tcp_target_error
            max_joint_tracking_error = max(max_joint_tracking_error, float(np.max(np.abs(q_ref-d.qpos[:6]))))
            rows.append([t, *true_xy, *est_xy, *tcp_xy, speed,
                         int(result is not None), int(last_target is not None), *(last_target[:2] if last_target is not None else [np.nan, np.nan])])
            if int(round(t*10)) % 10 == 0:
                err = np.linalg.norm(tcp_xy-np.asarray(est_xy)) if ekf.initialized else np.nan
                print(f't={t:4.1f}s camera={"OK" if result else "LOST"} '
                      f'ekf_speed={speed:.4f}m/s tcp-to-est={err*1000:.1f}mm')
        # Existing joint-position servo, with measured joint velocity (no oracle cube state).
        cmd = np.asarray(controller.compute(d.qpos[:6].copy(), d.qvel[:6].copy(), q_ref, qd_ref=qd_ref), float)
        if cmd.shape != (6,) or not np.all(np.isfinite(cmd)):
            raise RuntimeError('Controller returned invalid command')
        d.ctrl[:6] = cmd
        mujoco.mj_step(m, d)
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0.0, m.opt.timestep-(time.perf_counter()-tic)))
finally:
    cam.ren.close()
    if viewer is not None:
        viewer.close()
    suffix = f'tau{args.tau:.2f}_ff{int(args.feedforward)}'
    output = ROOT/f'tracking_test_J_{suffix}.csv'
    with output.open('w', newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['time','true_x','true_y','ekf_x','ekf_y','tcp_x','tcp_y','ekf_speed','detected','target_set','target_x','target_y'])
        writer.writerows(rows)
    print(f'=== TEST J SUMMARY (tau={args.tau:.2f}s, feedforward={args.feedforward}, step limit={max_target_step:.3f}) ===\nMeasurements seen: {seen}; EKF accepted: {accepted}; lost: {lost}; IK rejected: {ik_fail}')
    print(f'Reference clipping: {clip_frames}/{len(rows)} camera frames, {clip_axes} joint-axis events')
    print(f'Max |q_ref - q_actual|: {max_joint_tracking_error:.4f} rad')
    print('CSV:', output)
    if rows:
        a=np.asarray(rows, float)
        valid=np.isfinite(a[:,3])
        if valid.any():
            print('EKF XY RMSE [mm]:', round(1000*np.sqrt(np.mean(np.sum((a[valid,1:3]-a[valid,3:5])**2,axis=1))),3))
            print('TCP-to-cube XY RMSE [mm]:', round(1000*np.sqrt(np.mean(np.sum((a[valid,1:3]-a[valid,5:7])**2,axis=1))),3))
    print('Max |joint velocity feedforward| [rad/s]:', round(max_ff_velocity, 4))
    print('FF saturation frames:', ff_saturated)
    print('Last Jacobian condition number:', round(last_jac_cond, 3) if np.isfinite(last_jac_cond) else 'n/a')
    print('NOTE: Tracking-only; no suction or grasp attempted. No actuator gains modified.')
    qout = ROOT/f'joint_test_J_{suffix}.csv'
    with qout.open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['time']+[f'q_ref{i}' for i in range(1,7)]+[f'q_actual{i}' for i in range(1,7)]+[f'qd_ff{i}' for i in range(1,7)]+[f'qd_actual{i}' for i in range(1,7)])
        w.writerows(q_diagnostics)
    print('Joint CSV:',qout)
