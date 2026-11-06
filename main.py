"""UR3e RGB-D tracking and pick demo with bounded multi-path scenarios.
Run: python main.py [--paths circle] [--seed 42] [--headless].
Requires robot_model/pick_scene.xml and existing UR3e OBJ meshes.
"""
import argparse
import csv
import time
from pathlib import Path
import cv2
import mujoco
import mujoco.viewer
import numpy as np

from src.perception.sensing import camera
from src.estimation.cube_ekf import CubeEKF
from src.control.ik import make_ik
from src.control.controller import make_controller
from src.estimation.joint_observer import JointObserver

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--headless', action='store_true')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--tau', type=float, default=0.15)
ap.add_argument('--step-limit', type=float, default=0.016)
ap.add_argument('--no-feedforward', action='store_true')
ap.add_argument('--seeds', type=int, default=1, help='Number of seeded trials (default: one demo)')
ap.add_argument('--single-seed', type=int, default=None, help='Run just one trial')
ap.add_argument('--pause-start', type=float, default=3.0)
ap.add_argument('--pause-frames', type=int, default=2)
ap.add_argument('--stop-time', type=float, default=6.0)
ap.add_argument('--paths', type=str, default='circle,horizontal,vertical,diagonal,square,zigzag', help='Comma-separated trajectory types')
ap.add_argument('--margin', type=float, default=0.005, help='Margin inside tested XY workspace')
ap.add_argument('--amplitude', type=float, default=1.0, help='Scale of all trajectories, <=1 recommended')
ap.add_argument('--enc-noise', type=float, default=1.0, help='x nominal encoder noise (simulator side)')
ap.add_argument('--depth-noise', type=float, default=1.0, help='x nominal depth/rgb noise')
ap.add_argument('--torque-dist', type=float, default=0.0, help='white torque noise [Nm] per joint')
ap.add_argument('--torque-bias', type=float, default=0.0, help='constant torque disturbance [Nm] per joint')
ap.add_argument('--mass-mismatch', type=float, default=1.0, help="controller model mass scale (1.2 = +20%)")
ap.add_argument('--ekf-accel', type=float, default=0.30, help='cube EKF process noise [m/s^2]')
ap.add_argument('--obs-accel', type=float, default=1.0, help='joint observer process noise [rad/s^2]')
ap.add_argument('--q-w', type=float, default=2500.0, help='LQR position weight')
ap.add_argument('--tag', type=str, default='nominal', help='label for output files')
ap.add_argument('--obs-enc-scale', type=float, default=1.0, help="observer's assumed encoder sigma, x nominal")   # <- new
args = ap.parse_args()

m = mujoco.MjModel.from_xml_path(str(ROOT / 'robot_model' / 'pick_scene.xml'))
ik = make_ik(m)
box_dof = int(m.jnt_dofadr[m.joint('box_joint').id])
box_qpos = int(m.jnt_qposadr[m.joint('box_joint').id])
box_id = m.body('box').id
tcp_id = m.site('tcp').id
cup_id = m.geom('cup').id
box_geom_id = m.geom('box_geom').id
tool_id = m.body('tool').id
suction_id = m.equality('suction').id
check = mujoco.MjData(m)
est = mujoco.MjData(m)          # model evaluated at the ESTIMATE
ENC_STD = 5e-4                  # simulated encoder noise [rad]
BOX_MASS0 = float(m.body_mass[box_id])
BOX_INERTIA0 = m.body_inertia[box_id].copy()
LIGHT0 = m.light_diffuse.copy()
vel_cap = 0.6
damping = 0.03
sample_dt = 0.1
PATHS = ('horizontal','vertical','diagonal','square','circle','zigzag')
# Conservative workspace for THIS MuJoCo workcell, not the full UR3e workspace.
# Each candidate is additionally validated by IK and joint control limits.
XY_MIN = np.array([0.20, 0.04])
XY_MAX = np.array([0.34, 0.16])

def read_encoders(d, rng):
    """SIMULATED SENSOR: the simulator's true joint angles corrupted by encoder noise.
    This is the only place true joint state enters the control pipeline."""
    return d.qpos[:6] + rng.normal(0, ENC_STD * args.enc_noise, 6)
    
def tcp_from(q):
    """Cup-tip position from joint angles q (forward kinematics on the private model)."""
    est.qpos[:6] = q
    mujoco.mj_kinematics(m, est)
    return est.site_xpos[tcp_id].copy()

def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u*u*(3-2*u)

def path_offset(kind, u):
    """Bounded XY offsets in metres, parameter u from 0 to 1."""
    u = np.clip(u, 0.0, 1.0)
    if kind == 'horizontal':
        return np.array([0.022*np.sin(2*np.pi*u), 0.0])
    if kind == 'vertical':
        return np.array([0.0, 0.022*np.sin(2*np.pi*u)])
    if kind == 'diagonal':
        v = 0.020*np.sin(2*np.pi*u)
        return np.array([v, 0.75*v])
    if kind == 'circle':
        return np.array([0.020*np.sin(2*np.pi*u), 0.020*(1-np.cos(2*np.pi*u))/2])
    if kind == 'square':
        # Closed square, starting at centre of lower edge, rounded in time at corners.
        pts = np.array([[0.,0.], [0.020,0.], [0.020,0.020],
                        [-0.020,0.020], [-0.020,0.], [0.,0.]])
        f = min(u*5, 4.999999)
        j = int(f)
        return pts[j]+(pts[j+1]-pts[j])*smooth(f-j)
    if kind == 'zigzag':
        pts = np.array([[0.,0.],[0.018,0.012],[-0.018,0.024],
                        [0.018,0.012],[-0.018,0.0],[0.,0.]])
        f = min(u*5, 4.999999)
        j = int(f)
        return pts[j]+(pts[j+1]-pts[j])*smooth(f-j)
    raise ValueError('Unknown path '+str(kind))

def motion_xy(kind, t, centre, pause_start, pause_duration, stop_time):
    # During scripted pause freeze exact position; resume with continuous path time.
    active_t = min(t, pause_start) if t < pause_start+pause_duration else t-pause_duration
    active_t = max(0.0, min(active_t, stop_time-pause_duration))
    denom = max(stop_time-pause_duration, 0.1)
    return centre + args.amplitude*path_offset(kind, active_t/denom)

def reachable_preflight(kind, centre, q_seed, pause_duration):
    """Check every sampled path point against workspace, IK and camera geometry.
    Does not guarantee collision-free motion or grasp; the live trial validates those.
    """
    if args.amplitude <= 0 or args.amplitude > 1.0:
        return False, 'amplitude must be in (0,1]'
    q = q_seed.copy()
    for u in np.linspace(0, 1, 81):
        xy = centre+args.amplitude*path_offset(kind, float(u))
        if np.any(xy < XY_MIN+args.margin) or np.any(xy > XY_MAX-args.margin):
            return False, f'outside conservative workspace: {xy.round(4)}'
        for z in (0.30, 0.12, 0.035):
            qi = validated_ik([*xy,z], q)
            if qi is None:
                return False, f'IK rejected {xy.round(4)} at z={z:.3f}'
            if z == 0.30:
                q = qi
                # Approximate camera FOV check for cube top centre at tracking pose.
                scratch = mujoco.MjData(m)
                scratch.qpos[:6] = qi
                mujoco.mj_forward(m, scratch)
                cid = m.camera('wrist_camera').id
                cpos = scratch.cam_xpos[cid]
                R = scratch.cam_xmat[cid].reshape(3,3) @ np.diag([1,-1,-1])
                pc = R.T @ (np.array([*xy,0.04])-cpos)
                if pc[2] <= 0.05:
                    return False, 'cube behind camera at tracking pose'
                f = (480/2)/np.tan(np.deg2rad(float(m.cam_fovy[cid]))/2)
                uv = np.array([320,240])+f*pc[:2]/pc[2]
                if not (30 < uv[0] < 610 and 30 < uv[1] < 450):
                    return False, f'cube outside camera image margin: {uv.round(1)}'
    return True, 'OK'



def run_trial(seed, trajectory):
    global d, cam, ekf, controller, q_ref, qd_ref, last_detection, last_vision
    global next_camera, stage, stage_start, stage_duration, q_start, q_goal
    global locked_xy, stationary_frames, attached, contact_count, pick_error
    global initial_height, rows, seen, accepted, lost, ik_fail, clip_frames
    global max_joint_error, max_ff, check_target, aborts, countdown_started
    global false_pick, stable_time, tracking_resume, paused_frames_seen
    global q_hat, qd_hat
    rng = np.random.default_rng(seed)
    fric = rng.uniform(0.5, 1.2)                      # cube friction coefficient
    mass = rng.uniform(0.02, 0.08)                    # cube mass [kg]
    light = rng.uniform(0.6, 1.2)                     # lighting scale
    m.geom_friction[box_geom_id, 0] = fric
    m.body_mass[box_id] = mass
    m.body_inertia[box_id] = BOX_INERTIA0 * mass / BOX_MASS0    # keep inertia consistent with mass
    m.light_diffuse[:] = LIGHT0 * light
    pause_duration = args.pause_frames*sample_dt
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    # Conservative variation: small initial cube offsets around validated work area.
    centre = np.array([0.27 + rng.uniform(-0.015, 0.015),
                       0.09 + rng.uniform(-0.015, 0.015)])
    d.qpos[box_qpos:box_qpos+2] = centre
    q_scan = np.asarray(ik(np.array([0.27, 0.10, 0.30]), d.qpos[:6].copy()), float)
    if not np.all(np.isfinite(q_scan)):
        raise RuntimeError('Invalid scan IK')
    ok, reason = reachable_preflight(trajectory, centre, q_scan, pause_duration)
    if not ok:
        return dict(seed=seed, trajectory=trajectory, final_stage='PREFLIGHT_REJECT',
                    success=0, failure_reason=reason, rows=[],
                    friction=round(fric,3), cube_mass=round(mass,3), light_scale=round(light,3),
                    start_x=round(float(centre[0]),4), start_y=round(float(centre[1]),4))
    d.qpos[:6] = q_scan
    d.eq_active[suction_id] = False
    mujoco.mj_forward(m, d)
    cam = camera(m, seed=seed)
    ekf = CubeEKF(dt=sample_dt, K=cam.K, top_z=cam.top_z, accel_std=args.ekf_accel)
    mc = mujoco.MjModel.from_xml_path(str(ROOT / 'robot_model' / 'pick_scene.xml'))
    mc.body_mass[:] *= args.mass_mismatch
    mc.body_inertia[:] *= args.mass_mismatch
    controller = make_controller(mc, q0=q_scan, q_w=args.q_w)
    enc_rng = np.random.default_rng(seed + 1000)
    obs = JointObserver(m.opt.timestep, ENC_STD * args.obs_enc_scale, accel_std=args.obs_accel)   # <- changed
    dist_rng = np.random.default_rng(seed + 2000)
    err_log, sig_log, tau_peak = [], [], 0.0
    q_ref = q_scan.copy()
    qd_ref = np.zeros(6)
    last_detection = None
    last_vision = None
    next_camera = 0.0
    stage = 'TRACK'
    stage_start = 0.0
    stage_duration = 0.0
    q_start = q_scan.copy()
    q_goal = q_scan.copy()
    locked_xy = None
    stationary_frames = 0
    attached = False
    contact_count = 0
    pick_error = None
    initial_height = float(d.xpos[box_id, 2])
    rows = []
    seen = accepted = lost = ik_fail = clip_frames = 0
    max_joint_error = max_ff = 0.0
    aborts = countdown_started = false_pick = tracking_resume = paused_frames_seen = 0
    stable_time = None
    check_target = None
    viewer = None
    try:
        if not args.headless:
            viewer = mujoco.viewer.launch_passive(m, d)
            viewer.cam.lookat[:] = [0.25, 0.10, 0.15]
            viewer.cam.distance = 1.0
        while d.time < 20.0 and stage not in ('SUCCESS', 'FAIL') and (viewer is None or viewer.is_running()):
            tic = time.perf_counter()
            t = float(d.time)
            # Ground truth used ONLY to generate the motion scenario and evaluate results.
            # It is never fed into camera localization, EKF, IK or tracking control.
            short_pause = args.pause_start <= t < args.pause_start + pause_duration
            if not attached:
                xy = motion_xy(trajectory,t,centre,args.pause_start,pause_duration,args.stop_time)
                xy_next = motion_xy(trajectory,t+0.001,centre,args.pause_start,pause_duration,args.stop_time)
                d.qpos[box_qpos:box_qpos+2] = xy
                d.qvel[box_dof:box_dof+2] = (xy_next-xy)/0.001 
                d.qvel[box_dof+2:box_dof+6] = 0.0
                d.qpos[box_qpos+2] = 0.02
                d.qpos[box_qpos+3:box_qpos+7] = [1, 0, 0, 0]
            mujoco.mj_forward(m, d)
            z = read_encoders(d, enc_rng)                       # simulated sensor: true q + noise
            q_hat, qd_hat = obs.step(z, controller.last_a)      # observer: estimates q and qd from z
            tcp_hat = tcp_from(q_hat)                           # where the controller believes the cup tip is
            if t + 1e-9 >= next_camera and stage in ('TRACK', 'VERIFY', 'APPROACH'):
                next_camera += sample_dt
                rgb, depth = cam.take_photo(d, noise_scale=args.depth_noise)
                q_actual = q_hat.copy()
                cam_pos, cam_R = cam.pose(q_actual)
                result = cam.locate(rgb, depth, q_actual)
                if not args.headless:
                    vis = camera.debug_image(rgb, result[1] if result else None)
                    cv2.imshow('wrist camera (green = fitted cube)', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                    cv2.imshow('depth', (255 * (1 - np.clip(depth, 0, 1))).astype(np.uint8))
                    cv2.waitKey(1)
                if ekf.initialized and last_vision is not None:
                    ekf.predict(max(t - last_vision, 1e-6))
                last_vision = t
                measurement_ok = False
                if result is not None:
                    seen += 1
                    _, det = result
                    if ekf.update(det, cam_pos, cam_R):
                        accepted += 1
                        last_detection = t
                        measurement_ok = True
                else:
                    lost += 1
                fresh = (measurement_ok and ekf.initialized and
                         last_detection is not None and t-last_detection <= 0.15)
                if short_pause:
                    paused_frames_seen += 1
                if stage == 'TRACK':
                    if fresh:
                        est_xy = ekf.position[:2].copy()
                        p_target = np.array([*(est_xy + args.tau*ekf.velocity), 0.30])
                        q_candidate = validated_ik(p_target, q_actual)
                        if q_candidate is not None:
                            delta = q_candidate-q_ref
                            clip_frames += int(np.any(np.abs(delta)>args.step_limit))
                            q_ref += np.clip(delta, -args.step_limit, args.step_limit)
                            if not args.no_feedforward:
                                # Evaluate Jacobian at the measured configuration.
                                jp = np.zeros((3, m.nv))
                                jr = np.zeros((3, m.nv))
                                mujoco.mj_comPos(m, est)          # est already holds q_hat from tcp_from()
                                jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
                                mujoco.mj_jacSite(m, est, jp, jr, tcp_id)
                                J = jp[:2, :6]
                                raw = J.T @ np.linalg.solve(J@J.T+damping**2*np.eye(2), ekf.velocity)
                                qd_ref = np.clip(raw, -vel_cap, vel_cap)
                                max_ff = max(max_ff, float(np.max(np.abs(qd_ref))))
                            else:
                                qd_ref[:] = 0
                        else:
                            ik_fail += 1
                            qd_ref[:] = 0
                        tcp_xy_err = float(np.linalg.norm(tcp_hat[:2]-est_xy))
                        stationary = (ekf.speed < 0.003 and tcp_xy_err < 0.005)
                    else:
                        stationary = False
                        qd_ref[:] = 0
                    if stationary:
                        stationary_frames += 1
                        if stationary_frames == 1:
                            countdown_started += 1
                    else:
                        if stationary_frames:
                            aborts += 1
                            tracking_resume += 1
                            print(f'[seed={seed}] t={t:.2f}s countdown ABORT, resume TRACK '
                                  f'(previous={stationary_frames}/5)')
                        stationary_frames = 0
                    if stationary_frames >= 5:
                        # Do not commit yet: require independent confirmation while VERIFY.
                        stage = 'VERIFY'
                        stage_start = t
                        stable_time = t
                        locked_xy = ekf.position[:2].copy()
                        qd_ref[:] = 0
                        print(f'[seed={seed}] t={t:.2f}s countdown complete -> VERIFY')
                elif stage == 'VERIFY':
                    # Three additional fresh frames, including a speed and position consistency check.
                    if (not fresh or ekf.speed >= 0.003 or
                        np.linalg.norm(ekf.position[:2]-locked_xy) >= 0.003):
                        print(f'[seed={seed}] t={t:.2f}s VERIFY ABORT -> TRACK')
                        aborts += 1
                        tracking_resume += 1
                        stationary_frames = 0
                        locked_xy = None
                        stage = 'TRACK'
                    elif t-stage_start >= 0.3-1e-6:
                        locked_xy = ekf.position[:2].copy()
                        start_motion('APPROACH', [*locked_xy, 0.12], 1.5)
                        check_target = np.array([*locked_xy, 0.12])
                elif stage == 'APPROACH':
                    # Cancel approach on resumed cube movement, before descent begins.
                    if (not fresh or ekf.speed >= 0.004 or
                        np.linalg.norm(ekf.position[:2]-locked_xy) >= 0.005):
                        print(f'[seed={seed}] t={t:.2f}s APPROACH ABORT -> TRACK')
                        aborts += 1
                        tracking_resume += 1
                        stage = 'TRACK'
                        stationary_frames = 0
                        locked_xy = None
                        qd_ref[:] = 0
                if int(round(t*10)) % 10 == 0:
                    err = np.linalg.norm(d.site_xpos[tcp_id,:2]-ekf.position[:2])*1000 if ekf.initialized else np.nan
                    print(f'[seed={seed}] t={t:4.1f}s {stage} camera={"OK" if result is not None else "LOST"} '
                          f'speed={ekf.speed if ekf.initialized else np.nan:.4f} TCPerr={err:.1f}mm stable={stationary_frames}/5')
            if stage in ('APPROACH','DESCEND','LIFT'):
                f = np.clip((t-stage_start)/stage_duration,0,1)
                smooth = 3*f*f-2*f*f*f
                q_ref = q_start+smooth*(q_goal-q_start)
                qd_ref[:] = 0
                if stage == 'DESCEND' and not attached:
                    contact = any(((c.geom1==cup_id and c.geom2==box_geom_id) or
                                   (c.geom1==box_geom_id and c.geom2==cup_id)) and c.dist <= 0.001
                                  for c in d.contact)
                    if contact:
                        contact_count += 1
                        R = d.xmat[tool_id].reshape(3,3)
                        m.eq_data[suction_id,3:6] = R.T@(d.xpos[box_id]-d.xpos[tool_id])
                        inv = np.empty(4)
                        mujoco.mju_negQuat(inv,d.xquat[tool_id])
                        mujoco.mju_mulQuat(m.eq_data[suction_id,6:10],inv,d.xquat[box_id])
                        d.eq_active[suction_id] = True
                        attached = True
                        lift_target = tcp_hat.copy() + [0, 0, 0.15]
                        print(f'[seed={seed}] t={t:.2f}s CONTACT suction ON')
                        start_motion('LIFT',lift_target,1.5)
                        check_target = lift_target
                if t >= stage_start+stage_duration+0.35:
                    if stage == 'APPROACH':
                        # One last camera validation was done in the latest frame above.
                        if last_detection is None or t-last_detection > 0.15 or ekf.speed >= 0.004:
                            print(f'[seed={seed}] t={t:.2f}s PRE-DESCENT ABORT -> TRACK')
                            aborts += 1
                            tracking_resume += 1
                            stage = 'TRACK'
                            stationary_frames = 0
                            locked_xy = None
                        else:
                            check_target = np.array([*locked_xy,0.035])
                            start_motion('DESCEND',check_target,2.0)
                    elif stage == 'DESCEND':
                        stage = 'FAIL'
                        pick_error = 'No cup-to-box contact by descent timeout'
                    elif stage == 'LIFT':
                        rise = float(d.xpos[box_id,2]-initial_height)
                        if attached and rise > 0.10:
                            stage = 'SUCCESS'
                            print(f'[seed={seed}] PICK SUCCESS: cube rose {rise*1000:.1f}mm')
                        else:
                            stage = 'FAIL'
                            pick_error = f'Lift failed, rise={rise*1000:.1f}mm'
            cmd = np.asarray(controller.compute(q_hat, qd_hat, q_ref, qd_ref=qd_ref), float)
            if cmd.shape!=(6,) or not np.all(np.isfinite(cmd)):
                raise RuntimeError('Invalid joint command')
            d.ctrl[:6] = cmd + args.torque_bias + dist_rng.normal(0, args.torque_dist, 6)
            tau_peak = max(tau_peak, float(np.max(np.abs(cmd) / controller.limit)))
            err_log.append(np.r_[q_hat - d.qpos[:6], qd_hat - d.qvel[:6]])
            sig_log.append(obs.sigma.copy())
            max_joint_error = max(max_joint_error,float(np.max(np.abs(q_ref-d.qpos[:6]))))
            if int(round(t/m.opt.timestep))%10 == 0:
                rows.append([seed,t,stage,int(short_pause),stationary_frames,aborts,
                             *d.xpos[box_id,:3],*d.site_xpos[tcp_id],
                             *(ekf.position[:2] if ekf.initialized else [np.nan,np.nan]),
                             ekf.speed if ekf.initialized else np.nan,int(attached),
                             *q_ref,*d.qpos[:6],
                             *d.qvel[:6], *qd_hat, *obs.sigma[6:12], *q_hat, *obs.sigma[:6], *cmd,
                             *(ekf.velocity if ekf.initialized else [np.nan, np.nan]),
                             *d.qvel[box_dof:box_dof+2]])
            mujoco.mj_step(m,d)
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0,m.opt.timestep-(time.perf_counter()-tic)))
    finally:
        cam.ren.close()
        if viewer is not None:
            viewer.close()
    rise_mm = 1000*(d.xpos[box_id,2]-initial_height)
    E, S = np.array(err_log), np.array(sig_log)
    stats = dict(q_rmse_mrad=round(1e3*float(np.sqrt(np.mean(E[:, :6]**2))), 3),
                 qd_rmse=round(float(np.sqrt(np.mean(E[:, 6:]**2))), 4),
                 within_3sigma=round(float(np.mean(np.abs(E) < 3*S[:, :12])), 4),
                 tau_peak=round(tau_peak, 3))
    # A false pick is any suction/contact during the scripted temporary pause.
    # The test should also confirm tracking was maintained until the final stop.
    return dict(seed=seed,trajectory=trajectory,final_stage=stage,success=int(stage=='SUCCESS'),
                **stats, tag=args.tag, enc_noise=args.enc_noise, depth_noise=args.depth_noise,
                torque_dist=args.torque_dist, torque_bias=args.torque_bias,
                mass_mismatch=args.mass_mismatch,
                measurements=seen,accepted=accepted,lost=lost,ik_rejected=ik_fail,
                clipping_frames=clip_frames,abort_count=aborts, friction=round(fric,3), cube_mass=round(mass,3), light_scale=round(light,3),
                start_x=round(float(centre[0]),4), start_y=round(float(centre[1]),4),
                countdowns_started=countdown_started,resumes=tracking_resume,
                pause_camera_frames=paused_frames_seen,contact_events=contact_count,
                suction=int(attached),rise_mm=round(rise_mm,2),
                max_joint_error_rad=round(max_joint_error,4),
                max_ff_rad_s=round(max_ff,4),
                failure_reason=pick_error or '', rows=rows)

def validated_ik(p, q_seed):
    q = np.asarray(ik(np.asarray(p, float), q_seed), float)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        return None
    bounds = m.jnt_range[:6]
    if np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1]):
        return None
    check.qpos[:6] = q
    mujoco.mj_forward(m, check)
    if np.linalg.norm(check.site_xpos[tcp_id] - p) > 0.003:
        return None
    return q


def start_motion(new_stage, p, duration):
    global stage, stage_start, stage_duration, q_start, q_goal, qd_ref, pick_error, q_hat, qd_hat
    q = validated_ik(p, q_hat)
    if q is None:
        pick_error = f'IK rejected {new_stage} target {np.asarray(p).round(4)}'
        stage = 'FAIL'
        print(pick_error)
        return
    q_start = q_hat.copy()
    q_goal = q
    stage_start = float(d.time)
    stage_duration = float(duration)
    qd_ref[:] = 0
    stage = new_stage
    print(f't={d.time:.2f}s -> {stage}, target TCP {np.asarray(p).round(4)}')


def at_goal(tol_xyz=0.006, tol_q=0.05):
    return (np.linalg.norm(d.site_xpos[tcp_id] - check_target) < tol_xyz
            and np.max(np.abs(d.qpos[:6] - q_goal)) < tol_q)


if __name__ == '__main__':
    seeds = [args.single_seed] if args.single_seed is not None else list(range(args.seed, args.seed+args.seeds))
    paths = [p.strip() for p in args.paths.split(',') if p.strip()]
    if not paths or any(p not in PATHS for p in paths):
        ap.error(f'--paths must contain only {PATHS}')
    results = []
    output_dir = ROOT / 'results'
    output_dir.mkdir(exist_ok=True)
    for idx, seed in enumerate(seeds):
        trajectory = paths[idx % len(paths)]
        print(f'\n===== TEST M seed={seed} trajectory={trajectory} =====')
        try:
            result = run_trial(seed, trajectory)
        except Exception as e:
            result = dict(seed=seed, trajectory=trajectory, final_stage='EXCEPTION',success=0, failure_reason=repr(e), rows=[])
            print('TRIAL EXCEPTION:',repr(e))
        rows = result.pop('rows')
        if rows:
            with (output_dir/f'tracking_{trajectory}_seed{seed}_{args.tag}.csv').open('w',newline='') as f:
                w=csv.writer(f)
                w.writerow(['seed','time','stage','scripted_pause','stationary_frames','abort_count',
                            'cube_x','cube_y','cube_z','tcp_x','tcp_y','tcp_z',
                            'ekf_x','ekf_y','ekf_speed','attached']+
                           [f'q_ref{i}' for i in range(1,7)]+[f'q_actual{i}' for i in range(1,7)]+
                           [f'qd_true{i}' for i in range(1,7)]+[f'qd_hat{i}' for i in range(1,7)]+
                           [f'qd_sig{i}' for i in range(1,7)]+[f'q_hat{i}' for i in range(1,7)]+
                           [f'q_sig{i}' for i in range(1,7)]+[f'cmd{i}' for i in range(1,7)]+
                           ['ekf_vx','ekf_vy','cube_vx','cube_vy'])
                w.writerows(rows)
        results.append(result)
        print('TRIAL RESULT:',{k:v for k,v in result.items() if k not in ('rows',)})
    with (output_dir/f'run_summary_{args.tag}.csv').open('w',newline='') as f:
        keys=list(dict.fromkeys(k for r in results for k in r))
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    passed=sum(r['success'] for r in results)
    print('\n=== TEST M SUMMARY ===')
    print(f'Success: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)')
    print('Aborted countdowns / revalidation:',sum(r.get('abort_count',0) for r in results))
    print('Seeds with aborted countdown:',sum(r.get('abort_count',0)>0 for r in results))
    print('Scripted pause camera frames:',sum(r.get('pause_camera_frames',0) for r in results))
    print('All seeds:',[(r['seed'],r.get('trajectory'),r['final_stage'],r.get('abort_count',0)) for r in results])
    print('By path:', {p: (sum(r['success'] for r in results if r.get('trajectory')==p), sum(r.get('trajectory')==p for r in results)) for p in paths})
    print('CSV:',output_dir/f'run_summary_{args.tag}.csv')
    print('NOTE: workspace preflight is sampled IK+FOV, not continuous collision checking.')
    print('NOTE: simulation-only, no physical robot safety validation.')