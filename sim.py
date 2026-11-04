"""PickSim: runs one pick trial (observe -> localise -> plan -> descend -> grasp -> lift)."""
import sys
import time
import cv2
import mujoco
import numpy as np
from config import CUBE_HALF, DT, GRASP_CLEAR, GRASP_TOL_XY, GRASP_TOL_Z, HOME, JOINTS, LIFT_HEIGHT, N_CYCLES, OUT, PRE_HEIGHT, SETTLE_TIME, SLIDE_V0, SUCCESS_LIFT, TABLE_TOP_Z, Y_RANGE
from camera import Camera, world_to_pixel
from perception import localise
from ik import IK
from trajectory import hold, plan
from controller import StateSpaceController


#SIMULATION 
class PickSim:
    def __init__(self, model, data, rng, viewer=None, speed=1.0):
        self.m, self.d, self.rng, self.viewer = model, data, rng, viewer
        self.speed = speed
        self.ctrl = StateSpaceController(model, data, rng)
        self.ik = IK(model)
        self.cam = Camera(model, data)
        self.tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
        cj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_cube_free")
        self.cq, self.cv = model.jnt_qposadr[cj], model.jnt_dofadr[cj]
        self.q_home = np.array([HOME[j] for j in JOINTS])
        self.attached = False
        self.rel = None
        self.slide = None                      # (pos, vel) arrays while the cube is sliding
        self.slide_k = 0
        self.log = None
        self.stats = None

    # helpers
    def tcp_pose(self):
        return self.d.site_xpos[self.tcp].copy(), self.d.site_xmat[self.tcp].reshape(3, 3).copy()

    def place_cube(self, x, y):
        self.d.qpos[self.cq:self.cq + 7] = [x, y, TABLE_TOP_Z + CUBE_HALF, 1, 0, 0, 0]
        self.d.qvel[self.cv:self.cv + 6] = 0

    def reset(self, cube_xy):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.ctrl.qadr] = self.q_home
        self.place_cube(*cube_xy)
        mujoco.mj_forward(self.m, self.d)
        self.ctrl.reset(self.q_home)
        self.attached, self.rel = False, None
        self.slide, self.slide_k = None, 0

    def start_slide(self, waypoints, v0):
        """Cube follows the polyline 'waypoints' at CONSTANT speed v0 (instant reversals at the
        corners), then stops abruptly at the last waypoint."""
        w = np.asarray(waypoints, float)
        seg = np.diff(w, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        L = float(seg_len.sum())
        T = L / v0
        n = max(int(np.ceil(T / DT)), 1)
        dist = np.minimum(np.arange(1, n + 1) * DT * v0, L)           # distance travelled
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        i = np.minimum(np.searchsorted(cum, dist, side="right") - 1, len(seg) - 1)
        frac = (dist - cum[i]) / seg_len[i]
        pos = w[i] + frac[:, None] * seg[i]
        vel = v0 * seg[i] / seg_len[i][:, None]
        self.slide, self.slide_k = (pos, vel), 0
        return T, L

    def drive_cube(self):
        if self.slide is None:
            return
        pos, vel = self.slide
        k = self.slide_k
        if k < len(pos):
            self.d.qpos[self.cq:self.cq + 3] = [pos[k, 0], pos[k, 1], TABLE_TOP_Z + CUBE_HALF]
            self.d.qvel[self.cv:self.cv + 6] = [vel[k, 0], vel[k, 1], 0, 0, 0, 0]
            self.slide_k += 1
        else:
            self.d.qvel[self.cv:self.cv + 6] = 0
            self.slide = None

    def attach(self):
        p, R = self.tcp_pose()
        pc = self.d.xpos[self.cube].copy()
        Rc = self.d.xmat[self.cube].reshape(3, 3).copy()
        self.rel = (R.T @ (pc - p), R.T @ Rc)
        self.attached = True

    def carry_cube(self):
        if not self.attached:
            return
        p, R = self.tcp_pose()
        pos = p + R @ self.rel[0]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (R @ self.rel[1]).flatten())
        self.d.qpos[self.cq:self.cq + 3] = pos
        self.d.qpos[self.cq + 3:self.cq + 7] = quat
        self.d.qvel[self.cv:self.cv + 6] = 0

    def execute(self, ref):
        q, v, a = ref
        for k in range(len(q)):
            self.ctrl.step(q[k], v[k], a[k])
            mujoco.mj_step(self.m, self.d)
            self.drive_cube()
            self.carry_cube()
            e = self.d.qpos[self.ctrl.qadr] - q[k]
            ev = self.ctrl.vh - self.d.qvel[self.ctrl.dadr]
            s = self.stats
            s["n"] += 1
            s["q_sq"] += float(e @ e)
            s["q_max"] = max(s["q_max"], float(np.abs(e).max()))
            s["obs_sq"] += float(ev @ ev)
            if self.log is not None:
                self.log.append(np.concatenate([[self.d.time], self.d.qpos[self.ctrl.qadr],
                                                self.ctrl.qh, self.d.qvel[self.ctrl.dadr],
                                                self.ctrl.vh, q[k]]))
            if self.viewer is not None and (s["n"] % 8 == 0):
                if not self.viewer.is_running():
                    sys.exit(0)
                self.viewer.sync()
                time.sleep(8 * DT / self.speed)          # real-time playback

    #The simulation checks whether the TCP is within the configured horizontal and vertical tolerances of the cube.
    #  If it is, attach() records the cube's pose relative to the tool. 
    # carry_cube() then moves the cube with the tool, and the trial is marked successful if it rises by at least 8 cm.

    # one trial 
    def run_trial(self, idx, cube_xy, q_obs, R_des, save_images=False, log_signals=False):
        res = dict(trial=idx, cube_x=cube_xy[0], cube_y=cube_xy[1], detected=0, reachable=0,
                   loc_err_mm=np.nan, loc_err_xy_mm=np.nan, loc_err_z_mm=np.nan,
                   grasp_xy_mm=np.nan, grasp_z_mm=np.nan, attached=0, success=0,
                   slide_len_mm=np.nan, slide_v0=np.nan,
                   track_rms_mrad=np.nan, track_max_mrad=np.nan, obs_vel_rms=np.nan)
        self.stats = dict(n=0, q_sq=0.0, q_max=0.0, obs_sq=0.0)
        self.log = [] if log_signals else None
        # line along y at the random x; ends of the line = edges of the Y range
        ends = [np.array([cube_xy[0], Y_RANGE[0]]), np.array([cube_xy[0], Y_RANGE[1]])]
        if self.rng.random() < 0.5:
            ends.reverse()
        A, B = ends
        stop = np.array(cube_xy, float)
        v0 = self.rng.uniform(*SLIDE_V0)
        self.reset(A)

        # 1. go to observation pose, watch the cube shuttle back and forth, stop, then look
        self.execute(hold(self.q_home, 0.3))
        self.execute(plan([self.q_home, q_obs], 3.0))
        self.execute(hold(q_obs, 0.3))
        route = [A] + [B, A] * N_CYCLES + [stop]       # A->B->A->B->A (2 round trips) -> stop
        T_slide, L = self.start_slide(route, v0)
        res["slide_len_mm"], res["slide_v0"] = 1000 * L, v0
        self.execute(hold(q_obs, T_slide))          # cube moving at constant speed, then stops
        self.execute(hold(q_obs, SETTLE_TIME))      # cube at rest, then image is taken

        cam_pos, cam_R = self.cam.pose()
        rgb, depth = self.cam.render(self.rng, noise=True)
        if save_images:
            cv2.imwrite(str(OUT / f"trial{idx:02d}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        det = localise(rgb, depth, self.cam.K, cam_pos, cam_R)
        truth = self.d.xpos[self.cube].copy()
        if det is None:
            self._finish(res)
            return res
        res["detected"] = 1
        err = det["p_center"] - truth
        res["loc_err_mm"] = 1000 * np.linalg.norm(err)
        res["loc_err_xy_mm"] = 1000 * np.linalg.norm(err[:2])
        res["loc_err_z_mm"] = 1000 * err[2]
        if save_images:
            ann = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            x, y, w, h = det["bbox"]
            cv2.rectangle(ann, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(ann, (int(det["uv"][0]), int(det["uv"][1])), 5, (255, 0, 0), -1)
            gt = world_to_pixel(truth + np.array([0, 0, CUBE_HALF]), self.cam.K, cam_pos, cam_R)
            cv2.drawMarker(ann, (int(gt[0]), int(gt[1])), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
            cv2.imwrite(str(OUT / f"trial{idx:02d}_detection.png"), ann)

        # 2. PBVS: plan from the ESTIMATED position only
        top = det["p_top"]
        grasp = np.array([top[0], top[1], top[2] + GRASP_CLEAR])
        pre = grasp + np.array([0, 0, PRE_HEIGHT])
        lift = grasp + np.array([0, 0, LIFT_HEIGHT])
        path_down = self.ik.line(pre, grasp, R_des, q_obs)
        if path_down is None:
            self._finish(res)
            return res
        path_up = self.ik.line(grasp, lift, R_des, path_down[-1])
        if path_up is None:
            self._finish(res)
            return res
        res["reachable"] = 1

        self.execute(plan([q_obs, path_down[0]], 2.5))
        self.execute(hold(path_down[0], 0.4))
        self.execute(plan(path_down, 2.0))
        self.execute(hold(path_down[-1], 0.5))

        # 3. grasp check (virtual suction gripper)
        p_tcp, _ = self.tcp_pose()
        true_top = self.d.xpos[self.cube] + np.array([0, 0, CUBE_HALF])
        dxy = np.linalg.norm((p_tcp - true_top)[:2])
        dz = p_tcp[2] - true_top[2]
        res["grasp_xy_mm"] = 1000 * dxy
        res["grasp_z_mm"] = 1000 * dz
        if dxy <= GRASP_TOL_XY and GRASP_TOL_Z[0] <= dz <= GRASP_TOL_Z[1]:
            self.attach()
            res["attached"] = 1

        # 4. lift
        z0 = self.d.xpos[self.cube][2]
        self.execute(plan(path_up, 2.0))
        self.execute(hold(path_up[-1], 0.5))
        lifted = self.d.xpos[self.cube][2] - z0
        res["success"] = int(self.attached and lifted >= SUCCESS_LIFT)
        self._finish(res)
        return res

    def _finish(self, res):
        s = self.stats
        if s["n"]:
            res["track_rms_mrad"] = 1000 * np.sqrt(s["q_sq"] / (6 * s["n"]))
            res["track_max_mrad"] = 1000 * s["q_max"]
            res["obs_vel_rms"] = np.sqrt(s["obs_sq"] / (6 * s["n"]))
