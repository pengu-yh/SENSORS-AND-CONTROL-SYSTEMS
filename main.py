"""
ur3e_pick: vision-guided pick with a UR3e in MuJoCo (PBVS), split into modules.

Usage:   python main.py                       (viewer, 5 trials)
         python main.py --headless --trials 100
Output:  stage3_output/  (trials.csv, detection images, trial-0 signals)
"""
import argparse
import csv
import time
import mujoco
import numpy as np
from config import CUBE_HALF, DT, GRASP_CLEAR, LIFT_HEIGHT, N_TRIALS, OBS_TCP, OUT, PRE_HEIGHT, SEED, TABLE_TOP_Z, X_RANGE, Y_RANGE
from scene_builder import build_scene
from trajectory import hold, plan
from sim import PickSim


#  MAIN 
def check_workspace(sim, R_des, q_seed):
    """Reachability scan of the randomisation box (grasp and pre-grasp, tool down)."""
    ok, tot = 0, 0
    for x in np.linspace(*X_RANGE, 5):
        for y in np.linspace(*Y_RANGE, 5):
            top = np.array([x, y, TABLE_TOP_Z + 2 * CUBE_HALF + GRASP_CLEAR])
            tot += 1
            _, g = sim.ik.solve(top, R_des, q_seed)
            _, p = sim.ik.solve(top + [0, 0, PRE_HEIGHT + 0.0], R_des, q_seed)
            _, l = sim.ik.solve(top + [0, 0, LIFT_HEIGHT], R_des, q_seed)
            ok += int(g and p and l)
    print(f"[workspace] {ok}/{tot} grid points in the cube box are reachable (tool down)")
    if ok < tot:
        print("  -> shrink X_RANGE / Y_RANGE or lower TABLE_TOP_Z if this is low")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=None, help="default: 5 with viewer, 25 headless")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--headless", action="store_true", help="no viewer window (fast batch run)")
    ap.add_argument("--speed", type=float, default=1.0, help="viewer playback speed (1 = real time)")
    args = ap.parse_args()
    args.view = not args.headless
    if args.trials is None:
        args.trials = 5 if args.view else N_TRIALS

    scene = build_scene()
    print("Scene:", scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = DT
    data = mujoco.MjData(model)
    rng = np.random.default_rng(args.seed)

    viewer = None
    ctx = None
    if args.view:
        from mujoco import viewer as mj_viewer
        ctx = mj_viewer.launch_passive(model, data)
        viewer = ctx.__enter__()

    if viewer is not None:                        # camera looking at robot + table
        viewer.cam.lookat[:] = [0.30, 0.0, 0.15]
        viewer.cam.distance = 1.3
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -25
    sim = PickSim(model, data, rng, viewer, args.speed)
    CUBE_PARK = (0.62, 0.28)               # park cube out of the calibration view
    sim.reset(CUBE_PARK)

    # sanity check of the camera / tool convention 
    _, R_des = sim.tcp_pose()
    tool_axis = R_des[:, 1]      # wrist_3 local +Y
    cam_pos, cam_R = sim.cam.pose()
    look = -cam_R[:, 2]
    print("Home TCP (m):", np.round(sim.tcp_pose()[0], 3))
    print("Tool axis (wrist_3 +Y) in world:", np.round(tool_axis, 3), "(want ~[0 0 -1])")
    print("Camera view direction in world: ", np.round(look, 3), "(want ~[0 0 -1])")
    if tool_axis[2] > -0.9 or look[2] > -0.9:
        print("WARNING: tool/camera not pointing down at HOME; the flange axis of your model may "
              "not be wrist_3 +Y. Fix TCP_Y/CAM_POS/xyaxes in scene_builder.py.")

    #  observation pose via IK 
    q_obs, ok = sim.ik.solve(OBS_TCP, R_des, sim.q_home)
    if not ok:
        raise RuntimeError("Observation pose not reachable: change OBS_TCP in the CONFIG block")
    check_workspace(sim, R_des, q_obs)

    # camera calibration at the observation pose 
    sim.reset(CUBE_PARK)
    sim.stats = dict(n=0, q_sq=0.0, q_max=0.0, obs_sq=0.0)
    sim.execute(plan([sim.q_home, q_obs], 3.0))
    sim.execute(hold(q_obs, 1.0))
    sim.cam.calibrate()
    # hide markers for the trials
    for i in range(9):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"calib_{i}")
        model.geom_rgba[gid, 3] = 0.0

    # trials 
    rows = []
    t0 = time.time()
    for i in range(args.trials):
        xy = (rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE))
        res = sim.run_trial(i, xy, q_obs, R_des, save_images=(i < 3), log_signals=(i == 0))
        if i == 0 and sim.log:
            arr = np.array(sim.log)
            np.savez(OUT / "trial0_signals.npz", t=arr[:, 0], q=arr[:, 1:7], q_hat=arr[:, 7:13],
                     qd=arr[:, 13:19], qd_hat=arr[:, 19:25], q_ref=arr[:, 25:31])
        rows.append(res)
        print(f"trial {i:02d}  cube=({xy[0]:.3f},{xy[1]:+.3f})  det={res['detected']}  "
              f"loc_err={res['loc_err_mm']:.1f} mm  grasp_xy={res['grasp_xy_mm']:.1f} mm  "
              f"attached={res['attached']}  success={res['success']}")

    # save + summary 
    with open(OUT / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def col(k):
        return np.array([r[k] for r in rows], float)

    n = len(rows)
    det, reach, att, suc = col("detected"), col("reachable"), col("attached"), col("success")
    le = col("loc_err_mm")[det == 1]
    print("\n================ SUMMARY ================")
    print(f"trials                : {n}   (wall time {time.time() - t0:.0f} s)")
    print(f"detection rate        : {det.mean() * 100:.0f} %")
    print(f"IK-reachable rate     : {reach.mean() * 100:.0f} %")
    print(f"grasp (attach) rate   : {att.mean() * 100:.0f} %")
    print(f"pick success rate     : {suc.mean() * 100:.0f} %")
    if le.size:
        print(f"localisation error    : mean {le.mean():.2f} mm, std {le.std():.2f}, max {le.max():.2f} mm")
        print(f"   xy {np.nanmean(col('loc_err_xy_mm')):.2f} mm, z bias {np.nanmean(col('loc_err_z_mm')):.2f} mm")
    print(f"joint tracking error  : RMS {np.nanmean(col('track_rms_mrad')):.2f} mrad, "
          f"worst {np.nanmax(col('track_max_mrad')):.1f} mrad")
    print(f"observer velocity err : RMS {np.nanmean(col('obs_vel_rms')):.4f} rad/s")
    print(f"files in {OUT}")

    if ctx is not None:
        print("Trials finished - close the MuJoCo window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
        ctx.__exit__(None, None, None)
    sim.cam.renderer.close()          # avoids the GLFW teardown traceback on Windows


if __name__ == "__main__":
    main()
