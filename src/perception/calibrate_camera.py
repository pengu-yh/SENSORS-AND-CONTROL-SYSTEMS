import sys
from pathlib import Path
import numpy as np
import cv2
import mujoco

# Checkerboard calibration of the simulated RGB-D camera (intrinsics K and lens distortion).
# A 8x6-square board is rendered from many poses, corners are detected, cv2.calibrateCamera solves K.
# The true K (from fovy) is used only to CHECK the result, never to compute it.
# Usage: python src/perception/calibrate_camera.py
ROOT = Path(__file__).resolve().parents[2]
W, H, FOVY = 640, 480, 65.0              # must match the wrist camera in robot_model/pick_scene.xml
SQ, COLS, ROWS = 0.02, 8, 6              # square size [m], number of squares across and down
PATTERN = (COLS - 1, ROWS - 1)           # inner corners that OpenCV looks for: (7, 5)
N_VIEWS = 20

# build the board from boxes: a white base plus black squares in a chessboard pattern
geoms = [f'<geom type="box" size="{(COLS+2)*SQ/2} {(ROWS+2)*SQ/2} 0.0005" pos="0 0 -0.0005" rgba="1 1 1 1" '
         f'contype="0" conaffinity="0"/>']
for i in range(COLS):
    for j in range(ROWS):
        if (i + j) % 2 == 0:
            x, y = (i + 0.5) * SQ - COLS * SQ / 2, (j + 0.5) * SQ - ROWS * SQ / 2
            geoms.append(f'<geom type="box" size="{SQ/2} {SQ/2} 0.0005" pos="{x} {y} 0.0005" rgba="0 0 0 1" '
                         f'contype="0" conaffinity="0"/>')
XML = f"""<mujoco><visual><global offwidth="{W}" offheight="{H}"/></visual>
<worldbody><light pos="0 0 1" dir="0 0 -1" diffuse="1 1 1" ambient="0.6 0.6 0.6"/>
<camera name="cal" pos="0 0 0.3" fovy="{FOVY}"/>
{''.join(geoms)}</worldbody></mujoco>"""

m = mujoco.MjModel.from_xml_string(XML)
d = mujoco.MjData(m)
cam_id = m.camera("cal").id
ren = mujoco.Renderer(m, H, W)

def look_at(pos, target=np.zeros(3)):
    # MuJoCo cameras look along their own -z axis, with +y up
    z = pos - target; z /= np.linalg.norm(z)
    x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).flatten())
    m.cam_pos[cam_id], m.cam_quat[cam_id] = pos, quat

rng = np.random.default_rng(0)
objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQ      # corner grid on the board, z = 0
obj_pts, img_pts, shown = [], [], None

for k in range(N_VIEWS):
    r = rng.uniform(0.22, 0.38)                              # distance from board [m]
    tilt = np.radians(rng.uniform(0, 40))                    # angle away from straight-on
    az = rng.uniform(0, 2 * np.pi)
    pos = r * np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), np.cos(tilt)])
    look_at(pos, target=rng.uniform(-0.01, 0.01, 3) * [1, 1, 0])
    mujoco.mj_forward(m, d)
    ren.update_scene(d, camera="cal")
    rgb = ren.render()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, PATTERN)
    if not ok:
        continue
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
    obj_pts.append(objp); img_pts.append(corners)
    if shown is None:
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.drawChessboardCorners(vis, PATTERN, corners, ok)
        shown = vis

print(f"boards detected in {len(img_pts)} of {N_VIEWS} views")
if len(img_pts) < 8:
    raise SystemExit("too few detections to calibrate")

flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3          # simple lens model: radial k1, k2 only
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (W, H), None, None, flags=flags)

# per-view reprojection error [px]
errs = []
for o, i, rv, tv in zip(obj_pts, img_pts, rvecs, tvecs):
    proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
    errs.append(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, axis=1))))

f_true = (H / 2) / np.tan(np.radians(FOVY) / 2)           # ideal pinhole focal length from fovy (check only)
print("\nCalibrated K:\n", K.round(2))
print("distortion [k1 k2 p1 p2 k3]:", dist.ravel().round(5))
print(f"\nRMS reprojection error: {rms:.3f} px   (per-view mean {np.mean(errs):.3f}, worst {np.max(errs):.3f})")
print(f"\nnominal from fovy: fx = fy = {f_true:.2f}, cx = {W/2}, cy = {H/2}")
print(f"calibrated:        fx = {K[0,0]:.2f}, fy = {K[1,1]:.2f}, cx = {K[0,2]:.2f}, cy = {K[1,2]:.2f}")
print(f"focal length difference: {100 * abs(K[0,0] - f_true) / f_true:.2f} %")

(ROOT / "results").mkdir(exist_ok=True)
cv2.imwrite(str(ROOT / "results" / "calibration_corners.png"), shown)
np.savez(ROOT / "results" / "calibration.npz", K=K, dist=dist, rms=rms)
print("\nsaved results/calibration_corners.png and results/calibration.npz")