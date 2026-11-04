"""Simulated wrist RGB-D camera: intrinsics, projection helpers, calibration."""
import cv2
import mujoco
import numpy as np
from config import CAM_FOVY, DEPTH_NOISE_STD, HEIGHT, OUT, RGB_NOISE_STD, WIDTH


#  CAMERA 
def nominal_K():
    fy = (HEIGHT / 2) / np.tan(np.radians(CAM_FOVY) / 2)
    return np.array([[fy, 0, (WIDTH - 1) / 2],
                     [0, fy, (HEIGHT - 1) / 2],
                     [0, 0, 1.0]])


def world_to_pixel(p_w, K, cam_pos, cam_R):
    """MuJoCo camera frame: x right, y up, z backwards (looks along -z)."""
    pc = cam_R.T @ (np.asarray(p_w) - cam_pos)
    X, Y, Z = pc[0], -pc[1], -pc[2]
    return np.array([K[0, 0] * X / Z + K[0, 2], K[1, 1] * Y / Z + K[1, 2]])

#The camera initially gives me an image coordinate. I use the depth and camera calibration to recover a 3D point, then transform that point into the robot's world coordinate frame
#The camera detects the cube in image coordinates, \((u,v)\).
#  Depth and the camera intrinsic matrix convert that pixel into a 3D camera-frame point. 
# The camera's rotation and position then transform the point into the world frame.
def pixel_to_world(u, v, depth, K, cam_pos, cam_R):
    x = (u - K[0, 2]) * depth / K[0, 0]
    y = -(v - K[1, 2]) * depth / K[1, 1]
    p_cam = np.array([x, y, -depth])
    return cam_pos + cam_R @ p_cam


class Camera:
    def __init__(self, model, data, name="wrist_camera"):
        self.model, self.data, self.name = model, data, name
        self.id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if self.id < 0:
            raise RuntimeError(f"camera {name} not found")
        self.renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        self.K = nominal_K()

    def pose(self):
        return (self.data.cam_xpos[self.id].copy(),
                self.data.cam_xmat[self.id].reshape(3, 3).copy())

    def render(self, rng=None, noise=True, name=None):
        name = name or self.name
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=name)
        rgb = self.renderer.render().copy()
        depth = None
        if name == self.name:
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(self.data, camera=name)
            depth = self.renderer.render().copy()
            self.renderer.disable_depth_rendering()
        if noise and rng is not None:
            if RGB_NOISE_STD > 0:
                rgb = np.clip(rgb + rng.normal(0, RGB_NOISE_STD, rgb.shape), 0, 255).astype(np.uint8)
            if depth is not None and DEPTH_NOISE_STD > 0:
                depth = depth + rng.normal(0, DEPTH_NOISE_STD, depth.shape)
        return rgb, depth

    def calibrate(self):
        """
        Intrinsic calibration from known 3D-2D correspondences.
        9 green markers at known world positions are imaged; with the camera pose
        known, u = fx*X/Z + cx and v = fy*Y/Z + cy are linear in (fx,cx) and (fy,cy),
        so they are solved by least squares. Reports reprojection error.
        """
        m, d = self.model, self.data
        gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"calib_{i}") for i in range(9)]
        mujoco.mj_forward(m, d)
        pts = np.array([d.geom_xpos[g] for g in gids])
        cam_pos, cam_R = self.pose()

        rgb, _ = self.render(noise=False)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (40, 100, 60), (85, 255, 255))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        blobs = [cents[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 8]

        # associate blobs with markers using the nominal projection (association only)
        Kn = nominal_K()
        pairs = []
        for p in pts:
            pred = world_to_pixel(p, Kn, cam_pos, cam_R)
            if not blobs:
                break
            dists = [np.linalg.norm(pred - b) for b in blobs]
            j = int(np.argmin(dists))
            if dists[j] < 30:
                pairs.append((p, blobs[j]))
        if len(pairs) < 4:
            print(f"[calibration] only {len(pairs)} markers found - using nominal intrinsics")
            return self.K

        A_u, b_u, A_v, b_v = [], [], [], []
        for p, (u, v) in pairs:
            pc = cam_R.T @ (p - cam_pos)
            X, Y, Z = pc[0], -pc[1], -pc[2]
            A_u.append([X / Z, 1]); b_u.append(u)
            A_v.append([Y / Z, 1]); b_v.append(v)
        fx, cx = np.linalg.lstsq(np.array(A_u), np.array(b_u), rcond=None)[0]
        fy, cy = np.linalg.lstsq(np.array(A_v), np.array(b_v), rcond=None)[0]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

        errs = [np.linalg.norm(world_to_pixel(p, K, cam_pos, cam_R) - np.array(b)) for p, b in pairs]
        print(f"[calibration] {len(pairs)} markers")
        print(f"  nominal    fx={Kn[0,0]:.2f} fy={Kn[1,1]:.2f} cx={Kn[0,2]:.2f} cy={Kn[1,2]:.2f}")
        print(f"  calibrated fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
        print(f"  reprojection error: mean {np.mean(errs):.3f} px, max {np.max(errs):.3f} px")
        np.savetxt(OUT / "camera_K.txt", K)
        self.K = K
        return K
#The simulated wrist-mounted RGB-D camera captures a colour image and a depth image. 
# Calibration markers help estimate the camera's intrinsic parameters. 
# OpenCV then separates red pixels from the background.
