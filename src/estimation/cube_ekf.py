"""EKF for a cube moving on a tabletop, observed by a wrist RGB-D camera.

State: [world_x, world_y, world_vx, world_vy].
Measurement: [pixel_u, pixel_v, camera_depth_z] of the cube's visible top.
The known top height is set from the scene geometry (not from live ground truth).
"""
import numpy as np


class CubeEKF:
    def __init__(self, dt, K, top_z=0.04, accel_std=0.30,
                 pixel_std=2.0, depth_std=0.006):
        self.dt = float(dt)
        self.K = np.asarray(K, dtype=float)
        self.top_z = float(top_z)
        self.accel_std = float(accel_std)
        self.R = np.diag([pixel_std**2, pixel_std**2, depth_std**2])
        self.x = np.zeros(4, dtype=float)
        self.P = np.diag([0.02**2, 0.02**2, 0.2**2, 0.2**2])
        self.initialized = False

    def predict(self, dt=None):
        if not self.initialized:
            return None
        dt = self.dt if dt is None else float(dt)
        if dt <= 0:
            raise ValueError('dt must be positive')
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        G = np.array([[0.5*dt*dt, 0], [0, 0.5*dt*dt],
                      [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + (self.accel_std**2) * (G @ G.T)
        self.P = (self.P + self.P.T) / 2
        return self.x.copy()

    def _camera_measurement(self, x, cam_pos, cam_R):
        """cam_R maps optical camera coordinates to world coordinates."""
        p_world = np.array([x[0], x[1], self.top_z])
        p_cam = cam_R.T @ (p_world - cam_pos)
        X, Y, Z = p_cam
        if Z <= 0.01:
            raise ValueError('Cube is behind or too close to camera')
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        h = np.array([fx*X/Z + cx, fy*Y/Z + cy, Z])
        dh_dpc = np.array([[fx/Z, 0, -fx*X/(Z*Z)],
                            [0, fy/Z, -fy*Y/(Z*Z)],
                            [0, 0, 1]])
        H = np.zeros((3, 4))
        H[:, :2] = dh_dpc @ cam_R.T[:, :2]
        return h, H

    def update(self, detection, cam_pos, cam_R):
        """Update using sensing.camera.detect() output and sensing.camera.pose(q).

        Returns False for missing detections or invalid camera geometry.
        """
        if detection is None:
            return False
        cam_pos = np.asarray(cam_pos, dtype=float).reshape(3)
        cam_R = np.asarray(cam_R, dtype=float).reshape(3, 3)
        z = np.array([detection['u'], detection['v'], detection['z']], dtype=float)
        if not np.all(np.isfinite(z)):
            return False
        if not self.initialized:
            p_cam = np.asarray(detection['p_cam'], dtype=float).reshape(3)
            p_world = cam_pos + cam_R @ p_cam
            self.x[:] = [p_world[0], p_world[1], 0.0, 0.0]
            self.initialized = True
        try:
            z_pred, H = self._camera_measurement(self.x, cam_pos, cam_R)
        except ValueError:
            return False
        innovation = z - z_pred
        S = H @ self.P @ H.T + self.R
        K_gain = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K_gain @ innovation
        I_KH = np.eye(4) - K_gain @ H
        self.P = I_KH @ self.P @ I_KH.T + K_gain @ self.R @ K_gain.T
        self.P = (self.P + self.P.T) / 2
        return True

    @property
    def position(self):
        return np.array([self.x[0], self.x[1], self.top_z])

    @property
    def velocity(self):
        return self.x[2:4].copy()

    @property
    def speed(self):
        return float(np.linalg.norm(self.velocity))
