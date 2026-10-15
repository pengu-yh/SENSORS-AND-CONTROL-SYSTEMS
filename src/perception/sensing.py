import numpy as np
import cv2
import mujoco

FLIP = np.diag([1, -1, -1])  # flip y and z axes to match camera frame


class camera: 

    def __init__(self, model, cam_name="wrist_camera", W=320, H=240, depth_noise=(1e-3, 5e-3), rgb_noise=4.0, seed=0):
        self.m, self.cam_name, self.W, self.H = model, cam_name, W, H #w and h are the width and height of the camera image
        self.cam_id = model.camera(cam_name).id
        fovy = float(model.cam_fovy[self.cam_id])
        self.f = (H / 2) / np.tan(np.radians(fovy) / 2) #gets focal length
        self.K = np.array([[self.f, 0, W/2], [0, self.f, H/2], [0, 0, 1]]) # callibration matrix
        self.ren = mujoco.Renderer(model, H, W) #rendering context
        self.scratch = mujoco.MjData(model) #copy of the simulation data for rendering and figuring out joint positions
        self.rng = np.random.default_rng(seed) #randmly generating noise for the depth and rgb images
        self.depth_noise, self.rgb_noise = depth_noise, rgb_noise 
    def take_photo(self, data, noise_scale = 1.0):
        self.ren.update_scene(data, camera = self.cam_name) #update the scene with the current data and take the photo
        rgb = self.ren.render().astype(np.float32) #render the image as a float32 array
        self.ren.enable_depth_rendering()#enable depth rendering
        depth = self.ren.render().copy() #render the depth image
        self.ren.disable_depth_rendering()#disable depth rendering
        rgb = np.clip(rgb + self.rng.normal(0, self.rgb_noise * noise_scale, rgb.shape), 0, 255).astype(np.uint8) #add noise to the rgb image hence the change in the dtype to uint8
        s0, s1 = self.depth_noise
        sigma = (s0 + s1 * depth ** 2) * noise_scale 
        depth = (depth + self.rng.normal(0, 1, depth.shape) * sigma).astype(np.float32)
        return rgb, depth
    def pose(self, q):
        self.scratch.qpos[:6] = q
        mujoco.mj_kinematics(self.m, self.scratch)
        mujoco.mj_camlight(self.m, self.scratch)
        R = self.scratch.cam_xmat[self.cam_id].reshape(3, 3) @ FLIP
        return self.scratch.cam_xpos[self.cam_id].copy(), R
    def project(self, u, v, z):
        return np.array([(u - self.K[0, 2]) * z / self.K[0, 0],
                         (v - self.K[1, 2]) * z / self.K[1, 1], z])
    def detect(self, rgb, depth):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV) #converts rgb image to hue, saturation and value
        mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) #returns a binary image where the pixels in the range are white and the rest are black and filters for red (our object of interest)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask) #finds the connected components in the binary image and returns the number of components, the labels, the stats and the centroids
        if n < 2:
            return None
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) #finds the largest connected component in the binary image and returns its area
        if stats[k, cv2.CC_STAT_AREA] < 25:
            return None
        blob = lab == k
        d = depth[blob]
        d = d[(d > 0.05) & (d < 0.5)] #filters the depth to be between 5cm and 50cm
        if d.size ==0:
            return None
        top = blob & (np.abs(depth - np.percentile(d, 20)) < 0.012)
        ys, xs = np.nonzero(top)
        if xs.size < 10:
            return None
        u, v, z = xs.mean(), ys.mean(), float(np.median(depth[top]))
        return dict(u=u, v=v, z=z, p_cam=self.project(u, v, z), mask=top)
    def locate(self, rgb, depth, q):
        det = self.detect(rgb, depth)
        if det is None:
            return None
        p, R = self.pose(q)
        return p + R @ det["p_cam"], det
    
        
        
    

        
