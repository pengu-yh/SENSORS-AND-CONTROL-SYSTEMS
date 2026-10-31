"""RGB-D known-size square fitting for upright cubes on a level tabletop.

The centre is estimated from observed top-face boundary and a 40-mm square
model. No MuJoCo box position/ground truth is read. No empirical pixel shift.

IMPORTANT: This is a model-based estimator, not an unrestricted 6D pose solver.
By default cube yaw is known to be zero, as in the supplied scene. Set
fit_yaw=True to estimate yaw (less stable for a square, due to symmetry).
"""
import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares

FLIP = np.diag([1., -1., -1.])


class camera:
    def __init__(self, model, cam_name="wrist_camera", W=640, H=480,
                 depth_noise=(1e-3, 5e-3), rgb_noise=4.0, seed=0,
                 top_z=0.04, cube_size=0.04, face_tolerance=0.0035,
                 min_face_points=30, fit_yaw=False, yaw_hint=0.0):
        self.m, self.cam_name, self.W, self.H = model, cam_name, int(W), int(H)
        self.cam_id = model.camera(cam_name).id
        self.f = (self.H / 2) / np.tan(np.deg2rad(float(model.cam_fovy[self.cam_id])) / 2)
        self.K = np.array([[self.f, 0., self.W/2], [0., self.f, self.H/2], [0., 0., 1.]])
        self.ren = mujoco.Renderer(model, self.H, self.W)
        self.scratch = mujoco.MjData(model)
        self.rng = np.random.default_rng(seed)
        self.depth_noise, self.rgb_noise = depth_noise, rgb_noise
        self.top_z, self.cube_size = float(top_z), float(cube_size)
        self.face_tolerance = float(face_tolerance)
        self.min_face_points = int(min_face_points)
        self.fit_yaw, self.yaw_hint = bool(fit_yaw), float(yaw_hint)

    def take_photo(self, data, noise_scale=1.0):
        self.ren.update_scene(data, camera=self.cam_name)
        rgb = self.ren.render().astype(np.float32)
        self.ren.enable_depth_rendering()
        try:
            depth = self.ren.render().copy()
        finally:
            self.ren.disable_depth_rendering()
        rgb = np.clip(rgb + self.rng.normal(0, self.rgb_noise*noise_scale, rgb.shape),
                      0, 255).astype(np.uint8)
        s0, s1 = self.depth_noise
        sigma = (s0 + s1 * depth**2) * noise_scale
        depth = (depth + self.rng.normal(size=depth.shape)*sigma).astype(np.float32)
        return rgb, depth

    def pose(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,):
            raise ValueError("Expected 6 robot joint angles")
        self.scratch.qpos[:6] = q
        mujoco.mj_forward(self.m, self.scratch)
        R = self.scratch.cam_xmat[self.cam_id].reshape(3, 3) @ FLIP
        return self.scratch.cam_xpos[self.cam_id].copy(), R.copy()

    def project(self, u, v, z):
        return np.array([(u-self.K[0, 2])*z/self.f,
                         (v-self.K[1, 2])*z/self.f, z])

    def detect(self, rgb, depth):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.bitwise_or(cv2.inRange(hsv, (0, 90, 70), (12, 255, 255)),
                              cv2.inRange(hsv, (170, 90, 70), (179, 255, 255)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n < 2:
            return None
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[k, cv2.CC_STAT_AREA] < 25:
            return None
        blob = labels == k
        valid = blob & np.isfinite(depth) & (depth > .05) & (depth < 2.)
        ys, xs = np.nonzero(valid)
        if xs.size < self.min_face_points:
            return None
        u, v, z = float(xs.mean()), float(ys.mean()), float(np.median(depth[valid]))
        return dict(u=u, v=v, z=z, p_cam=self.project(u, v, z),
                    mask=valid, blob=blob)

    def _world_to_pixels(self, pts_world, cam_pos, R_wc):
        pc = (np.asarray(pts_world)-cam_pos) @ R_wc
        z = pc[:, 2]
        uv = np.column_stack((self.f*pc[:, 0]/z + self.W/2,
                              self.f*pc[:, 1]/z + self.H/2))
        return uv

    def _corners(self, cx, cy, yaw):
        h = self.cube_size/2
        local = np.array([[-h,-h], [h,-h], [h,h], [-h,h]])
        c,s = np.cos(yaw),np.sin(yaw)
        rot = np.array([[c,-s],[s,c]])
        xy = local @ rot.T + [cx,cy]
        return np.column_stack((xy, np.full(4,self.top_z)))

    @staticmethod
    def _edge_distances(points, corners):
        """Distance from each 2D observed boundary pixel to closest model edge."""
        ds=[]
        for i in range(4):
            a,b=corners[i],corners[(i+1)%4]
            ab=b-a
            t=np.clip(((points-a)@ab)/(ab@ab),0.,1.)
            proj=a+t[:,None]*ab
            ds.append(np.linalg.norm(points-proj,axis=1))
        return np.min(np.stack(ds,axis=1),axis=1)

    def locate(self, rgb, depth, q):
        det=self.detect(rgb,depth)
        if det is None:
            return None
        cam_pos,R_wc=self.pose(q)
        ys,xs=np.nonzero(det['mask'])
        z=depth[ys,xs].astype(np.float64)
        pc=np.column_stack(((xs-self.W/2)*z/self.f,
                            (ys-self.H/2)*z/self.f,z))
        pw=pc@R_wc.T+cam_pos
        face=np.abs(pw[:,2]-self.top_z)<self.face_tolerance
        if np.count_nonzero(face)<self.min_face_points:
            return None
        face_mask=np.zeros((self.H,self.W),np.uint8)
        face_mask[ys[face],xs[face]]=255
        # Use only the largest connected TOP-FACE component. The depth mask
        # isolates the plane; its contour supplies actual visible boundaries.
        n,labs,stats,_=cv2.connectedComponentsWithStats(face_mask)
        if n<2:
            return None
        k=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        if stats[k,cv2.CC_STAT_AREA]<self.min_face_points:
            return None
        face_mask=(labs==k).astype(np.uint8)*255
        contours,_=cv2.findContours(face_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contour=max(contours,key=cv2.contourArea).reshape(-1,2).astype(float)
        if len(contour)<16:
            return None
        # Subsample for fast, bounded optimization.
        if len(contour)>300:
            contour=contour[np.linspace(0,len(contour)-1,300).astype(int)]
        # Initial centre from the *full* 3D face footprint, not mask centroid.
        # Optimizer then aligns all 4 known-size projected edges to contour.
        xy=pw[face,:2]
        init=np.array([(np.percentile(xy[:,0],2)+np.percentile(xy[:,0],98))/2,
                       (np.percentile(xy[:,1],2)+np.percentile(xy[:,1],98))/2])
        # Multi-start offsets prevent the optimizer matching a single visible
        # edge to the wrong square side. Pixel-space robust loss tolerates noise.
        h=self.cube_size
        seeds=[init,init+[h/4,0],init-[h/4,0],init+[0,h/4],init-[0,h/4]]
        if self.fit_yaw:
            yaws=[self.yaw_hint+d for d in (-.2,0,.2)]
        else:
            yaws=[self.yaw_hint]
        best=None
        for seed in seeds:
            for yaw in yaws:
                x0=np.r_[seed,yaw] if self.fit_yaw else seed
                def residual(x):
                    angle=x[2] if self.fit_yaw else self.yaw_hint
                    uv=self._world_to_pixels(self._corners(x[0],x[1],angle),cam_pos,R_wc)
                    d=self._edge_distances(contour,uv)
                    # Mild centre regularization prevents fitting a single
                    # partial edge with the wrong side of the model.
                    regular=(x[:2]-init)*self.f/0.33 * .12
                    return np.r_[d,regular]
                try:
                    result=least_squares(residual,x0,loss='soft_l1',f_scale=1.,
                                         max_nfev=100,xtol=1e-9,ftol=1e-9)
                except (ValueError,FloatingPointError):
                    continue
                # Use robust pixel error, not just solver cost across starts.
                score=np.mean(np.minimum(residual(result.x)[:-2]**2,9.))
                score+=.02*np.sum(((result.x[:2]-init)*1000)**2)
                if best is None or score<best[0]:
                    best=(score,result.x)
        if best is None:
            return None
        pars=best[1]
        angle=float(pars[2]) if self.fit_yaw else self.yaw_hint
        centre=np.array([pars[0],pars[1],self.top_z])
        corners=self._corners(pars[0],pars[1],angle)
        fitted_uv=self._world_to_pixels(corners,cam_pos,R_wc)
        # Preserve old diagnostic keys, but u/v now represent final fitted
        # centre projection (not the red-blob centroid).
        uv_c=self._world_to_pixels(centre[None,:],cam_pos,R_wc)[0]
        det.update(u=float(uv_c[0]),v=float(uv_c[1]),
                   p_cam=(centre-cam_pos)@R_wc,
                   world_centre=centre,face_points=int(np.count_nonzero(face)),
                   top_face_mask=face_mask,observed_contour=contour,
                   fitted_corners_world=corners,fitted_corners_uv=fitted_uv,
                   fitted_yaw=angle,fit_score_px2=float(best[0]),
                   method='known-size square edge reprojection fit')
        return centre,det

    @staticmethod
    def debug_image(rgb,det):
        """Overlay observed top-face contour and fitted 40mm square."""
        out=rgb.copy()
        if det is None:
            return out
        if 'observed_contour' in det:
            pts=np.round(det['observed_contour']).astype(np.int32).reshape(-1,1,2)
            cv2.polylines(out,[pts],True,(255,255,0),1)
        if 'fitted_corners_uv' in det:
            pts=np.round(det['fitted_corners_uv']).astype(np.int32).reshape(-1,1,2)
            cv2.polylines(out,[pts],True,(0,255,0),2)
            cv2.circle(out,(int(round(det['u'])),int(round(det['v']))),4,(0,255,0),-1)
        return out
