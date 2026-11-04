"""Red-cube segmentation and depth-based 3D localisation."""
import cv2
import numpy as np
from config import CUBE_HALF, HEIGHT, MIN_AREA, WIDTH
from camera import pixel_to_world


#PERCEPTION 
def red_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, (0, 90, 60), (12, 255, 255)) | cv2.inRange(hsv, (168, 90, 60), (180, 255, 255))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

#localise() rejects detections that are too small, cut off at the image boundary, or lack enough valid depth pixels. 
# It estimates the top face using nearby depth values to reduce errors from the cube's visible side faces.
def localise(rgb, depth, K, cam_pos, cam_R):
    """Return dict with estimated cube-top and cube-centre world positions, or None."""
    m = red_mask(rgb)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[best, cv2.CC_STAT_AREA])
    if area < MIN_AREA:
        return None
    x, y, w, h = [int(v) for v in stats[best, :4]]
    if x == 0 or y == 0 or x + w >= WIDTH or y + h >= HEIGHT:
        return None                                    # cube cut by the image border
    blob = lab == best

    dvals = depth[blob]
    dvals = dvals[np.isfinite(dvals) & (dvals > 0)]
    if dvals.size < MIN_AREA:
        return None
    # Top face = nearest surface of the blob. Using only top-face pixels removes the
    # bias that visible side faces would add to the centroid.
    d0 = np.percentile(dvals, 10)
    top = blob & (depth < d0 + 0.006) & (depth > 0)
    if top.sum() < 50:
        return None
    vs, us = np.nonzero(top)
    u, v = us.mean(), vs.mean()
    d_top = float(np.median(depth[top]))

    p_top = pixel_to_world(u, v, d_top, K, cam_pos, cam_R)
    p_ctr = p_top - np.array([0, 0, CUBE_HALF])
    return dict(p_top=p_top, p_center=p_ctr, uv=(u, v), bbox=(x, y, w, h), area=area, depth=d_top)
