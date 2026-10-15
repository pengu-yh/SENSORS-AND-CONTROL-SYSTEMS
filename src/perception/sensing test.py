"""Sensing check: move to a scan pose, then measure detection error vs ground truth (EVALUATION ONLY)
at three noise levels, and (optionally) show the live camera windows.   python src/perception/check_sensing.py"""
import sys
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.perception.sensing import camera
from src.control.ik import make_ik

XML = ROOT / "universal_robots_ur3e" / "ur3e.xml"
CAM, BOX_BODY, BOX_GEOM = "wrist_camera", "box", "box_geom"
SCAN_TCP = (0.27, 0.10, 0.30)           # fixed scan pose: cup tip 30 cm above the table, over the workspace

m = mujoco.MjModel.from_xml_path(str(XML)); d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
q_scan = make_ik(m)(np.array(SCAN_TCP), d.qpos[:6])
d.qpos[:6] = q_scan; d.ctrl[:6] = q_scan; d.qvel[:] = 0
for _ in range(1500):
    mujoco.mj_step(m, d)                                          # let the arm settle

cam = camera(m, cam_name=CAM)
print("Intrinsic matrix K:\n", cam.K.round(1))
truth = d.xpos[m.body(BOX_BODY).id] + [0, 0, m.geom(BOX_GEOM).size[2]]     # ground truth: box top centre
print("true box top centre (base frame):", truth.round(3), "\n")

print("noise level | found | bias x,y,z [mm]      | std x,y,z [mm]")
for ns in (0, 1, 3):
    E = []
    for _ in range(50):
        rgb, depth = cam.take_photo(d, ns)
        p, det = cam.locate(rgb, depth, d.qpos[:6])               # joint angles q -> camera pose -> base frame
        if p is not None:
            E.append((p - truth) * 1000)
    E = np.array(E)
    if len(E):
        print(f"   x{ns}       | {len(E):2d}/50 | {E.mean(0).round(2)} | {E.std(0).round(2)}")
    else:
        print(f"   x{ns}       |  0/50 | box not detected")

if "--no-gui" not in sys.argv:
    import cv2
    print("\nLive view: press Esc to quit")
    while True:
        rgb, depth = cam.take_photo(d, 1)
        result = cam.locate(rgb, depth, d.qpos[:6])
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if result is not None:
            p, det = result
            cv2.circle(vis, (int(det["u"]), int(det["v"])), 8, (0, 255, 0), 2)
            cv2.putText(vis, f"base xyz = {p.round(3)}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(vis, f"error = {np.linalg.norm(p - truth) * 1000:.1f} mm", (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        else:
            cv2.putText(vis, "box not detected", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("wrist RGB + detection", vis)
        cv2.imshow("wrist depth", (255 * (1 - np.clip(depth, 0, 1))).astype(np.uint8))
        if cv2.waitKey(30) == 27:
            break