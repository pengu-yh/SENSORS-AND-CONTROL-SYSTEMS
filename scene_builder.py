"""Builds ur3e_vision_scene.xml from ur3e.xml (table, cube, markers, wrist camera, TCP)."""
import xml.etree.ElementTree as ET
from pathlib import Path
from config import CALIB_XS, CALIB_YS, CAM_FOVY, CAM_POS, CUBE_HALF, CUBE_START_XY, MARKER_R, PROJECT_FOLDER, TABLE_CENTER_X, TABLE_TOP_Z, TCP_Y


def build_scene() -> Path:
    matches = [p for p in PROJECT_FOLDER.rglob("ur3e.xml")]
    if not matches:
        raise FileNotFoundError(
            f"Could not find ur3e.xml under {PROJECT_FOLDER}\n"
            "Put the robot folder inside the project folder, e.g. "
            f"{PROJECT_FOLDER / 'ur3e' / 'ur3e.xml'}, with the 'assets' folder (the .obj meshes) "
            "next to it, and make sure the file is named exactly ur3e.xml.")
    model_path = matches[0]
    scene_path = model_path.with_name("ur3e_vision_scene.xml")

    tree = ET.parse(model_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("ur3e.xml has no <worldbody>")

    # Torque control: drop position actuators (and anything that refers to them).
    for tag in ("actuator", "keyframe", "sensor"):
        for el in root.findall(tag):
            root.remove(el)

    #  lighting / rendering 
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient", "0.45 0.45 0.45")
    headlight.set("diffuse", "0.7 0.7 0.7")
    headlight.set("specular", "0 0 0")
    glob = visual.find("global")

    if glob is None:
        glob = ET.SubElement(visual, "global")
    glob.set("offwidth", "1280")
    glob.set("offheight", "960")

    ET.SubElement(worldbody, "light", {
        "name": "pick_light", "pos": "0.4 0 1.5", "dir": "0 0 -1",
        "diffuse": "0.6 0.6 0.6"})
    ET.SubElement(worldbody, "geom", {
        "name": "pick_floor", "type": "plane", "size": "2 2 0.1",
        "rgba": "0.85 0.85 0.85 1"})

    #  table (collides with robot and cube) 
    half_h = TABLE_TOP_Z / 2
    table = ET.SubElement(worldbody, "body", {
        "name": "pick_table", "pos": f"{TABLE_CENTER_X} 0 {half_h}"})
    ET.SubElement(table, "geom", {
        "name": "table_top", "type": "box",
        "size": f"0.30 0.35 {half_h}",
        "rgba": "0.45 0.48 0.52 1",
        "contype": "3", "conaffinity": "3"})

    #  cube 
    # contype=4 / conaffinity=2: collides with the table (bit 2) but not with the
    # robot geoms (bit 1). The "gripper" is an abstraction, see ur3e_pick.py.
    cx, cy = CUBE_START_XY
    cube = ET.SubElement(worldbody, "body", {
        "name": "target_cube",
        "pos": f"{cx} {cy} {TABLE_TOP_Z + CUBE_HALF}"})
    ET.SubElement(cube, "freejoint", {"name": "target_cube_free"})
    ET.SubElement(cube, "geom", {
        "name": "target_cube_geom", "type": "box",
        "size": f"{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}",
        "rgba": "0.95 0.12 0.08 1", "mass": "0.08",
        "friction": "1.0 0.01 0.001",
        "contype": "4", "conaffinity": "2"})

    #  calibration markers (visual only; hidden during trials) 
    markers = ET.SubElement(worldbody, "body", {"name": "calib_markers", "pos": "0 0 0"})
    k = 0
    for x in CALIB_XS:
        for y in CALIB_YS:
            ET.SubElement(markers, "geom", {
                "name": f"calib_{k}", "type": "sphere", "size": f"{MARKER_R}",
                "pos": f"{x} {y} {TABLE_TOP_Z + MARKER_R}",
                "rgba": "0.1 0.9 0.1 1", "contype": "0", "conaffinity": "0"})
            k += 1

    #  wrist camera, TCP site, visual tool 
    wrist = root.find(".//body[@name='wrist_3_link']")
    if wrist is None:
        raise RuntimeError("Could not find body 'wrist_3_link' in ur3e.xml")

    # xyaxes = (camera X axis, camera Y axis) in the wrist_3_link frame.
    # camera X = wrist +X, camera Y(up) = wrist +Z  =>  camera Z = wrist -Y,
    # and MuJoCo cameras look along -Z, i.e. along wrist +Y: out of the flange.
    # xyaxes is unit-independent (unlike euler, which can be radians or degrees).
    ET.SubElement(wrist, "camera", {
        "name": "wrist_camera",
        "pos": " ".join(str(v) for v in CAM_POS),
        "xyaxes": "1 0 0 0 0 1",
        "fovy": str(CAM_FOVY)})
    ET.SubElement(wrist, "site", {
        "name": "tcp", "pos": f"0 {TCP_Y} 0", "size": "0.006",
        "rgba": "1 1 0 1"})
    ET.SubElement(wrist, "geom", {
        "name": "tool_visual", "type": "cylinder",
        "fromto": f"0 0.09 0  0 {TCP_Y} 0", "size": "0.018",
        "rgba": "0.15 0.15 0.15 1", "contype": "0", "conaffinity": "0"})

    # Fixed overview camera (for report figures).
    ET.SubElement(worldbody, "camera", {
        "name": "overview", "pos": "1.0 -0.9 0.8",
        "xyaxes": "0.67 0.74 0  -0.3 0.27 0.91", "fovy": "50"})

    tree.write(scene_path, encoding="utf-8", xml_declaration=True)
    return scene_path
