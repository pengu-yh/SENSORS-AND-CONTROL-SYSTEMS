"""Shared constants: scene geometry, trial config, controller/sensor settings, paths."""
from pathlib import Path
import numpy as np


PROJECT_FOLDER = Path(__file__).resolve().parent

#  scene geometry (metres) 
TABLE_TOP_Z = 0.10          # table is LOW so the wrist camera has room to look down
TABLE_CENTER_X = 0.45
CUBE_HALF = 0.025           # 5 cm cube
CUBE_START_XY = (0.42, 0.0)

# Everything below is expressed in the wrist_3_link frame.
# From your printed rotation matrix at the home pose: wrist_3 local +Y points
# world-down (out of the flange), local +Z points world +X, local X is world -Y.
TCP_Y = 0.12                # tool tip, along the flange axis (+Y)
CAM_POS = (0.0, 0.09, 0.07) # camera sits 7 cm beside the tool axis, at the flange plane
CAM_FOVY = 70.0             # vertical field of view, degrees

# 3x3 grid of calibration markers (world frame, on the table)
CALIB_XS = (0.34, 0.42, 0.50)
CALIB_YS = (-0.14, 0.0, 0.14)
MARKER_R = 0.008


#  CONFIG 
N_TRIALS = 2
SEED = 1

# Randomisation region for the cube centre (m, world frame)
X_RANGE = (0.34, 0.46)
Y_RANGE = (-0.12, 0.12)

WIDTH, HEIGHT = 640, 480
DT = 0.002                                   
# simulation = control step

# Observation pose: tool-centre-point target, tool pointing down (world frame)

#Your code estimates the cube's top position and creates three targets:
# Pre-grasp: above the cube.
#Grasp: close to its top surface.
#Lift: above the grasp point.
OBS_TCP = np.array([0.35, 0.0, 0.30])
PRE_HEIGHT = 0.12                            
# pre-grasp height above cube top
GRASP_CLEAR = 0.003                         
 # TCP stops this far above the top face
LIFT_HEIGHT = 0.15

# Controller / observer poles (observer ~5x faster than controller)
CTRL_POLES = (-12.0, -14.0)
OBS_POLES = (-60.0, -70.0)
TORQUE_LIMIT = np.array([56.0, 56.0, 28.0, 12.0, 12.0, 12.0])   # UR3e joint limits (Nm)

# Sensor noise (set to 0 to disable)
ENC_NOISE_STD = 1e-4                         # rad, joint encoder
RGB_NOISE_STD = 2.0                          # 8-bit counts
DEPTH_NOISE_STD = 0.001                      # m

# "Gripper" (suction-style abstraction): attaches if the TCP is this close to the cube top
GRASP_TOL_XY = 0.012
GRASP_TOL_Z = (-0.004, 0.012)                # TCP height above true cube top
SUCCESS_LIFT = 0.08                          # cube must rise this much

# Moving cube: it shuttles back and forth along a line (N_CYCLES round trips) at constant
# speed, then keeps going and stops abruptly at a random point on that line.
N_CYCLES = 2                                 # number of back-and-forth round trips
SLIDE_V0 = (0.25, 0.55)                      # m/s, random constant speed
SETTLE_TIME = 0.8                            # s, robot waits after the cube stops

MIN_AREA = 300                               # px, smallest accepted red blob

HOME = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": 0.0,
}
JOINTS = list(HOME.keys())

OUT = Path(__file__).resolve().parent / "stage3_output"
OUT.mkdir(exist_ok=True)
