This project implements Track B (Manipulator: Vision-Guided Pick) using a UR3e robotic arm in MuJoCo with a wrist-mounted RGB-D camera.

The system detects a red cube on a table, estimates its 3D position using image and depth information, and picks it up using Position-Based Visual Servoing (PBVS).

The cube position is estimated once from the camera image and depth data, transformed into the world coordinate frame, and converted into joint targets using damped least-squares inverse kinematics (DLS IK).

To evaluate robustness, the cube position is randomised on every trial, the cube slides across the table before coming to rest, and sensor measurements contain noise. The system is designed to localise and pick up the cube despite these variations.

Project Group Number: 34


Contribution
Part A: Dija
Part B:Alzaid Muhammed 
Part C:Do Nguyen Khoi Nguyen 
Part D: Alzaid Muhammed, Leon Markovic 
Part E: All members
 
| Item | Value |
|---|---|
| Robot / simulator | UR3e, MuJoCo, torque motors |
| Visual servoing | PBVS, eye-in-hand wrist camera |
| Joint estimator | 12-state Kalman filter, encoders on q only, commanded acceleration as input |
| Cube estimator | 4-state EKF [x, y, vx, vy], pixel + depth measurement |
| Controller | Computed torque (feedback linearisation) + per-joint discrete LQR |
| Randomised per trial | cube start (±15 mm), friction, cube mass, lighting |

## 2. Install and run
```
pip install -r requirements.txt
python main.py --single-seed 42 --tag normal
python main.py --single-seed 42 --torque-dist 0.2 --tag disturbed
python main.py --seeds [N] --headless        # batch of seeds, [CHECK in main.py --help]
```
Run from the repo folder. Python [3.9.5], MuJoCo [3.3.7]. Remove `--headless` to watch the viewer.


## 3. Video of Disturbed vs Undisturbed trial with small explanation on sensing/modeling/estimation
<iframe width="100%" height="480" src="https://www.youtube.com/embed/zLxY4an9fT4)" title="Demo video" frameborder="0" allowfullscreen></iframe>

[Watch the demo video](https://youtu.be/zLxY4an9fT4)

