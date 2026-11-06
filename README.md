**Repository:** [github.com/pengu-yh/SENSORS-AND-CONTROL-SYSTEMS](https://github.com/pengu-yh/SENSORS-AND-CONTROL-SYSTEMS)

##Sensors and Control Integrated Project (TRACK B)

#Overview
This project implements track B (manipulator: vision guided pick) using a ur3e robotic arm with a wrist mounted RGB-D camera in mujoco with pbvs. The camera detects a red 40mm cube tracks it while it moves around then executes a predictive pick using a combination of EKF, LQR Joint Controllers and arm Kinematics to do this. The cube is detected by its colour, its 3D position then calculated from the image and its depth, and the cameras pose from the forward kinematics of arm is then used to express the position in the base frame of the robot. An EKF is used to track the cube over time and predict (if its moving) where the cube will be, the filtered position is then converted to joint targets by damped least-squares inverse kinematics. Once the cube has been stationary for five frames the arm descends and executes the pick. A separate KF estimates joint angles and velocities from artificial noisy encoder readings, in conjunction with a LQR feedback controller that runs on estimated state. for Robustness each trial randomizes position, friction, mass, lighting and the cube slides across one of six paths before coming to reset and noise can also be added to each trial with simple changes in a the main.py command.


Contribution
Part A: Dija
Part B: Alzaid Muhammed 
Part C: Do Nguyen Khoi Nguyen 
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
<iframe width="100%" height="480" src="https://www.youtube.com/embed/zLxY4an9fT4" title="Demo video" frameborder="0" allowfullscreen></iframe>

[Watch the demo video](https://youtu.be/zLxY4an9fT4)




