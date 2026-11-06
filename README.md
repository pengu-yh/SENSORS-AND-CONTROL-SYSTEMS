**Repository:** [github.com/pengu-yh/SENSORS-AND-CONTROL-SYSTEMS](https://github.com/pengu-yh/SENSORS-AND-CONTROL-SYSTEMS)

## Sensors and Control Integrated Project (TRACK B)

# Overview
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

## 4. Model

**Plant.** The UR3e is modelled as six rigid links driven by joint torques:

```
M(q) q̈ + h(q, q̇) = τ
```

`M(q)` is the mass matrix, `h` holds gravity, Coriolis and centrifugal terms, and `τ` is the vector of six joint torques. We linearise around the scan (cup tip at [0.27, 0.10, 0.30] m, tool down), where the camera first scans the cube, with the nominal torque holding the arm still.


```
ẋ = A x + B u,   y = C x
A = [ 0   I ]    B = [ 0     ]    C = [ I  0 ]
    [ K̃   0 ]        [ M⁻¹ ]
```

`K̃ = −M⁻¹ ∂g/∂q` is the effect of gravity on small deviations. The velocity is not measured, so it has to be estimated by the observer (section 6).

**Analysis** (`src/control/model_analysis.py`):
when ran output:
```
Linearised UR3e: states 12, inputs 6, outputs 6
controllability rank: 12 / 12
observability rank  : 12 / 12
open-loop eigenvalues: [ 0.  +0.j    0.  +0.j   -5.5 +0.j    5.5 +0.j    2.84+0.j   -2.84+0.j
  0.  +1.65j  0.  -1.65j -0.  +1.69j -0.  -1.69j -0.  +0.j    0.  +0.j  ]
per-joint ctrb rank: 2 / 2
per-joint obsv rank: 2 / 2
```

- **Controllability:** rank 12 of 12, so every state can be steered with the torques.
- **Observability:** rank 12 of 12, so the full state, including velocity, can be recovered from the angle measurements.
- **Open-loop poles:** ±5.5, ±2.84, ±1.65j and ±1.69j. The real poles show that the arm is unstable without feedback: gravity acts like negative stiffness, so a small tilt grows. The imaginary pairs are oscillatory modes.








