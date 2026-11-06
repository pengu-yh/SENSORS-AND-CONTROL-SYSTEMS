<!-- MathJax for the published GitHub Pages site. GitHub renders README math natively. -->
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true
  },
  options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

**Repository:** [github.com/pengu-yh/SENSORS-AND-CONTROL-SYSTEMS](https://github.com/pengu-yh/SENSORS-AND-CONTROL-SYSTEMS)

## Sensors and Control Integrated Project (TRACK B)

# Overview
This project implements Track B (manipulator: vision-guided pick) using a UR3e robotic arm with a wrist-mounted RGB-D camera in MuJoCo with PBVS. The camera detects a red 40 mm cube and tracks it while it moves around, then executes a predictive pick using a combination of an EKF, LQR joint controllers and arm kinematics. The cube is detected by its colour, its 3D position is then calculated from the image and its depth, and the camera’s pose from the arm’s forward kinematics is used to express the position in the robot’s base frame. An EKF tracks the cube over time and predicts where it will be if it is moving. The filtered position is then converted to joint targets by damped least-squares inverse kinematics. Once the cube has been stationary for five frames, the arm descends and executes the pick. A separate KF estimates joint angles and velocities from artificially noisy encoder readings, in conjunction with an LQR feedback controller that runs on the estimated state. For robustness, each trial randomises position, friction, mass and lighting, and the cube slides across one of six paths before coming to rest. Noise can also be added to each trial with simple changes to the main.py command.

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

## 4. Model (Part A)

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
when run, the output is:
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


## 6. State Estimation — Moving-Cube EKF (Part C)

### 6.1 Purpose and State Definition

The wrist RGB-D camera supplies discrete, noisy observations of the moving cube but does not directly measure its velocity. A **four-state Extended Kalman Filter (EKF)** estimates the cube's planar position and velocity in the world frame:

$$
\mathbf{x}_k = [p_x,\ p_y,\ v_x,\ v_y]^T.
$$

The cube's visible top is assumed to have a known world-frame height of **0.04 m** based on scene geometry. This height is not read from live simulator ground truth.

### 6.2 Constant-Velocity Prediction

Between camera measurements, the cube is approximated by a constant-velocity model:

$$
\hat{\mathbf{x}}^-_k = F_k\hat{\mathbf{x}}_{k-1},\qquad
F_k = \begin{bmatrix}1&0&\Delta t&0\cr 0&1&0&\Delta t\cr 0&0&1&0\cr 0&0&0&1\end{bmatrix}.
$$

Changes in velocity are represented by acceleration process noise:

$$
G_k=\begin{bmatrix}\Delta t^2/2&0\cr 0&\Delta t^2/2\cr \Delta t&0\cr 0&\Delta t\end{bmatrix},\qquad
Q_k=\sigma_a^2G_kG_k^T,\qquad
P^-_k=F_kP_{k-1}F_k^T+Q_k.
$$

The implementation uses an acceleration-noise standard deviation of **0.30 m/s²** by default. The prediction interval can use the elapsed time between camera observations.

### 6.3 Nonlinear RGB-D Observation

Each valid detection supplies the measurement **$\mathbf{z}_k=[u,v,Z]^T$**, where $u,v$ are image coordinates and $Z$ is camera-frame depth. The camera position $\mathbf{t}\_{wc}$ and optical-to-world rotation $R\_{wc}$ are obtained from the robot kinematic model.

$$
\mathbf{p}_w=[p_x,p_y,0.04]^T,\qquad
\mathbf{p}_c=R_{wc}^T(\mathbf{p}_w-\mathbf{t}_{wc})=[X_c,Y_c,Z_c]^T.
$$

The predicted pinhole-camera observation is:

$$
h(\mathbf{x}_k)=\begin{bmatrix}
f_xX_c/Z_c+c_x\cr f_yY_c/Z_c+c_y\cr Z_c
\end{bmatrix}.
$$

This measurement is nonlinear because the projected pixel coordinates depend on depth. The filter therefore uses an analytical measurement Jacobian $H_k=\partial h/\partial\mathbf{x}$ rather than treating the RGB-D measurement as a direct linear observation of world position. The measurement Jacobian has zero columns for the velocity states; velocity is inferred from observations over time.

### 6.4 EKF Correction and Uncertainty

For a valid camera detection, the filter computes:

$$
\mathbf{r}_k=\mathbf{z}_k-h(\hat{\mathbf{x}}^-_k),\qquad
S_k=H_kP^-_kH_k^T+R,
$$

$$
K_k=P^-_kH_k^TS_k^{-1},\qquad
\hat{\mathbf{x}}_k=\hat{\mathbf{x}}^-_k+K_k\mathbf{r}_k.
$$

The covariance uses the **Joseph update**:

$$
P_k=(I-K_kH_k)P^-_k(I-K_kH_k)^T+K_kRK_k^T.
$$

This formulation is chosen for numerical robustness. The filter initialises from the first valid RGB-D detection transformed into world coordinates with zero initial velocity. Missing detections or invalid camera geometry are rejected rather than used for a measurement correction.

| EKF parameter | Implemented value |
|---|---|
| State | $[p_x,p_y,v_x,v_y]^T$ (4 states) |
| Process model | Constant velocity |
| Acceleration noise standard deviation $\sigma_a$ | 0.30 m/s² |
| Pixel noise standard deviation | 2.0 px (each axis) |
| Depth noise standard deviation | 0.006 m |
| Measurement covariance $R$ | $\mathrm{diag}(2^2,2^2,0.006^2)$ |
| Initial covariance $P_0$ | $\mathrm{diag}(0.02^2,0.02^2,0.2^2,0.2^2)$ |
| Cube-top height | 0.04 m |
| Lookahead horizon | 0.15 s |

### 6.5 Predictive Target for Visual Tracking

The estimated planar velocity supports a short lookahead target:

$$
\mathbf{p}_{\mathrm{target},xy}=\hat{\mathbf{p}}_{xy}+0.15\hat{\mathbf{v}}_{xy}.
$$

This target is supplied to the inverse-kinematics and motion-control pipeline. It is a **predicted cube target** not a measurement of the robot's actual tool-centre-point position.

### 6.6 Dedicated EKF Evaluation (Test N)

The script `test_N_ekf_visualization.py` evaluates RGB-D localisation and EKF tracking in a **10-second** simulation. The cube is driven along a prescribed moving trajectory for the first **6 seconds** and then stops. Camera observations are requested approximately every **0.10 seconds**. Simulator cube truth is used to drive the test motion and **evaluate error only** not as the EKF measurement.

Run from the repository root:

```powershell
python test_N_ekf_visualization.py
```

The script writes `test_N_results/results.csv` and the seven plots below. **The PNG files must be committed to the repository for these image links to render on GitHub Pages.**

![World X position: truth, raw RGB-D and EKF](test_N_results/01_X_position.png)

*Figure C1. World X position: ground truth, raw RGB-D and EKF estimate.*

![World Y position: truth, raw RGB-D and EKF](test_N_results/02_Y_position.png)

*Figure C2. World Y position: ground truth, raw RGB-D and EKF estimate.*

![Top-down cube trajectory](test_N_results/03_XY_trajectory.png)

*Figure C3. Top-down XY trajectory comparing ground truth, RGB-D detections and EKF estimates.*

![Raw RGB-D versus EKF position error](test_N_results/04_error_comparison.png)

*Figure C4. Two-dimensional position error. This figure should be interpreted alongside measured RMSE; the EKF is not assumed to outperform raw localisation in every interval.*

![EKF speed estimation](test_N_results/05_speed_estimation.png)

*Figure C5. EKF-estimated speed versus the prescribed motion-driver speed; the cube stops at 6 s.*

![Predicted target position](test_N_results/06_predictive_target_explained.png)

*Figure C6. Current EKF X-position and 0.15-second predictive target. This is not a robot TCP tracking plot.*

#### Quantitative Results (Fixed-Scan Test N)

The corrected isolated test detected the cube in **100/100 frames** and accepted **100/100 EKF measurement updates**.

| Interval | Raw RGB-D 2D position RMSE | EKF 2D position RMSE |
|---|---:|---:|
| Moving (0–6 s) | 0.5496 mm | 0.5524 mm |
| Stopped (from 7 s) | 0.4689 mm | 0.4687 mm |
| Entire trial | 0.5195 mm | 0.5198 mm |

After stopping, the mean EKF-estimated speed was **0.000215 m/s** (0.215 mm/s). These results show reliable detection and sub-millimetre position estimation in this particular fixed-scan test. **The EKF does not improve overall position RMSE over raw RGB-D in this trial** (0.5198 mm versus 0.5195 mm). Its additional value is estimating velocity and enabling short-horizon prediction. These figures are from a single dedicated test not a statistical guarantee across randomised trials.

#### Posterior Covariance and ±3σ Consistency

![EKF signed position errors and posterior 3-sigma bounds](test_N_results/07_EKF_uncertainty_3sigma.png)

*Figure C7. Signed EKF X and Y estimation errors compared separately with posterior ±3σ bounds computed from $P_{xx}$ and $P_{yy}$. The bounds express the filter's model-based marginal uncertainty, not a guaranteed error limit.*

The seventh plot is generated by the updated Test N script. Inspect whether the signed errors remain consistent with the reported covariance; no coverage percentage is asserted before examining the actual output. The CSV also records `sigma_x` and `sigma_y` for reproducibility.

**Evaluation scope:** Test N isolates cube sensing and estimation. It uses the simulated joint configuration for camera localisation. It therefore does not validate the full joint-observer and LQR pipeline or disturbance robustness. The separate integrated 30/30 nominal pick-and-lift regression should be reported under system integration not as Test N EKF accuracy.

**Uncertainty evidence:** Figure C7 compares signed estimation errors with posterior marginal ±3σ bounds. The plot requires rerunning the updated Test N script and committing its generated PNG.


## 9. Simulation and Controller Parameters

The following tables summarise the default parameters used in the integrated UR3e moving-cube tracking and pick-and-lift simulation. Values are taken from all the working files of the team. Unless otherwise stated, these are nominal settings, command-line arguments can override selected values.

### 9.1 Simulation and RGB-D Sensing

| Parameter | Default value | Unit / Description |
|---|---|---|
| Robot | UR3e | 6-DOF manipulator |
| Simulator | MuJoCo | Physics simulation |
| Control timestep | `m.opt.timestep` | s; defined by the MuJoCo model |
| Camera resolution | 640 × 480 | pixels |
| Camera sampling interval | 0.10 | s (10 Hz) |
| Cube size | 0.04 | m |
| Cube-top reference height | 0.04 | m |
| RGB noise standard deviation | 4.0 | image intensity units |
| Depth noise standard deviation | 0.001 + 0.005 × depth² | m; depth-dependent |
| Encoder noise standard deviation | 0.0005 | rad |
| Camera noise scale | 1.0 | `--depth-noise` |
| Encoder noise scale | 1.0 | `--enc-noise` |

### 9.2 Cube Extended Kalman Filter

| Parameter | Default value | Unit / Description |
|---|---|---|
| State vector | [px, py, vx, vy] | 4 states |
| Process model | Constant velocity | Planar cube motion |
| EKF prediction interval | 0.10 | s nominal; elapsed time can be used |
| Acceleration noise standard deviation | 0.30 | m/s² |
| Pixel measurement noise | 2.0 | px per axis |
| Depth measurement noise | 0.006 | m |
| Initial position uncertainty | 0.02 | m standard deviation |
| Initial velocity uncertainty | 0.20 | m/s standard deviation |
| Initial velocity estimate | 0 | m/s |
| Prediction horizon | 0.15 | s (`--tau`) |
| Measurement covariance | diag(4, 4, 0.000036) | px², px², m² |

### 9.3 Joint-State Observer

| Parameter | Default value | Unit / Description |
|---|---|---|
| State dimension | 12 | 6 joint positions + 6 velocities |
| Measured states | Joint positions | 6 encoder readings |
| Observer model | Constant acceleration input | Discrete Kalman filter |
| Observer timestep | `m.opt.timestep` | s |
| Acceleration process noise | 1.0 | rad/s² |
| Assumed encoder noise | 0.0005 | rad at nominal scale |
| Observer encoder-noise scale | 1.0 | `--obs-enc-scale` |
| Initial velocity estimate | 0 | rad/s |
| Initial velocity uncertainty | 0.5 | rad/s standard deviation |
| Covariance update | Joseph form | Numerical robustness |

### 9.4 Computed-Torque and LQR Controller

| Parameter | Default value | Unit / Description |
|---|---|---|
| Controller | Computed torque + discrete LQR | Joint-space feedback |
| LQR position weight | 2500 | `q_w` |
| LQR velocity weight | 1.0 | `qd_w` |
| LQR acceleration-effort weight | 1.0 | `r` |
| LQR state | [joint position error, joint velocity error] | Per joint |
| Control timestep | `m.opt.timestep` | s |
| Joint 1–2 torque limits | ±56 | Nm |
| Joint 3 torque limit | ±28 | Nm |
| Joint 4–6 torque limits | ±12 | Nm |
| Reference joint-step limit | 0.016 | rad per camera update |
| Joint-velocity feedforward limit | 0.6 | rad/s |
| Damped least-squares parameter | 0.03 | Jacobian damping |
| Controller model mass scale | 1.0 | `--mass-mismatch` |

The LQR gains are calculated using the discrete algebraic Riccati equation. They are not fixed numerical gains entered manually.

### 9.5 Tracking, Pick and Safety Logic

| Parameter | Default value | Unit / Description |
|---|---|---|
| Moving-cube trajectory stop time | 6.0 | s |
| Temporary pause start | 3.0 | s |
| Temporary pause duration | 0.20 | s (2 camera frames) |
| Stationary speed threshold | 0.003 | m/s |
| Tracking TCP-to-cube tolerance | 0.005 | m |
| Stationary confirmation | 5 | consecutive camera frames |
| Verification duration | 0.30 | s |
| Verification position tolerance | 0.003 | m |
| Maximum fresh-detection age | 0.15 | s |
| Approach target height | 0.12 | m |
| Approach duration | 1.5 | s |
| Approach motion speed threshold | 0.004 | m/s |
| Approach motion debounce | 3 | consecutive camera updates |
| Approach displacement abort threshold | 0.005 | m |
| Descent target height | 0.035 | m |
| Descent duration | 2.0 | s |
| Lift distance command | 0.15 | m |
| Lift duration | 1.5 | s |
| Successful lift threshold | 0.10 | m |
| Maximum trial simulation time | 20.0 | s |

### 9.6 Randomisation and Robustness Testing

| Parameter | Default / Range | Unit / Description |
|---|---|---|
| Cube starting X offset | ±0.015 | m |
| Cube starting Y offset | ±0.015 | m |
| Cube friction coefficient | 0.5–1.2 | Randomised per trial |
| Cube mass | 0.02–0.08 | kg |
| Lighting scale | 0.6–1.2 | Relative multiplier |
| Motion trajectories | 6 | Horizontal, vertical, diagonal, square, circle, zigzag |
| Trajectory amplitude scale | 1.0 | Maximum nominal scale |
| Workspace X | 0.20–0.34 | m |
| Workspace Y | 0.04–0.16 | m |
| Workspace safety margin | 0.005 | m |
| Added random torque disturbance | 0.0 | Nm standard deviation |
| Constant torque bias | 0.0 | Nm |
| Controller mass mismatch | 1.0 | Nominal model multiplier |
| Default random seed | 42 | Reproducibility |
| Default number of trials | 1 | `--seeds` |
| Simulation mode | Viewer enabled | Use `--headless` for batch tests |

**Notes:** The torque limits are defined in `robot_model/pick_scene.xml`. The physics timestep is read from the MuJoCo model rather than hard-coded in `main.py`. The six motion trajectories are generated by scripted simulation motion while the estimator and controller use sensor-derived estimates. Default disturbance values are zero and non-zero robustness settings must be specified explicitly.
