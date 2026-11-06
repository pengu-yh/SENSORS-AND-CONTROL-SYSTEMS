---
layout: default
title: UR3e Vision-Guided Tracking and Pick-and-Lift
---

# UR3e Vision-Guided Cube Tracking and Pick-and-Lift

**41014 Sensors and Control in Mechatronics Systems · Integrated Project · Track B**

**Group:** [GROUP NUMBER] · **Members:** [NAMES]  
**[Source repository](https://github.com/pengu-yh/SENSORS-AND-CONTROL-SYSTEMS)** · **[Demonstration video](VIDEO_URL)**

> **Validated nominal result:** 30/30 successful pick-and-lift trials over six cube-motion paths (five trials each). This result describes the tested nominal simulation settings, not guaranteed success under arbitrary disturbances.

## 1. Project overview and system specifications

This project implements a vision-guided pick-and-lift task with a **Universal Robots UR3e** arm in **MuJoCo**. A wrist-mounted RGB-D camera observes a red cube on a tabletop. The robot repeatedly detects the cube, estimates its motion, tracks its predicted position, waits for a stable grasp opportunity, then approaches and lifts it using a simulated suction attachment.

The feedback architecture combines **RGB-D perception**, a **cube Extended Kalman Filter (EKF)**, a **joint-state observer**, **damped-least-squares inverse kinematics (IK)**, and **computed-torque control with discrete-time LQR feedback**. Unlike a one-shot PBVS approach, the cube estimate is updated repeatedly as camera measurements arrive.

| Subsystem | Implementation | Purpose |
|---|---|---|
| Robot and simulation | UR3e in MuJoCo; `robot_model/pick_scene.xml` | Six-joint manipulation and physical trials |
| Sensor | Wrist-mounted RGB-D camera | Red-cube image detection and depth measurement |
| Cube estimator | Nonlinear-measurement EKF, 4 states | Estimate tabletop XY position and velocity |
| Joint estimator | Joint-state observer | Estimate joint positions and velocities from noisy encoders |
| Motion planning | Damped-least-squares IK; stage-based sequence | Generate feasible joint targets |
| Joint control | Computed torque + discrete LQR | Track references using estimated joint state |
| End effector | Simulated suction constraint | Attach and lift cube |

**System block diagram — insert group frame diagram here:**

`RGB-D image/depth → cube detection → Cube EKF → predicted XY target → IK / motion stages → LQR + computed torque → UR3e`  
`Noisy joint encoders → joint observer → estimated joint state → controller, camera pose, IK`

**Coordinate frames:** Define the world/table frame, UR3e base frame, tool/TCP frame and camera optical frame in the final diagram. The EKF state uses **world-frame XY**; camera observations are **pixel coordinates and optical depth**. The camera pose is computed using estimated robot joint positions.

## 2. Mathematical modelling

### 2.1 Robot dynamics and control input

The six-joint manipulator is modelled using

$$M(q)\ddot q + h(q,\dot q)=\tau,$$

where $q\in\mathbb{R}^6$ denotes joint angles, $M(q)$ the mass matrix, $h$ the combined bias forces (including gravity and velocity-dependent effects), and $\tau\in\mathbb{R}^6$ the commanded joint torques. The control input is **joint torque**, not direct joint position.

After nominal model-based dynamic compensation, each joint is approximated by a double integrator with error state $e_i=[q_i-q_{i,ref},\ \dot q_i-\dot q_{i,ref}]^\top$. For controller timestep $\Delta t$:

$$e_{k+1}=A_de_k+B_da_k,\qquad
A_d=\begin{bmatrix}1&\Delta t\\0&1\end{bmatrix},\quad
B_d=\begin{bmatrix}\tfrac12\Delta t^2\\\Delta t\end{bmatrix}.$$

For this idealised compensated joint model, $[B_d\ A_dB_d]$ has rank 2 for $\Delta t>0$, so each joint is controllable. With joint position as the measured output, $C=[1\ 0]$, the idealised double-integrator observability matrix $[C^\top\ (CA_d)^\top]^\top$ has rank 2 for $\Delta t>0$. **These are properties of the simplified model; they are not a numerical controllability/observability proof for the complete nonlinear UR3e.**

**To complete:** Report the actual controller sampling interval, calculated open-loop discrete poles (the ideal double integrator has two poles at 1 per joint), and any numerical model-analysis outputs produced by `src/control/model_analysis.py`.

### 2.2 Cube motion model

The cube EKF estimates

$$x=[p_x,p_y,v_x,v_y]^\top,$$

where positions and velocities are in the world XY plane. A constant-velocity model predicts

$$x_{k+1}=F(\Delta t)x_k+w_k,\qquad
F=\begin{bmatrix}1&0&\Delta t&0\\0&1&0&\Delta t\\0&0&1&0\\0&0&0&1\end{bmatrix}.$$

Acceleration uncertainty is represented by $Q=\sigma_a^2GG^\top$, with

$$G=\begin{bmatrix}\Delta t^2/2&0\\0&\Delta t^2/2\\\Delta t&0\\0&\Delta t\end{bmatrix}.$$

The tabletop geometry provides a fixed cube-top height of **0.04 m** for the observation model; this is a scene assumption rather than live simulator ground-truth feedback.

## 3. RGB-D sensing and calibration

The camera pipeline segments the red object in RGB imagery, obtains its pixel location and depth, and uses camera intrinsics and camera pose to relate the observation to world coordinates. The cube EKF uses the measurement vector

$$z=[u,v,Z]^\top,$$

where $u,v$ are image pixels and $Z$ is optical-axis depth in metres. With $p_c=[X,Y,Z]^\top=R_{wc}^\top(p_w-t_{wc})$, the predicted measurement is

$$h(x)=\begin{bmatrix}f_xX/Z+c_x\\f_yY/Z+c_y\\Z\end{bmatrix}.$$

The implementation calculates the analytical measurement Jacobian $H=\partial h/\partial x$ for EKF correction. Camera pose depends on the estimated joint configuration rather than directly reading true robot joint angles for feedback.

**Calibration evidence to add:** camera calibration method, image or checkerboard/target setup, fitted intrinsics, reprojection or localisation error, calibration file handling, and a figure showing camera/world/TCP frames. The project supports loading a calibration file from `results/calibration.npz`; verify and document the exact deployed calibration before claiming numerical accuracy.

**Figure placeholder:** `![Camera calibration](assets/camera_calibration.png)` — add the actual image and uncomment this line.

## 4. State estimation

### 4.1 Cube Extended Kalman Filter

The EKF predicts the state and covariance:

$$\hat x_k^-=F\hat x_{k-1}^+,\qquad P_k^-=FP_{k-1}^+F^\top+Q.$$

The nonlinear camera measurement is linearised at the predicted state. The innovation and gain are

$$r_k=z_k-h(\hat x_k^-),\qquad S_k=H_kP_k^-H_k^\top+R,$$

$$K_k=P_k^-H_k^\top S_k^{-1},\qquad
\hat x_k^+=\hat x_k^-+K_kr_k.$$

Covariance is updated using the numerically stable **Joseph form**:

$$P_k^+=(I-K_kH_k)P_k^-(I-K_kH_k)^\top+K_kRK_k^\top.$$

The implementation initialises world XY from the first valid RGB-D detection and starts with zero estimated XY velocity. Missing detections do not trigger a measurement update; prediction uncertainty continues to evolve.

### 4.2 Joint-state observer

Simulated encoders supply noisy joint-position measurements. The joint observer estimates both $\hat q$ and $\hat{\dot q}$, enabling feedback without directly using true simulator joint velocities. The controller and camera-pose calculations use the estimated joint configuration. Document the observer's exact state transition, measurement model, process covariance, encoder-noise covariance and consistency checks from `src/estimation/joint_observer.py`.

### 4.3 Estimator tuning parameters

| Parameter | Value in reviewed implementation | Meaning |
|---|---:|---|
| Cube EKF state | 4 variables | World XY position and velocity |
| Cube top height | 0.04 m | Known geometry |
| EKF acceleration standard deviation | 0.30 m/s² | Process-noise tuning |
| Pixel measurement standard deviation | 2 px | RGB measurement model |
| Depth measurement standard deviation | 0.006 m | Depth measurement model |
| Simulated encoder noise standard deviation | 0.0005 rad | Nominal encoder noise |
| Prediction horizon | 0.15 s | Moving-target lead time |
| Joint observer process acceleration scale | 1.0 rad/s² | Default CLI setting; verify implementation |

**Figures to add:** estimated versus true cube XY, velocity estimates, uncertainty bounds, joint position and velocity estimation errors, and observer 3σ coverage. Simulator truth is used **for offline evaluation**, not as the controller's state feedback.

## 5. Controller design

### 5.1 Predicted visual target and inverse kinematics

To compensate for target motion and control delay, the commanded target uses

$$p_{xy,target}=\hat p_{xy}+\tau\hat v_{xy},\qquad \tau=0.15\ \mathrm{s}.$$

A damped-least-squares IK solver computes feasible joint targets for the desired TCP pose. A Jacobian-based velocity feedforward term is calculated from the estimated cube XY velocity and limited before use. This reduces the need to chase a moving target using position feedback alone.

### 5.2 Discrete LQR and computed torque

For each dynamically compensated joint, the discrete LQR minimises

$$J=\sum_{k=0}^{\infty}(e_k^\top Q_Le_k+a_k^\top R_La_k),$$

with nominal weights $Q_L=\operatorname{diag}(2500,1)$ and $R_L=[1]$. The discrete algebraic Riccati equation determines feedback gain $K_L$; desired acceleration is

$$a=-K_Le.$$

The torque command is

$$\tau_{cmd}=\operatorname{clip}\big(M(\hat q)a+h(\hat q,\hat{\dot q}),\ \tau_{min},\tau_{max}\big).$$

This combines LQR error regulation with nonlinear model-based torque compensation. Motor torque saturation is explicitly enforced. There is no integral term, so persistent model bias can produce residual steady-state error.

### 5.3 Pick-and-lift sequence and safety revalidation

The controller tracks the cube while moving, checks that it is sufficiently stable, and then starts the approach and suction sequence. During approach, it revalidates camera freshness and target displacement. To avoid false aborts caused by a single noisy EKF velocity sample, the final validated implementation requires **three consecutive camera frames** above the 4 mm/s estimated-speed threshold before triggering the speed-based approach abort. Stale measurements or sufficiently large target displacement can still abort immediately. The approach counter resets when a new approach begins.

**Figure to add:** state machine or timeline showing tracking, stability check, approach, contact/suction, lift, and abort/retry paths.

## 6. Experimental evaluation

### 6.1 Nominal multi-trajectory regression

The final nominal CSV, `results/run_summary_final30.csv`, contains **30 trials**, using seeds 42–71 and five trials for each of six motion paths.

| Motion path | Successful trials | Total |
|---|---:|---:|
| Circle | 5 | 5 |
| Horizontal | 5 | 5 |
| Vertical | 5 | 5 |
| Diagonal | 5 | 5 |
| Square | 5 | 5 |
| Zigzag | 5 | 5 |
| **Total** | **30** | **30** |

| Recorded metric | Nominal result |
|---|---:|
| Pick-and-lift success | **30/30 (100%)** |
| Accepted camera measurements | **2,869/2,869** |
| IK rejections | **0** |
| Suction attachment successes | **30/30** |
| Mean joint-position observer RMSE | **0.1563 mrad** |
| Mean fraction within recorded 3σ bound | **99.90%** |
| Total approach/countdown abort events | **96** |
| Mean recorded lift height | **146.14 mm** |

The repeated abort events reflect active revalidation and retries, not failed final picks. The success percentage is an observed rate on **this finite set of nominal simulated trials**. The CSV records nominal noise multipliers of 1, zero added torque disturbance/bias, and mass-mismatch factor 1; it does **not** establish robustness under nonzero disturbances.

### 6.2 Disturbance tests and additional evidence

**To complete from verified logs:** provide a separate table for encoder noise, RGB-D noise, external torque disturbance, torque bias, and model mass mismatch. Record the tested parameter values, number of trials, success rate, observer RMSE, 3σ consistency and failure reasons. Do not combine these with the nominal 30/30 result.

**Figures to add:** tracking trajectory, estimation error, observer confidence bounds, control effort, and representative successful/failed runs. Include seed 42 as a reproducible example if it matches the group's recorded video and figures.

## 7. Limitations and engineering decisions

- **Approximate model:** computed-torque control depends on the quality of the MuJoCo dynamics model; mass mismatch and unmodelled disturbances can degrade tracking.
- **No integral action:** the LQR regulator may retain steady-state offset under constant bias.
- **Velocity threshold sensitivity:** cube-speed estimates can spike when the cube is nearly stationary; the three-frame debounce reduces false approach aborts but introduces a small detection delay.
- **Perception dependence:** poor segmentation, missing depth or stale camera frames can prevent valid EKF correction.
- **Calibration and geometry:** fixed cube-top height and camera extrinsics are assumptions; hand-eye calibration should be documented or identified as an unvalidated limitation.
- **Restricted workcell:** tested XY bounds and motion paths are not the full robot workspace.
- **Simulation-only evidence:** successful simulated suction attachment does not validate a physical suction cup or real-world grasp reliability.

**Failure analysis to add:** investigate the group's reported `d_enc`/`d_depth` failure cases and seed 47 only after identifying the exact scripts, conditions and logs. Explain why alternative estimation/control methods were not selected based on actual design comparisons.

## 8. Reproducing the simulation

Use a supported Python version with the UR3e mesh assets and `robot_model/pick_scene.xml` present. In PowerShell from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py --seed 42
```

For a headless nominal batch matching the 30-trial test configuration:

```powershell
python main.py --seed 42 --seeds 30 --headless --tag final30
```

Selected CLI arguments from the reviewed `main.py`:

| Argument | Default | Purpose |
|---|---|---|
| `--seed` | `42` | Starting random seed |
| `--seeds` | `1` | Number of trials |
| `--single-seed` | None | Run one specified seed |
| `--headless` | off | Run without GUI |
| `--paths` | all six paths | Comma-separated trajectory selection |
| `--tau` | `0.15` | Prediction horizon, seconds |
| `--step-limit` | `0.016` | Joint-step limit |
| `--enc-noise` | `1.0` | Encoder noise multiplier |
| `--depth-noise` | `1.0` | Camera noise multiplier |
| `--torque-dist` | `0.0` | Random torque disturbance, Nm |
| `--torque-bias` | `0.0` | Constant torque disturbance, Nm |
| `--mass-mismatch` | `1.0` | Controller mass-model scaling |
| `--ekf-accel` | `0.30` | Cube EKF process-noise scale |
| `--obs-accel` | `1.0` | Joint observer process-noise scale |
| `--q-w` | `2500.0` | LQR position-error weight |
| `--tag` | `nominal` | Output filename label |

**Note:** The code uses `--seeds` as a trial count, not a list of individual seeds. Verify the repository's current `main.py` if it has changed since this draft was prepared.

## 9. Team contributions

| Member | Contribution |
|---|---|
| [Member 1] | [Specific verified contribution] |
| [Member 2] | [Specific verified contribution] |
| [Member 3] | [Specific verified contribution] |
| [Member 4] | [Specific verified contribution] |

Each contribution should reflect actual implementation, testing, documentation and integration work. Use Git history and team records to confirm.

## 10. AI-use declaration

**[Complete according to the course's required AI declaration policy.]** Identify any AI tools used and describe their specific role (for example, explanation, debugging, documentation editing), and confirm that team members reviewed and validated submitted technical claims and results. Do not claim that AI was not used if it was.

---

**Project repository:** [pengu-yh/SENSORS-AND-CONTROL-SYSTEMS](https://github.com/pengu-yh/SENSORS-AND-CONTROL-SYSTEMS)  
**Demo:** [Insert video link] · **Source and experiment logs:** [Insert links to final results and plots]
