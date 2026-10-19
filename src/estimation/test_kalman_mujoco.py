
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from pathlib import Path

from src.estimation.kalman_filter import KalmanFilter


def main():
    # --------------------------------------------------
    # 1. Load UR3e MuJoCo model
    # --------------------------------------------------
    root = Path(__file__).resolve().parents[2]
    xml_path = root / "universal_robots_ur3e" / "ur3e.xml"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    n = 6
    dt = 0.01
    duration = 5.0
    encoder_std = 0.005
    accel_std = 2.0

    # Check the joint and actuator layout.
    print("Model nq:", model.nq)
    print("Model nv:", model.nv)
    print("Actuators:", model.nu)

    joint_ids = [
        j for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]

    if len(joint_ids) != n:
        raise ValueError(
            f"Expected 6 hinge joints, found {len(joint_ids)}. "
            "Check the UR3e model."
        )

    qpos_ids = model.jnt_qposadr[joint_ids]
    qvel_ids = model.jnt_dofadr[joint_ids]

    if model.nu != n:
        raise ValueError(
            f"Expected 6 position actuators, found {model.nu}."
        )

    # Confirm actuator transmission and type.
    for i in range(n):
        if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError("Expected joint actuators.")

        if model.actuator_trnid[i, 0] != joint_ids[i]:
            raise ValueError(
                "Actuator order does not match joint order."
            )

        if model.actuator_gaintype[i] != mujoco.mjtGain.mjGAIN_FIXED:
            raise ValueError("Unexpected actuator gain type.")

        if not np.isclose(model.actuator_biasprm[i, 1],
                          -model.actuator_gainprm[i, 0]):
            raise ValueError(
                "Actuator may not be a position servo. "
                "Inspect the XML."
            )

    # Use the simulator's actual physics timestep.
    dt = float(model.opt.timestep)
    steps = int(duration / dt)
    time = np.arange(steps) * dt

    # --------------------------------------------------
    # 2. Initialise KF
    # --------------------------------------------------
    kf = KalmanFilter(
        dt=dt,
        encoder_std=encoder_std,
        accel_std=accel_std
    )

    rng = np.random.default_rng(42)

    q_home = data.qpos[qpos_ids].copy()

    q_true_log = []
    qd_true_log = []
    q_measured_log = []
    q_est_log = []
    qd_est_log = []
    q_ref_log = []

    # --------------------------------------------------
    # 3. Run MuJoCo simulation
    # --------------------------------------------------
    for t in time:
        # Command a smooth joint trajectory.
        q_ref = q_home.copy()
        q_ref[0] += 0.25 * np.sin(2 * np.pi * 0.3 * t)

        # Position actuator targets.
        data.ctrl[:n] = q_ref

        mujoco.mj_step(model, data)

        # Ground truth: evaluation only.
        q_true = data.qpos[qpos_ids].copy()
        qd_true = data.qvel[qvel_ids].copy()

        # Simulated noisy joint encoders.
        q_measured = q_true + rng.normal(
            0.0, encoder_std, n
        )

        # KF prediction + measurement update.
        q_hat, qd_hat = kf.step(q_measured)

        q_true_log.append(q_true)
        qd_true_log.append(qd_true)
        q_measured_log.append(q_measured)
        q_est_log.append(q_hat.copy())
        qd_est_log.append(qd_hat.copy())
        q_ref_log.append(q_ref.copy())

    q_true = np.asarray(q_true_log)
    qd_true = np.asarray(qd_true_log)
    q_measured = np.asarray(q_measured_log)
    q_est = np.asarray(q_est_log)
    qd_est = np.asarray(qd_est_log)
    q_ref = np.asarray(q_ref_log)

    # --------------------------------------------------
    # 4. Quantitative evaluation
    # --------------------------------------------------
    raw_rmse = np.sqrt(np.mean(
        (q_measured - q_true) ** 2
    ))

    position_rmse = np.sqrt(np.mean(
        (q_est - q_true) ** 2
    ))

    velocity_rmse = np.sqrt(np.mean(
        (qd_est - qd_true) ** 2
    ))

    print("\nMuJoCo UR3e - Kalman Filter Results")
    print("-----------------------------------")
    print(f"Encoder RMSE:     {raw_rmse:.6f} rad")
    print(f"KF position RMSE: {position_rmse:.6f} rad")
    print(f"KF velocity RMSE: {velocity_rmse:.6f} rad/s")

    # --------------------------------------------------
    # 5. Plot joint 1 position
    # --------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(time, q_true[:, 0], label="MuJoCo ground truth")
    plt.plot(
        time, q_measured[:, 0],
        label="Noisy encoder", alpha=0.3
    )
    plt.plot(time, q_est[:, 0], label="KF estimate")
    plt.plot(
        time, q_ref[:, 0], "--",
        label="Position command", alpha=0.7
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Joint 1 position (rad)")
    plt.title("UR3e MuJoCo - KF Position Estimation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # --------------------------------------------------
    # 6. Plot joint 1 velocity
    # --------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(time, qd_true[:, 0], label="MuJoCo ground truth")
    plt.plot(time, qd_est[:, 0], label="KF estimate")

    plt.xlabel("Time (s)")
    plt.ylabel("Joint 1 velocity (rad/s)")
    plt.title("UR3e MuJoCo - KF Velocity Estimation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
