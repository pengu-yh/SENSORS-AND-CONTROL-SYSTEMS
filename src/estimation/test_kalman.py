
import numpy as np
import matplotlib.pyplot as plt

from src.estimation.kalman_filter import KalmanFilter


def main():
    dt = 0.01
    duration = 5.0
    encoder_std = 0.005
    accel_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    rng = np.random.default_rng(42)
    t = np.arange(0, duration, dt)

    # Synthetic 6-joint trajectory
    q_true = np.column_stack([
        0.4 * np.sin(2 * np.pi * 0.3 * t + i * 0.2)
        for i in range(6)
    ])

    qd_true = np.column_stack([
        0.4 * 2 * np.pi * 0.3
        * np.cos(2 * np.pi * 0.3 * t + i * 0.2)
        for i in range(6)
    ])

    # Generate measurements ONCE for fair comparison
    q_measured = q_true + rng.normal(
        0, encoder_std, q_true.shape
    )

    raw_rmse = np.sqrt(np.mean(
        (q_measured - q_true) ** 2
    ))

    results = {}

    print(f"Raw encoder RMSE: {raw_rmse:.6f} rad")
    print()
    print(f"{'accel_std':>10} | {'Position RMSE':>15} | {'Velocity RMSE':>15}")
    print("-" * 48)

    for accel_std in accel_values:
        kf = KalmanFilter(
            dt=dt,
            encoder_std=encoder_std,
            accel_std=accel_std
        )

        q_estimates = []
        qd_estimates = []

        for measurement in q_measured:
            q_hat, qd_hat = kf.step(measurement)
            q_estimates.append(q_hat)
            qd_estimates.append(qd_hat)

        q_estimates = np.array(q_estimates)
        qd_estimates = np.array(qd_estimates)

        pos_rmse = np.sqrt(np.mean(
            (q_estimates - q_true) ** 2
        ))

        vel_rmse = np.sqrt(np.mean(
            (qd_estimates - qd_true) ** 2
        ))

        results[accel_std] = {
            "q": q_estimates,
            "qd": qd_estimates,
            "pos_rmse": pos_rmse,
            "vel_rmse": vel_rmse
        }

        print(
            f"{accel_std:10.1f} | "
            f"{pos_rmse:15.6f} | "
            f"{vel_rmse:15.6f}"
        )

    # Joint 1 position comparison
    plt.figure(figsize=(11, 5))
    plt.plot(t, q_true[:, 0], "k--",
             label="True position", linewidth=2)

    for accel_std in accel_values:
        plt.plot(
            t,
            results[accel_std]["q"][:, 0],
            label=f"accel_std={accel_std}",
            alpha=0.8
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Joint 1 position (rad)")
    plt.title("KF Position - Acceleration Noise Comparison")
    plt.legend()
    plt.grid(True)

    # Joint 1 velocity comparison
    plt.figure(figsize=(11, 5))
    plt.plot(t, qd_true[:, 0], "k--",
             label="True velocity", linewidth=2)

    for accel_std in accel_values:
        plt.plot(
            t,
            results[accel_std]["qd"][:, 0],
            label=f"accel_std={accel_std}",
            alpha=0.8
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Joint 1 velocity (rad/s)")
    plt.title("KF Velocity - Acceleration Noise Comparison")
    plt.legend()
    plt.grid(True)

    # RMSE comparison
    pos_errors = [
        results[a]["pos_rmse"] for a in accel_values
    ]
    vel_errors = [
        results[a]["vel_rmse"] for a in accel_values
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(accel_values, pos_errors, "o-")
    axes[0].axhline(raw_rmse, linestyle="--",
                    label="Raw encoder RMSE")
    axes[0].set_title("Position RMSE")
    axes[0].set_xlabel("accel_std (rad/s²)")
    axes[0].set_ylabel("RMSE (rad)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(accel_values, vel_errors, "o-")
    axes[1].set_title("Velocity RMSE")
    axes[1].set_xlabel("accel_std (rad/s²)")
    axes[1].set_ylabel("RMSE (rad/s)")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
