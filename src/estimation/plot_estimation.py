import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Plots the joint observer's estimate against ground truth, with its own +-3 sigma band.
# Ground truth (qd_true, q_actual, cube_x/y) is used for EVALUATION ONLY, never by the controller.
# Usage: python src/estimation/plot_estimation.py [csv path] [joint 1-6]
ROOT = Path(__file__).resolve().parents[2]            # this file sits in src/estimation/
FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "tracking_circle_seed42_nominal.csv"
J = int(sys.argv[2]) if len(sys.argv) > 2 else 2      # joint to plot (2 = shoulder lift, moves the most)

data = np.genfromtxt(FILE, delimiter=",", names=True, ndmin=1)
need = [f"qd_true{J}", f"qd_hat{J}", f"qd_sig{J}", f"q_actual{J}", f"q_hat{J}", f"q_sig{J}"]
missing = [n for n in need if n not in data.dtype.names]
if missing:
    raise SystemExit(f"{FILE.name} has no columns {missing}. Rerun main.py to regenerate the CSV.")

t = data["time"]
qd, qdh, qds = data[f"qd_true{J}"], data[f"qd_hat{J}"], data[f"qd_sig{J}"]     # velocity: truth, estimate, 1-sigma
q, qh, qs = data[f"q_actual{J}"], data[f"q_hat{J}"], data[f"q_sig{J}"]         # angle: truth, estimate, 1-sigma

# numbers for the report
ev, ep = qdh - qd, (qh - q) * 1e3
print(f"File: {FILE.name}   joint {J}")
print(f"velocity error RMS {np.sqrt(np.mean(ev**2)):.4f} rad/s | angle error RMS {np.sqrt(np.mean(ep**2)):.3f} mrad")
print(f"inside +-3 sigma: velocity {100*np.mean(np.abs(ev) < 3*qds):.2f}% | angle {100*np.mean(np.abs(ep) < 3e3*qs):.2f}%")

# figure 1: joint observer
fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
ax[0].plot(t, qd, "g", label="true (ground truth)")
ax[0].plot(t, qdh, "k--", lw=1, label="observer estimate")
ax[0].fill_between(t, qdh - 3 * qds, qdh + 3 * qds, color="gray", alpha=0.3, label="±3σ")
ax[0].set_ylabel(f"joint {J} velocity [rad/s]"); ax[0].legend()
ax[1].plot(t, ev, "r", lw=0.8); ax[1].plot(t, 3 * qds, "k:"); ax[1].plot(t, -3 * qds, "k:")
ax[1].set_ylabel("velocity error [rad/s]"); ax[1].set_ylim(-0.1, 0.1)            # limits hide the startup spike
ax[2].plot(t, ep, "b", lw=0.8); ax[2].plot(t, 3e3 * qs, "k:"); ax[2].plot(t, -3e3 * qs, "k:")
ax[2].set_ylabel("angle error [mrad]"); ax[2].set_xlabel("time [s]"); ax[2].set_ylim(-2, 2)
fig.suptitle(f"Joint observer vs ground truth: {FILE.name}")
fig.tight_layout()
fig.savefig(ROOT / "results" / f"estimation_joint{J}.png", dpi=130)

# figure 2: cube position estimate vs truth (no sigma band: the cube filter's covariance is not logged)
if "ekf_x" in data.dtype.names and "cube_x" in data.dtype.names:
    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for a, c in ((ax2[0], "x"), (ax2[1], "y")):
        a.plot(t, data[f"cube_{c}"], "g", label="true cube position")
        a.plot(t, data[f"ekf_{c}"], "k--", lw=1, label="camera + EKF estimate")
        a.set_ylabel(f"cube {c} [m]"); a.legend()
    ax2[1].set_xlabel("time [s]")
    err = np.hypot(data["cube_x"] - data["ekf_x"], data["cube_y"] - data["ekf_y"]) * 1e3
    ok = np.isfinite(err)
    late = ok & (t > 7) & (t < 13)                       # cube stationary, before the pick                        
    if ok.any():
        msg = f"cube position error: mean {err[ok].mean():.2f} mm, max {err[ok].max():.2f} mm"
        if late.any():
            msg += f" (after 7 s: mean {err[late].mean():.2f} mm)"
        print(msg)
    fig2.suptitle("Cube position: camera + EKF estimate vs ground truth")
    fig2.tight_layout()
    fig2.savefig(ROOT / "results" / "estimation_cube.png", dpi=130)

plt.show()