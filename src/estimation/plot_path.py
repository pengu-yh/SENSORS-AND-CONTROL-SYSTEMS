import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Plots the cube's path on the table (x vs y): true path vs the EKF estimate.
# Usage: python src/estimation/plot_path.py [csv path]
ROOT = Path(__file__).resolve().parents[2]
FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "tracking_circle_seed42_normal.csv"
data = np.genfromtxt(FILE, delimiter=",", names=True, ndmin=1)
moving = data["time"] < 6.5                      # the scripted motion stops at about 6 s
plt.figure(figsize=(6, 6))
plt.plot(data["cube_x"][moving], data["cube_y"][moving], "g", label="true cube path")
plt.plot(data["ekf_x"][moving], data["ekf_y"][moving], "k--", lw=1, label="EKF estimate")
plt.scatter(data["cube_x"][0], data["cube_y"][0], c="g", label="start")
plt.axis("equal"); plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.legend()
plt.title(f"Cube path on the table: {FILE.name}")
plt.savefig(ROOT / "results" / "cube_path.png", dpi=130)
plt.show()