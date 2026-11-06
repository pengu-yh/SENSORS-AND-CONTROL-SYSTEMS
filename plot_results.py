import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# CSV: time,q_ref1,...,q_ref6,q_actual1,...,q_actual6 (plus other logged columns, which are ignored)
# Use one trial. q_ref = desired trajectory; q_actual is ground truth, used for evaluation only.
ROOT = Path(__file__).resolve().parents[2]            # this file sits in src/control/
FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "tracking_circle_seed42_nominal.csv"
data = np.genfromtxt(FILE, delimiter=",", names=True, ndmin=1)

time = data["time"]
q_ref = np.column_stack([data[f"q_ref{i}"] for i in range(1, 7)])
q_actual = np.column_stack([data[f"q_actual{i}"] for i in range(1, 7)])

# Check for missing values or mixed trials before calculating results.
if len(time) < 2 or not np.all(np.isfinite(np.column_stack((time, q_ref, q_actual)))):
    raise ValueError("CSV needs at least two complete rows of numbers")
if np.any(np.diff(time) <= 0):
    raise ValueError("Time must increase. Use one trial per CSV")

# Convert angles to degrees so the results are easier to read.
q_ref = np.degrees(q_ref)
q_actual = np.degrees(q_actual)
error = q_ref - q_actual
rmse = np.sqrt(np.mean(error**2, axis=0))
max_error = np.max(np.abs(error), axis=0)
final_error = np.abs(error[-1])

print(f"File: {FILE.name}")
print("Joint | RMSE | Max error | Final error (all in degrees)")
for i in range(6):
    print(i + 1, round(rmse[i], 3), round(max_error[i], 3), round(final_error[i], 3))

# Save the error measurements for all six joints.
results = np.column_stack((np.arange(1, 7), rmse, max_error, final_error))
np.savetxt(ROOT / "results" / "tracking_results_summary.csv", results, delimiter=",", fmt="%.6f",
           header="joint,rmse_deg,max_error_deg,final_error_deg", comments="")

# Compare the desired and actual movement of each joint.
plt.figure(figsize=(10, 8))
for i in range(6):
    plt.subplot(3, 2, i + 1)
    plt.plot(time, q_ref[:, i], "--", label="Desired")
    plt.plot(time, q_actual[:, i], label="Actual")
    plt.title(f"Joint {i + 1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.grid()
    plt.legend()

plt.tight_layout()
plt.savefig(ROOT / "results" / "tracking_results_tracking.png", dpi=150)
plt.show()