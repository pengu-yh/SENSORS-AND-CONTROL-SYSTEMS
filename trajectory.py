"""Min-jerk joint-space trajectory generation."""
import numpy as np
from config import DT


# TRAJECTORIES 
def min_jerk(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def plan(waypoints, T):
    """Joint-space reference (q, qdot, qddot) along waypoints with a min-jerk time law."""
    w = np.atleast_2d(waypoints)
    if len(w) == 1:
        w = np.vstack([w, w])
    n = int(round(T / DT))
    s = min_jerk(np.arange(n + 1) / n)
    idx = s * (len(w) - 1)
    i0 = np.minimum(idx.astype(int), len(w) - 2)
    f = (idx - i0)[:, None]
    q = w[i0] * (1 - f) + w[i0 + 1] * f
    k = 31                                              # smooth waypoint kinks (62 ms)
    pad = np.pad(q, ((k // 2, k // 2), (0, 0)), mode="edge")
    ker = np.ones(k) / k
    q = np.stack([np.convolve(pad[:, j], ker, mode="valid") for j in range(q.shape[1])], axis=1)
    v = np.gradient(q, DT, axis=0)
    a = np.gradient(v, DT, axis=0)
    return q, v, a


def hold(q, T):
    n = int(round(T / DT))
    return np.tile(q, (n, 1)), np.zeros((n, 6)), np.zeros((n, 6))
