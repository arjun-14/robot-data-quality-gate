"""Calibrate thresholds from a clean reference set. Every value carries provenance."""
from __future__ import annotations
import numpy as np
from .checks import best_lag, idle_fraction, velocity

SOURCED = "sourced"
DERIVED = "data-derived"
OWN = "own choice"


def calibrate(eps):
    fps = eps[0].fps
    V = np.concatenate([velocity(e) for e in eps])
    p999 = np.percentile(V, 99.9, axis=0)
    S = np.concatenate([e.state for e in eps])
    lo, hi = S.min(0), S.max(0); margin = 0.25 * (hi - lo)
    grip = [float(np.ptp(e.state[:, -1])) for e in eps]
    lens = [e.n for e in eps]
    idles = [idle_fraction(e) for e in eps]
    lags = [best_lag(e) for e in eps]
    th = dict(
        fps=fps, n_calibration_episodes=len(eps),
        ts_tol=1e-4, vid_tol_frames=2,
        vel_p999=p999.tolist(), vel_limit=(2 * p999).tolist(), vel_fail_mult=2.0,
        range_lo=(lo - margin).tolist(), range_hi=(hi + margin).tolist(),
        stuck_window=60, stuck_other_motion=15.0,
        grip_min_range=0.25 * min(grip),
        lag_ref=int(np.median(lags)), lag_tol=2,
        len_lo=0.7 * min(lens), len_hi=1.5 * max(lens),
        idle_vel=1.0, idle_max=1.25 * max(idles),
    )
    th["provenance"] = {
        "ts_tol": [SOURCED, "LeRobot tolerance_s default (1e-4 s); known to false-alarm on float32 rounding and miss near-miss frames (lerobot issue 5187)"],
        "vid_tol_frames": [OWN, "2 frames of slack between video metadata and parquet rows"],
        "vel_limit": [DERIVED, "2 x p99.9 of |velocity| per joint on the calibration episodes; fail above 2x that limit"],
        "range": [DERIVED, "observed min/max per joint widened by 25% of the observed span"],
        "stuck": [OWN, "zero variance over 60 frames while another joint moves more than 15 degrees"],
        "grip_min_range": [DERIVED, "25% of the smallest per-episode gripper range"],
        "lag": [DERIVED, "median best action->state lag on calibration episodes, tolerance +-2 frames"],
        "length": [DERIVED, "0.7 x shortest to 1.5 x longest calibration episode"],
        "idle": [DERIVED, "1.25 x highest idle fraction on calibration episodes"],
    }
    return th
