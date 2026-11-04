"""Fault injection: 13 faults x 3 severities, applied to in-memory copies (originals untouched).

Each fault lists the taxonomy codes that count as a *correct attribution*.
"""
from __future__ import annotations
import numpy as np

SEVERITIES = ("subtle", "moderate", "obvious")


def _keep(e, mask):
    e.state, e.action, e.ts, e.frame_index = e.state[mask], e.action[mask], e.ts[mask], e.frame_index[mask]


def jitter(e, ms, rng, th):
    e.ts = e.ts + rng.normal(0, ms / 1000, e.n)


def drop(e, k, rng, th):
    m = np.ones(e.n, bool); m[rng.choice(np.arange(5, e.n - 5), k, replace=False)] = False; _keep(e, m)


def duplicate(e, k, rng, th):
    reps = np.ones(e.n, int); reps[rng.choice(np.arange(5, e.n - 5), k, replace=False)] = 2
    e.state, e.action, e.ts, e.frame_index = (np.repeat(x, reps, axis=0) for x in (e.state, e.action, e.ts, e.frame_index))


def nonmonotonic(e, k, rng, th):
    for i in rng.choice(np.arange(5, e.n - 5, 4), k, replace=False):
        e.ts[i], e.ts[i + 1] = e.ts[i + 1], e.ts[i]


def drift(e, eps, rng, th):
    e.ts = e.ts * (1 + eps)


def spike(e, mult, rng, th):
    j = int(rng.integers(0, e.state.shape[1] - 1)); t = int(rng.integers(10, e.n - 10))
    e.state[t, j] += mult * th["vel_p999"][j] / e.fps


def range_violation(e, k, rng, th):
    j = int(rng.integers(0, e.state.shape[1])); a = int(rng.integers(10, e.n - 30))
    hi, lo = th["range_hi"][j], th["range_lo"][j]
    e.state[a:a + 15, j] = hi + k * (hi - lo)


def stuck_joint(e, n, rng, th):
    for _ in range(50):
        j = int(rng.integers(0, 5)); a = int(rng.integers(5, max(6, e.n - n - 5)))
        w = e.state[a:a + n]
        if np.delete(np.ptp(w, axis=0), j).max() > th["stuck_other_motion"] * 2:
            break
    e.state[a:a + n, j] = e.state[a, j]


def nan_values(e, k, rng, th):
    for _ in range(k):
        e.state[int(rng.integers(0, e.n)), int(rng.integers(0, e.state.shape[1]))] = np.nan


def gripper_flat(e, factor, rng, th):
    g = e.state[:, -1]; e.state[:, -1] = g.mean() + (g - g.mean()) * factor


def lag_shift(e, m, rng, th):
    e.state = np.concatenate([np.repeat(e.state[:1], m, axis=0), e.state[:-m]])


def truncate(e, frac, rng, th):
    cut = int(round(frac * e.n)); _keep(e, np.arange(e.n) < e.n - cut)


def idle_pad(e, n, rng, th):
    e.state = np.vstack([e.state, np.repeat(e.state[-1:], n, 0)]); e.action = np.vstack([e.action, np.repeat(e.action[-1:], n, 0)])
    e.ts = np.concatenate([e.ts, e.ts[-1] + np.arange(1, n + 1) / e.fps])
    e.frame_index = np.concatenate([e.frame_index, e.frame_index[-1] + np.arange(1, n + 1)])
    e.video_frames = {k: v + n for k, v in e.video_frames.items()}   # assume padded video too


# name -> (function, params per severity, codes that count as correct attribution, unit label)
FAULTS = {
    "timestamp_jitter":  (jitter,          (0.1, 0.5, 2.0),         {"TIME-001"},                 "ms sigma"),
    "dropped_frames":    (drop,            (1, 5, 20),              {"TIME-002", "TIME-003"},     "frames"),
    "duplicated_rows":   (duplicate,       (1, 5, 20),              {"TIME-002", "TIME-003"},     "rows"),
    "non_monotonic_ts":  (nonmonotonic,    (1, 3, 10),              {"TIME-001"},                 "swaps"),
    "clock_drift":       (drift,           (1e-5, 1e-4, 1e-3),      {"TIME-001"},                 "scale error"),
    "velocity_spike":    (spike,           (1.5, 3.0, 10.0),        {"JOINT-003"},                "x p99.9"),
    "range_violation":   (range_violation, (0.01, 0.05, 0.3),       {"JOINT-002"},                "span beyond limit"),
    "stuck_joint":       (stuck_joint,     (30, 90, 200),           {"JOINT-004"},                "frames"),
    "nan_values":        (nan_values,      (1, 5, 20),              {"JOINT-001"},                "samples"),
    "gripper_flat":      (gripper_flat,    (0.5, 0.2, 0.0),         {"GRIP-001"},                 "amplitude kept"),
    "action_state_lag":  (lag_shift,       (2, 5, 10),              {"XSIG-001"},                 "extra frames"),
    "truncation":        (truncate,        (0.01, 0.05, 0.25),      {"TIME-003", "EPIS-001"},     "fraction cut"),
    "idle_padding":      (idle_pad,        (60, 150, 300),          {"EPIS-002"},                 "frames"),
}


def inject(ep, fault, sev, th, rng):
    fn, params, _, _ = FAULTS[fault]
    e = ep.copy(); fn(e, params[sev], rng, th); return e
