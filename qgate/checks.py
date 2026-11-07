"""The 11 first-cut checks. Each takes (Episode, thresholds) and returns a Finding."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

PASS, WARN, FAIL = "pass", "warn", "fail"
RANK = {PASS: 0, WARN: 1, FAIL: 2}

# Stable taxonomy codes (see docs/taxonomy.md)
CODES = {
    "TIME-001": "timestamp grid",
    "TIME-002": "frame index continuity",
    "TIME-003": "video frame count",
    "JOINT-001": "NaN / Inf",
    "JOINT-002": "range limit",
    "JOINT-003": "velocity spike",
    "JOINT-004": "stuck joint",
    "JOINT-005": "no signal",
    "GRIP-001": "flat gripper",
    "XSIG-001": "action/state lag",
    "EPIS-001": "episode length",
    "EPIS-002": "idle fraction",
}


@dataclass
class Finding:
    code: str
    status: str
    detail: str
    frames: list = field(default_factory=list)

    @property
    def name(self):
        return CODES[self.code]

    def to_dict(self):
        d = asdict(self); d["name"] = self.name
        return d


def _f(code, bad, detail_bad, detail_ok, frames=(), warn=False):
    status = (WARN if warn else FAIL) if bad else PASS
    return Finding(code, status, detail_bad if bad else detail_ok, [int(x) for x in list(frames)[:40]])


def velocity(ep):
    return np.abs(np.diff(np.nan_to_num(ep.state), axis=0)) * ep.fps


def best_lag(ep, max_lag=20):
    S, A, n = np.nan_to_num(ep.state), np.nan_to_num(ep.action), ep.n
    errs = [np.mean(np.abs(A[: n - k] - S[k:])) for k in range(0, min(max_lag, n - 2) + 1)]
    return int(np.argmin(errs))


def idle_fraction(ep, vel_thresh=1.0):
    return float((velocity(ep) < vel_thresh).all(axis=1).mean())


def time_grid(ep, th):
    dev = np.abs(ep.ts - ep.frame_index / ep.fps)
    bad = np.where(dev > th["ts_tol"])[0]
    nonmono = int((np.diff(ep.ts) <= 0).sum())
    return _f("TIME-001", len(bad) or nonmono,
              f"{len(bad)} samples off the 1/fps grid (max {dev.max()*1000:.2f} ms), {nonmono} non-increasing steps",
              "on the 1/fps grid", bad)


def frame_continuity(ep, th):
    d = np.diff(ep.frame_index); bad = np.where(d != 1)[0]
    return _f("TIME-002", len(bad) or ep.frame_index[0] != 0,
              f"{len(bad)} breaks in frame_index (gaps/duplicates)", "frame_index contiguous", bad)


def video_count(ep, th):
    off = {k: v - ep.n for k, v in ep.video_frames.items() if abs(v - ep.n) > th["vid_tol_frames"]}
    return _f("TIME-003", bool(off), f"video vs rows mismatch (frames): {off}", "video frame counts match rows")


def nan_inf(ep, th):
    bad = ~np.isfinite(ep.state).all(axis=1) | ~np.isfinite(ep.action).all(axis=1)
    return _f("JOINT-001", bad.any(), f"{int(bad.sum())} rows with NaN/Inf", "all finite", np.where(bad)[0])


def range_limit(ep, th):
    S = np.nan_to_num(ep.state, nan=0.0)
    out = ((S < np.array(th["range_lo"])) | (S > np.array(th["range_hi"]))).any(axis=1)
    return _f("JOINT-002", out.any(), f"{int(out.sum())} frames outside calibrated range", "within calibrated range", np.where(out)[0])


def vel_spike(ep, th):
    ratio = velocity(ep) / np.array(th["vel_limit"])
    mx = float(ratio.max()) if len(ratio) else 0.0
    frames = np.where((ratio > 1).any(axis=1))[0]
    status_fail = mx > th["vel_fail_mult"]
    f = _f("JOINT-003", mx > 1, f"peak velocity {mx:.2f}x limit", f"peak velocity {mx:.2f}x limit", frames, warn=not status_fail)
    return f


def stuck(ep, th):
    w = th["stuck_window"]; S = np.nan_to_num(ep.state)
    if ep.n < w:
        return Finding("JOINT-004", PASS, "episode shorter than window")
    win = sliding_window_view(S, w, axis=0)[::5]               # (windows, d, w)
    ptp = win.max(-1) - win.min(-1)
    hits = []
    for j in range(S.shape[1] - 1):   # arm joints only; the gripper is legitimately constant and has GRIP-001
        others = np.delete(ptp, j, axis=1).max(1)
        hits += list(np.where((ptp[:, j] == 0) & (others > th["stuck_other_motion"]))[0] * 5)
    return _f("JOINT-004", bool(hits), f"joint constant for {w}+ frames while others move", "no stuck windows", sorted(set(hits)))


def flat_gripper(ep, th):
    r = float(np.ptp(np.nan_to_num(ep.state)[:, -1]))
    return _f("GRIP-001", r < th["grip_min_range"], f"gripper range {r:.1f} < {th['grip_min_range']:.1f}", f"gripper range {r:.1f}")


def lag(ep, th):
    k = best_lag(ep)
    return _f("XSIG-001", abs(k - th["lag_ref"]) > th["lag_tol"], f"action leads state by {k} frames (expected {th['lag_ref']}±{th['lag_tol']})",
              f"action leads state by {k} frames", warn=True)


def length(ep, th):
    return _f("EPIS-001", not (th["len_lo"] <= ep.n <= th["len_hi"]), f"{ep.n} frames outside [{th['len_lo']:.0f}, {th['len_hi']:.0f}]",
              f"{ep.n} frames", warn=True)


def idle(ep, th):
    f = idle_fraction(ep, th["idle_vel"])
    return _f("EPIS-002", f > th["idle_max"], f"{f:.0%} idle > {th['idle_max']:.0%}", f"{f:.0%} idle", warn=True)


def no_signal(ep, th):
    """State or action never changes in any channel: a placeholder (e.g. zeros) or a dead stream, not robot data."""
    dead = [name for name, x in (("state", ep.state), ("action", ep.action))
            if len(x) and (np.ptp(np.nan_to_num(x), axis=0) == 0).all()]
    zeros = all(not np.nan_to_num(x).any() for x in (ep.state, ep.action))
    return _f("JOINT-005", bool(dead), ("state and action are all zeros (placeholder or missing data)" if zeros else
              f"{' and '.join(dead)} never change{'s' if len(dead) == 1 else ''} (placeholder or missing data)"), "signals change")


ALL_CHECKS = [time_grid, frame_continuity, video_count, nan_inf, range_limit, vel_spike, stuck, flat_gripper, lag, length, idle,
              no_signal]
# Checks that need no calibrated thresholds (ts_tol is LeRobot's own constant): usable on any LeRobot dataset as-is.
STRUCTURAL = [time_grid, frame_continuity, nan_inf, no_signal]
STRUCTURAL_TH = {"ts_tol": 1e-4}


def run_checks(ep, th, checks=None):
    findings = [c(ep, th) for c in (checks or ALL_CHECKS)]
    worst = max(findings, key=lambda f: RANK[f.status]).status
    return worst, findings
