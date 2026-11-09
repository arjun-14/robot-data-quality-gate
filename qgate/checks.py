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
    "XSIG-002": "state copied from action",
    "EPIS-001": "episode length",
    "EPIS-002": "idle fraction",
    "EPIS-003": "too short",
    "EPIS-004": "idle start",
    "EPIS-005": "idle end",
}


@dataclass
class Finding:
    code: str
    status: str
    detail: str
    frames: list = field(default_factory=list)
    span: tuple | None = None      # (first frame, end frame) of the stretch flagged, when it is one stretch (idle start/end)

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


def too_short(ep, th):
    """Under one second of frames: a recording started and stopped by mistake, nothing to learn from."""
    s = ep.n / ep.fps
    return _f("EPIS-003", ep.n < ep.fps, f"only {ep.n} frame{'s' if ep.n != 1 else ''} ({s:.2f} s), under 1 s", f"{s:.1f} s")


# the robot "moves" once any channel is this fraction of the episode's largest channel range away from its start value.
# Not each channel's own range: a joint that barely moves has a tiny range, so an operator brushing it would count
# (maux/so100_test ep 4: 0.26 deg wrist twitch at 6 s, real start at 21.6 s).
IDLE_MOVE = 0.02
IDLE_MAX_S = 3.0     # longest still stretch at either end before warning (research T8's choice, not sourced)


def first_move(ep, from_end=False):
    """Frames before the first (or after the last) frame where any state/action channel has moved; None if nothing moves."""
    X = np.hstack([np.nan_to_num(ep.state), np.nan_to_num(ep.action)])
    if from_end:
        X = X[::-1]
    moving = (np.abs(X - X[0]) > IDLE_MOVE * np.ptp(X, axis=0).max()).any(axis=1)
    return int(np.argmax(moving)) if moving.any() else None


def idle_start(ep, th):
    """Robot still for more than 3 s before the demonstration starts: teaches the policy to wait at the start pose."""
    k = first_move(ep)
    if k is None:      # nothing ever moves: still for the whole episode
        k = ep.n
    f = _f("EPIS-004", k / ep.fps > IDLE_MAX_S, (f"still for the whole episode ({k / ep.fps:.1f} s); nothing moves" if k == ep.n else
           f"still for the first {k / ep.fps:.1f} s; trim the first {k} frames"),
           f"still throughout, but only {k / ep.fps:.1f} s" if k == ep.n else f"moves after {k / ep.fps:.1f} s", frames=range(k), warn=True)
    f.span = (0, k) if f.status != PASS else None
    return f


def idle_end(ep, th):
    """Robot parked for more than 3 s after the demonstration ends (often the recording timer running out)."""
    k = first_move(ep, from_end=True)
    if k is None:      # nothing ever moves: still for the whole episode
        k = ep.n
    f = _f("EPIS-005", k / ep.fps > IDLE_MAX_S, (f"still for the whole episode ({k / ep.fps:.1f} s); nothing moves" if k == ep.n else
           f"still for the last {k / ep.fps:.1f} s; trim the last {k} frames"),
           f"still throughout, but only {k / ep.fps:.1f} s" if k == ep.n else f"still for {k / ep.fps:.1f} s at the end",
           frames=range(ep.n - k, ep.n), warn=True)
    f.span = (ep.n - k, ep.n) if f.status != PASS else None
    return f


def state_copy(ep, th, atol=1e-6):
    """State is a copy of action, or action is the recorded state shifted one frame: the policy would learn to copy its input."""
    S, A = np.nan_to_num(ep.state), np.nan_to_num(ep.action)
    if S.shape != A.shape or ep.n < 2:
        return Finding("XSIG-002", PASS, "state and action differ in width; not comparable")
    how = ("state is identical to action" if np.allclose(S, A, atol=atol) else
           "action is the next frame's state" if np.allclose(A[:-1], S[1:], atol=atol) else
           "action is the previous frame's state" if np.allclose(A[1:], S[:-1], atol=atol) else None)
    return _f("XSIG-002", how is not None, f"{how} on every frame (copied, not recorded separately)",
              "state and action recorded separately")


ALL_CHECKS = [time_grid, frame_continuity, video_count, nan_inf, range_limit, vel_spike, stuck, flat_gripper, lag, length, idle,
              no_signal]
# Checks that need no calibrated thresholds (ts_tol is LeRobot's own constant): usable on any LeRobot dataset as-is.
STRUCTURAL_TH = {"ts_tol": 1e-4}
# Checks shown on the Any dataset page: need no thresholds and each caught real faults in the Hub sample
# (tickets/T8-research-state-action-checks.md). TIME-001/002 and JOINT-001 never fired on Hub data, so they are not here.
UNIVERSAL = [no_signal, too_short, idle_start, idle_end, state_copy]
UNIVERSAL_CODES = ("JOINT-005", "EPIS-003", "EPIS-004", "EPIS-005", "XSIG-002")
GATE_CODES = tuple(c for c in CODES if c not in UNIVERSAL_CODES or c == "JOINT-005")   # codes ALL_CHECKS reports


def universal_verdict(ep):
    """(verdict, findings) from the universal checks alone: what the Any dataset page shows the moment an episode loads."""
    return run_checks(ep, STRUCTURAL_TH, UNIVERSAL)


# Findings that say nothing about one episode when every episode shows them: state equal to action across a whole dataset is how
# it was recorded (a sim exporter, a labelling convention), while in a few episodes of many it is a fault.
CONVENTION_CODES = ("XSIG-002",)


def dataset_conventions(findings_per_episode):
    """Codes in CONVENTION_CODES that failed in every episode checked (two or more: one episode cannot show a convention)."""
    if len(findings_per_episode) < 2:
        return ()
    return tuple(c for c in CONVENTION_CODES
                 if all(any(f.code == c and f.status == FAIL for f in fs) for fs in findings_per_episode))


def relax(findings, codes):
    """(verdict, findings) with the failures of `codes` turned into passes that say why, for a dataset that does this throughout."""
    out = [Finding(f.code, PASS, f"{f.detail}; same in every episode checked, so it looks deliberate", f.frames, f.span)
           if f.code in codes and f.status == FAIL else f for f in findings]
    return max(out, key=lambda f: RANK[f.status]).status, out


def run_checks(ep, th, checks=None):
    findings = [c(ep, th) for c in (checks or ALL_CHECKS)]
    worst = max(findings, key=lambda f: RANK[f.status]).status
    return worst, findings
