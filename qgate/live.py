"""Streaming gate: feed an episode one frame at a time and get findings the moment they become true.

Eight checks can decide on a frame-by-frame basis (timestamps, frame index, NaN, range, velocity, stuck joint,
too-long episode, gripper clearing its minimum range). The rest need the whole episode (video frame count,
action/state lag, idle fraction, too-short episode, a gripper that never moved) and stay "pending" until
`finish()`. At `finish()` the batch gate runs on the buffered frames, so the final verdict is exactly what
`run_checks` would give; tests/test_live.py checks that every online status agrees with it.
"""
from __future__ import annotations
import numpy as np
from .checks import CODES, FAIL, PASS, RANK, STRUCTURAL, STRUCTURAL_TH, WARN, run_checks
from .dataset import Episode

PENDING = "pending"
OFF = "off"            # check not run: no thresholds for this dataset
# checks that can only be decided with the complete episode (they show as pending while streaming)
END_ONLY = {"TIME-003", "GRIP-001", "XSIG-001", "EPIS-001", "EPIS-002", "JOINT-005"}
STRUCTURAL_CODES = {"TIME-001", "TIME-002", "JOINT-001", "JOINT-005"}


class LiveGate:
    """th=None runs only the structural checks (no calibrated thresholds needed); the rest report OFF."""

    def __init__(self, th: dict | None, fps: float, video_frames: dict | None = None, idx: int = -1):
        self.full = th is not None
        th = th if self.full else dict(STRUCTURAL_TH)
        self.th, self.fps, self.idx, self.video_frames = th, float(fps), idx, dict(video_frames or {})
        if self.full:
            self.lo, self.hi = np.array(th["range_lo"]), np.array(th["range_hi"])
            self.vlim = np.array(th["vel_limit"])
        self.ts, self.fi, self.S, self.A = [], [], [], []   # raw frames, handed to the batch gate at finish()
        self.Sn = []                                         # NaN-cleaned state, used by the running checks
        self.status = {c: (OFF if not self.full and c not in STRUCTURAL_CODES else PENDING if c in END_ONLY else PASS)
                       for c in CODES}
        self.detail = {c: "" for c in CODES}
        self.frames = {c: [] for c in CODES}     # every frame flagged so far, per code
        self.first = {}                           # code -> (frame, status) when it first left pass/pending
        self.peak_vel, self.idle_rows, self.g_lo, self.g_hi = 0.0, 0, np.inf, -np.inf
        self.final = None                         # (verdict, findings) once finished

    @property
    def n(self) -> int:
        return len(self.ts)

    def _raise(self, code, status, frame, detail, events):
        self.frames[code].append(frame)
        self.detail[code] = detail
        was = self.status[code]
        if RANK[status] > RANK.get(was, -1):      # first time, or warn -> fail
            self.status[code] = status
            self.first.setdefault(code, (frame, status))
            events.append(dict(frame=frame, code=code, status=status, detail=detail, escalated=was == WARN))

    def push(self, ts: float, frame_index: int, state, action) -> list[dict]:
        """Ingest one frame. Returns the findings this frame raised (usually none)."""
        th, t, ev = self.th, self.n, []
        s_raw, a_raw = np.asarray(state, float), np.asarray(action, float)
        s = np.nan_to_num(s_raw)
        prev_ts = self.ts[-1] if self.ts else None
        prev_fi = self.fi[-1] if self.fi else None
        prev_s = self.Sn[-1] if self.Sn else None
        self.ts.append(float(ts)); self.fi.append(int(frame_index)); self.S.append(s_raw); self.Sn.append(s); self.A.append(a_raw)

        dev = abs(ts - frame_index / self.fps)
        if dev > th["ts_tol"] or (prev_ts is not None and ts <= prev_ts):
            why = f"{dev * 1000:.2f} ms off the 1/fps grid" if dev > th["ts_tol"] else "timestamp did not increase"
            self._raise("TIME-001", FAIL, t, f"frame {t}: {why}", ev)
        if (t == 0 and frame_index != 0) or (prev_fi is not None and frame_index - prev_fi != 1):
            step = "starts at " + str(frame_index) if t == 0 else f"jumps {prev_fi} → {frame_index}"
            self._raise("TIME-002", FAIL, t, f"frame_index {step}", ev)
        if not (np.isfinite(s_raw).all() and np.isfinite(a_raw).all()):
            self._raise("JOINT-001", FAIL, t, f"NaN/Inf at frame {t}", ev)
        if self.status["JOINT-005"] == PENDING:   # decided at the end (a robot may sit still at first); report progress
            a0, s0 = np.nan_to_num(self.A[0]), self.Sn[0]
            self.moved = getattr(self, "moved", False) or bool((s != s0).any() and (np.nan_to_num(a_raw) != a0).any())
            self.detail["JOINT-005"] = "signals change" if self.moved else f"state/action unchanged for {(t + 1) / self.fps:.1f} s so far"
        if not self.full:
            return ev
        out = (s < self.lo) | (s > self.hi)
        if out.any():
            self._raise("JOINT-002", FAIL, t, f"joint {int(np.argmax(out))} outside calibrated range at frame {t}", ev)

        if prev_s is not None:
            v = np.abs(s - prev_s) * self.fps
            ratio = float((v / self.vlim).max())
            self.peak_vel = max(self.peak_vel, ratio)
            if ratio > 1:
                status = FAIL if ratio > th["vel_fail_mult"] else WARN
                self._raise("JOINT-003", status, t - 1, f"velocity {ratio:.2f}x limit at frame {t}", ev)
            self.idle_rows += bool((v < th["idle_vel"]).all())

        w = th["stuck_window"]
        start = t - w + 1
        if start >= 0 and start % 5 == 0:        # same window stride as the batch check
            win = np.array(self.Sn[start:t + 1])
            ptp = win.max(0) - win.min(0)
            for j in range(len(s) - 1):
                if ptp[j] == 0 and np.delete(ptp, j).max() > th["stuck_other_motion"]:
                    self._raise("JOINT-004", FAIL, start, f"joint {j} frozen for {w} frames from frame {start}", ev)

        if t + 1 > th["len_hi"] and self.status["EPIS-001"] == PENDING:
            self._raise("EPIS-001", WARN, t, f"already {t + 1} frames, above the {th['len_hi']:.0f} limit", ev)

        self.g_lo, self.g_hi = min(self.g_lo, s[-1]), max(self.g_hi, s[-1])
        if self.status["GRIP-001"] == PENDING and self.g_hi - self.g_lo >= th["grip_min_range"]:
            self.status["GRIP-001"] = PASS
            self.detail["GRIP-001"] = f"gripper cleared its minimum range at frame {t}"
        return ev

    def idle_so_far(self) -> float:
        return self.idle_rows / max(self.n - 1, 1)

    def episode(self) -> Episode:
        return Episode(self.idx, self.fps, np.array(self.S), np.array(self.A), np.array(self.ts),
                       np.array(self.fi, dtype=np.int64), self.video_frames)

    def finish(self) -> list[dict]:
        """End of episode: settle the pending checks with the batch gate. Returns the findings this raised."""
        verdict, findings = run_checks(self.episode(), self.th, None if self.full else STRUCTURAL)
        self.final = (verdict, findings)
        ev = []
        for f in findings:
            if RANK[f.status] > RANK.get(self.status[f.code], 0):     # pending counts as pass here
                ev.append(dict(frame=self.n - 1, code=f.code, status=f.status, detail=f.detail, escalated=False, at_end=True))
                self.first.setdefault(f.code, (self.n - 1, f.status))
            self.status[f.code], self.detail[f.code] = f.status, f.detail
        return ev

    @property
    def verdict(self) -> str:
        if self.final:
            return self.final[0]
        return max((s for s in self.status.values() if s not in (PENDING, OFF)), key=lambda s: RANK[s], default=PASS)

    def findings(self) -> list[dict]:
        """Current findings in Finding.to_dict() shape (pending checks report as pass)."""
        if self.final:
            return [f.to_dict() for f in self.final[1]]
        return [dict(code=c, name=CODES[c], status=PASS if s == PENDING else s, detail=self.detail[c] or
                     ("decided at end of episode" if s == PENDING else "ok so far"), frames=self.frames[c][:40])
                for c, s in self.status.items() if s != OFF]
