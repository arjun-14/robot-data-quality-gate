"""The streaming gate must agree with the batch gate, and must never raise something the batch gate would not."""
import numpy as np
import pytest
from bench.inject import FAULTS, inject
from qgate.checks import RANK, run_checks
from qgate.live import END_ONLY, PENDING, LiveGate
from tests.test_checks import make_episode


@pytest.fixture(scope="module")
def th():
    from qgate.thresholds import calibrate
    return calibrate([make_episode(i) for i in range(20)])


def stream(ep, th):
    g = LiveGate(th, ep.fps, ep.video_frames, ep.idx)
    online = []
    for k in range(ep.n):
        online += g.push(ep.ts[k], ep.frame_index[k], ep.state[k], ep.action[k])
    before = dict(g.status)
    g.finish()
    return g, before, online


def cases(th):
    yield "clean", make_episode(99)
    for fault in FAULTS:
        for sev in range(3):
            yield f"{fault}-{sev}", inject(make_episode(99), fault, sev, th, np.random.default_rng(sev))


def test_final_verdict_matches_batch(th):
    for name, ep in cases(th):
        g, before, _ = stream(ep, th)
        verdict, findings = run_checks(ep, th)
        assert g.verdict == verdict, name
        for f in findings:
            if f.code in END_ONLY:
                continue
            # every frame-by-frame check reached the batch status before the episode ended
            assert before[f.code] == f.status, (name, f.code, before[f.code], f.status)


def test_online_never_overcalls(th):
    for name, ep in cases(th):
        g, before, online = stream(ep, th)
        final = {f.code: f.status for f in g.final[1]}
        for e in online:
            assert RANK[e["status"]] <= RANK[final[e["code"]]], (name, e)


def test_faults_caught_before_the_end(th):
    """Faults with a frame-local signature are raised while the episode is still streaming."""
    for fault, code in (("nan_values", "JOINT-001"), ("range_violation", "JOINT-002"), ("non_monotonic_ts", "TIME-001")):
        ep = inject(make_episode(99), fault, 2, th, np.random.default_rng(0))
        g, before, online = stream(ep, th)
        assert any(e["code"] == code for e in online), fault
        assert g.first[code][0] < ep.n - 1


def push_all(g, ep):
    out = []
    for k in range(ep.n):
        out += g.push(ep.ts[k], ep.frame_index[k], ep.state[k], ep.action[k])
    return out


def test_placeholder_zeros_fail_no_signal_at_episode_end():
    """robofac-style placeholder: state and action are all zeros for the whole episode."""
    from qgate.dataset import Episode
    n = 90
    ep = Episode(0, 30.0, np.zeros((n, 8)), np.zeros((n, 8)), np.arange(n) / 30, np.arange(n))
    g = LiveGate(None, 30.0)
    push_all(g, ep)
    assert g.status["JOINT-005"] == PENDING
    g.finish()
    assert g.status["JOINT-005"] == "fail" and g.verdict == "fail"


def test_duplicated_frame_is_flagged_on_that_frame_without_thresholds():
    ep = make_episode(0, n=120)
    ep = inject(ep, "duplicated_rows", 0, {}, np.random.default_rng(3))      # one row written twice
    dup = int(np.where(np.diff(ep.frame_index) == 0)[0][0]) + 1
    g = LiveGate(None, ep.fps)
    events = push_all(g, ep)
    # the repeated row repeats its timestamp too, so both timing checks fire on that frame and nothing else does
    assert sorted((e["code"], e["frame"]) for e in events) == [("TIME-001", dup), ("TIME-002", dup)]
    assert g.status["JOINT-003"] == "off" and g.status["JOINT-002"] == "off"


def test_structural_mode_agrees_with_batch_structural_checks(th):
    from qgate.checks import STRUCTURAL, STRUCTURAL_TH
    for name, ep in cases(th):
        g = LiveGate(None, ep.fps)
        online = push_all(g, ep)
        g.finish()
        verdict, findings = run_checks(ep, STRUCTURAL_TH, STRUCTURAL)
        assert g.verdict == verdict, name
        final = {f.code: f.status for f in findings}
        assert all(RANK[e["status"]] <= RANK[final[e["code"]]] for e in online), name


def test_robot_still_at_first_then_moving_is_not_no_signal():
    ep = make_episode(0, n=150)
    ep.state[:60] = ep.state[60]; ep.action[:60] = ep.action[60]      # waits 2 s before moving
    g = LiveGate(None, ep.fps)
    push_all(g, ep)
    g.finish()
    assert g.status["JOINT-005"] == "pass" and g.verdict == "pass"


def test_end_only_checks_pending_until_finish(th):
    ep = make_episode(99)
    g = LiveGate(th, ep.fps, ep.video_frames)
    for k in range(50):
        g.push(ep.ts[k], ep.frame_index[k], ep.state[k], ep.action[k])
    assert g.status["XSIG-001"] == PENDING and g.status["TIME-003"] == PENDING
