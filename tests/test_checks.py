import numpy as np
import pytest
from qgate.dataset import Episode
from qgate.checks import run_checks
from qgate.thresholds import calibrate
from bench.inject import FAULTS, SEVERITIES, inject


def make_episode(idx, n=400, seed=0):
    rng = np.random.default_rng(seed + idx)
    t = np.arange(n) / 30
    ph = rng.uniform(0, 6, 6)
    state = np.stack([30 * np.sin(0.8 * t + ph[j]) for j in range(5)] + [15 + 12 * np.sin(0.5 * t + ph[5])], axis=1)
    state = np.round(state / 0.0879) * 0.0879          # quantised like a real encoder
    action = np.vstack([state[3:], np.repeat(state[-1:], 3, 0)])   # action leads state by 3 frames
    return Episode(idx, 30.0, state, action, t, np.arange(n), {"top": n, "wrist": n})


@pytest.fixture(scope="module")
def th():
    return calibrate([make_episode(i) for i in range(20)])


def test_clean_episode_passes(th):
    verdict, findings = run_checks(make_episode(99), th)
    assert verdict == "pass", [f for f in findings if f.status != "pass"]


@pytest.mark.parametrize("fault", [f for f in FAULTS])
def test_obvious_fault_is_attributed(fault, th):
    ep = make_episode(99)
    bad = inject(ep, fault, 2, th, np.random.default_rng(1))
    _, findings = run_checks(bad, th)
    got = {f.code for f in findings if f.status != "pass"}
    assert got & FAULTS[fault][2], f"{fault}: expected one of {FAULTS[fault][2]}, got {got}"


def test_injection_does_not_touch_original(th):
    ep = make_episode(99); before = ep.state.copy()
    inject(ep, "velocity_spike", 2, th, np.random.default_rng(1))
    assert np.array_equal(ep.state, before)


def test_subtle_velocity_spike_never_fails(th):
    # 1.5x p99.9 sits under the 2x limit, but the spike stacks on the joint's existing velocity,
    # so it can reach a warn; it must never quarantine the episode.
    verdict, _ = run_checks(inject(make_episode(99), "velocity_spike", 0, th, np.random.default_rng(1)), th)
    assert verdict != "fail"
