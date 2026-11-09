"""Checks that run on any LeRobot dataset, with no calibrated thresholds (the Any dataset page)."""
import numpy as np
from qgate.checks import STRUCTURAL_TH, UNIVERSAL, run_checks
from qgate.dataset import Episode
from tests.test_checks import make_episode


def status(ep):
    _, findings = run_checks(ep, STRUCTURAL_TH, UNIVERSAL)
    return {f.code: f for f in findings}


def test_episode_under_one_second_fails_too_short():
    ep = make_episode(0, n=20)                       # 20 frames at 30 fps = 0.67 s
    assert status(ep)["EPIS-003"].status == "fail"


def test_one_second_episode_is_not_too_short():
    assert status(make_episode(0, n=30))["EPIS-003"].status == "pass"


def still_then_moving(still, n=300, fps=30.0):
    """Robot holds its start pose for `still` frames, then jumps into motion on frame `still`."""
    k = np.arange(n)
    x = np.where(k < still, 0.0, 20 + 10 * np.sin(k / 10))
    state = np.stack([x, 0.5 * x, -x], axis=1)
    return Episode(0, fps, state, state * 1.01 + 1, k / fps, k)


def test_four_seconds_still_at_start_warns_with_frames_to_trim():
    f = status(still_then_moving(120))["EPIS-004"]
    assert f.status == "warn"
    assert "4.0 s" in f.detail and "first 120 frames" in f.detail


def moving_then_still(still, n=300, fps=30.0):
    """Demonstration ends, then the recording runs on with the robot parked for the last `still` frames."""
    e = still_then_moving(still, n, fps)
    return Episode(0, fps, e.state[::-1].copy(), e.action[::-1].copy(), e.ts, e.frame_index)


def test_robot_parked_for_last_five_seconds_warns_idle_end():
    f = status(moving_then_still(150))["EPIS-005"]
    assert f.status == "warn"
    assert "5.0 s" in f.detail and "last 150 frames" in f.detail


def test_still_start_does_not_count_as_idle_end():
    assert status(still_then_moving(150))["EPIS-005"].status == "pass"


def test_state_identical_to_action_fails_state_copy():
    ep = make_episode(0)
    ep.state = ep.action.copy()                      # e.g. a sim exporter writing the command into the state column
    assert status(ep)["XSIG-002"].status == "fail"


def test_action_that_is_just_the_next_state_fails_state_copy():
    ep = make_episode(0)
    ep.action = np.vstack([ep.state[1:], ep.state[-1:]])   # "action" built by shifting the recorded state one frame
    assert status(ep)["XSIG-002"].status == "fail"


def test_real_teleop_action_leading_state_passes_state_copy():
    assert status(make_episode(0))["XSIG-002"].status == "pass"    # follower lags the leader by 3 frames


def test_state_and_action_of_different_width_pass_state_copy():
    ep = make_episode(0)
    ep.action = ep.action[:, :3]                     # e.g. joint state with a Cartesian action
    assert status(ep)["XSIG-002"].status == "pass"


def test_small_twitch_on_a_barely_used_joint_does_not_end_the_idle_start():
    """maux/so100_test ep 4: the leader's wrist roll twitched 0.26 deg at 6 s; the arm really started at 21.6 s."""
    ep = still_then_moving(120)
    roll = np.where(np.arange(ep.n) < 120, 0.0, 1.0)     # a joint that only moves 1 unit in the whole episode
    roll[30:] += 0.05                                     # operator brushes it at 1 s
    ep.state = np.hstack([ep.state, roll[:, None]]); ep.action = np.hstack([ep.action, roll[:, None]])
    f = status(ep)["EPIS-004"]
    assert f.status == "warn" and "first 120 frames" in f.detail


def test_two_second_settle_at_start_is_fine():
    assert status(still_then_moving(60))["EPIS-004"].status == "pass"
