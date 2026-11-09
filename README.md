---
title: qgate
emoji: 🤖
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
short_description: Quality gate for LeRobot datasets on the Hugging Face Hub
---

# robot-data-quality-gate

A quality gate for robot manipulation data. Enter any public LeRobot dataset on the Hugging Face Hub (v2.0, v2.1 or v3.0). It checks every episode, gives each a **pass / warn / fail** verdict with a stable taxonomy code, and lets you open any episode to see its video, its state and action signals, and the stretches the checks flagged.

![qgate checking samsitol/so100_peg01](docs/screenshot.png)

## The app

```bash
pip install -e ".[app]"
streamlit run app/app.py
```

Enter a dataset (`owner/name`) or pick an example. Episodes are read straight from the Hub ([`qgate/hub.py`](qgate/hub.py)). Only the state and action columns are loaded into memory, and nothing is written to disk. Large datasets are checked on an evenly spaced sample unless you ask for every episode.

- **Dataset verdict:** how many episodes passed, warned and failed, and how many tripped each check. A check that fires in every episode is treated as how the dataset was recorded, not as a fault.
- **Episode grid:** one square per episode per check. Click a square to open that episode.
- **Episode view:** the video, with buttons that jump to each flagged stretch. Below it, the state (solid) and commanded action (dotted) signals, with the flagged stretches shaded.

### Checks

These five checks need no calibrated thresholds, use only state and action, and each caught real faults when replayed over 7,792 episodes from 37 Hub datasets ([T8](tickets/T8-research-state-action-checks.md)).

| Code | Check | Verdict | Episodes caught |
|---|---|---|---|
| `JOINT-005` | No signal: state or action never changes | fail | 31 (e.g. robofac's placeholder zeros) |
| `EPIS-003` | Episode under 1 s | fail | 7 |
| `EPIS-004` | Robot still for over 3 s at the start, with the frames to trim | warn | 612 |
| `EPIS-005` | Robot still for over 3 s at the end, with the frames to trim | warn | 488 |
| `XSIG-002` | State copied from action | fail | 32 (robofac sim) |

The timestamp, frame-index and NaN checks never fired on real Hub data, so the app doesn't run them.

## CLI and calibrated checks

```bash
pip install -e ".[dev]"
pytest                                    # unit tests
python scripts/fetch_data.py data/so100   # metadata + parquet for the SO-100 dataset, a few MB, no video
qgate check data/so100                    # exit code 1 if any episode fails
qgate bench data/so100                    # rerun the fault-injection benchmark
```

`qgate check` runs 11 checks with thresholds calibrated on [`lerobot/svla_so100_pickplace`](https://huggingface.co/datasets/lerobot/svla_so100_pickplace) (50 episodes, 19,631 frames, SO-100 arm, Apache-2.0). They need a calibration set, so they're CLI-only.

- **Checks:** `TIME-001` timestamp grid, `TIME-002` frame-index continuity, `TIME-003` video frame count vs rows, `JOINT-001` NaN/Inf, `JOINT-002` range, `JOINT-003` velocity spike, `JOINT-004` stuck joint, `GRIP-001` flat gripper, `XSIG-001` action/state lag, `EPIS-001` length, `EPIS-002` idle fraction.
- **Verdicts:** structural violations **fail** (quarantine). Statistical ones (velocity, lag, length, idle) **warn** (human review). An episode takes its worst verdict.
- **Thresholds come from the data:** robust percentiles of a clean calibration set plus a margin. [`qgate/thresholds.py`](qgate/thresholds.py) records, for each threshold, whether it is sourced from the literature, derived from the data, or my own choice.

### Benchmark

[`bench/inject.py`](bench/inject.py) injects 13 faults at 3 severities into in-memory copies of held-out episodes (25–49) that pass clean. Thresholds are calibrated on episodes 0–24. Each cell has 10 episodes and a fixed seed, and reads `detected/attributed`. *Detected* means the copy didn't pass. *Attributed* means an expected taxonomy code was among the findings.

| Fault | Subtle | Moderate | Obvious |
|---|---|---|---|
| timestamp_jitter (ms sigma) | 10/10 (0.1) | 10/10 (0.5) | 10/10 (2.0) |
| dropped_frames | 10/10 (1) | 10/10 (5) | 10/10 (20) |
| duplicated_rows | 10/10 (1) | 10/10 (5) | 10/10 (20) |
| non_monotonic_ts (swaps) | 10/10 (1) | 10/10 (3) | 10/10 (10) |
| clock_drift (scale error) | 10/10 (1e-5) | 10/10 (1e-4) | 10/10 (1e-3) |
| velocity_spike (× p99.9) | 0/0 (1.5) | 10/10 (3.0) | 10/10 (10.0) |
| range_violation (span beyond limit) | 10/10 (0.01) | 10/10 (0.05) | 10/10 (0.3) |
| stuck_joint (frames) | 7/0 (30) | 10/10 (90) | 10/10 (200) |
| nan_values (samples) | 10/10 (1) | 10/10 (5) | 10/10 (20) |
| gripper_flat (amplitude kept) | 0/0 (0.5) | 7/7 (0.2) | 10/10 (0.0) |
| action_state_lag (extra frames) | 0/0 (2) | 10/10 (5) | 10/10 (10) |
| truncation (fraction cut) | 10/10 (0.01) | 10/10 (0.05) | 10/10 (0.25) |
| idle_padding (frames) | 1/1 (60) | 9/9 (150) | 10/10 (300) |

**False positives:** 2 of the 25 clean held-out episodes failed (95% CI 2–25%): episodes 31 and 36, both on `JOINT-002` range.

The misses are by design of the thresholds. At *subtle* severity, a 1.5× velocity spike, a gripper at half amplitude and a 2-frame lag all slip through. A 30-frame frozen joint is flagged under the wrong code: the stuck-joint window is 60 frames, and the velocity check catches the jump when the joint unfreezes.

## Limitations

- **Few thresholds have a primary source.** Only the timestamp tolerance (LeRobot's `tolerance_s`, 1e-4 s) and the episode rules borrowed in spirit from UMI are sourced. The rest are derived from 25 episodes, so they are noisy, and the benchmark's confidence intervals are wide.
- **Timing faults look easier than they are.** This dataset's timestamps sit exactly on the 1/fps grid, so even 0.1 ms of jitter is caught. Real collection hardware won't be this clean.
- **Natural outliers.** Episodes 31 and 36 trip the range check, and episode 19 trips the stuck-joint check, on unmodified data.
- **Injected faults are synthetic.** They are injected into arrays in memory, not into the files or the video. Video-level faults aren't tested.

## Not built yet

- Versioned schema design and migration across collection modalities.
- Tracing quality issues back to operators and hardware (calibration, sensor drift).
- Camera-pose and tracking-loss checks for UMI-style data; frozen or black video frames.
- Ingestion at warehouse scale.
