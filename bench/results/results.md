# Benchmark results

Calibrated on episodes 0-24; faults injected into copies of held-out episodes 25-49; 10 episodes per cell, drawn from the held-out episodes that pass clean; seed 0.

Cells show `detected/attributed` out of N: *detected* = the injected copy was not a pass; *attributed* = one of the expected taxonomy codes was among them.

| Fault | Subtle | Moderate | Obvious |
|---|---|---|---|
| timestamp_jitter | 10/10 of 10 (`0.1` ms sigma) | 10/10 of 10 (`0.5` ms sigma) | 10/10 of 10 (`2.0` ms sigma) |
| dropped_frames | 10/10 of 10 (`1` frames) | 10/10 of 10 (`5` frames) | 10/10 of 10 (`20` frames) |
| duplicated_rows | 10/10 of 10 (`1` rows) | 10/10 of 10 (`5` rows) | 10/10 of 10 (`20` rows) |
| non_monotonic_ts | 10/10 of 10 (`1` swaps) | 10/10 of 10 (`3` swaps) | 10/10 of 10 (`10` swaps) |
| clock_drift | 10/10 of 10 (`1e-05` scale error) | 10/10 of 10 (`0.0001` scale error) | 10/10 of 10 (`0.001` scale error) |
| velocity_spike | 0/0 of 10 (`1.5` x p99.9) | 10/10 of 10 (`3.0` x p99.9) | 10/10 of 10 (`10.0` x p99.9) |
| range_violation | 10/10 of 10 (`0.01` span beyond limit) | 10/10 of 10 (`0.05` span beyond limit) | 10/10 of 10 (`0.3` span beyond limit) |
| stuck_joint | 7/0 of 10 (`30` frames) | 10/10 of 10 (`90` frames) | 10/10 of 10 (`200` frames) |
| nan_values | 10/10 of 10 (`1` samples) | 10/10 of 10 (`5` samples) | 10/10 of 10 (`20` samples) |
| gripper_flat | 0/0 of 10 (`0.5` amplitude kept) | 7/7 of 10 (`0.2` amplitude kept) | 10/10 of 10 (`0.0` amplitude kept) |
| action_state_lag | 0/0 of 10 (`2` extra frames) | 10/10 of 10 (`5` extra frames) | 10/10 of 10 (`10` extra frames) |
| truncation | 10/10 of 10 (`0.01` fraction cut) | 10/10 of 10 (`0.05` fraction cut) | 10/10 of 10 (`0.25` fraction cut) |
| idle_padding | 1/1 of 10 (`60` frames) | 9/9 of 10 (`150` frames) | 10/10 of 10 (`300` frames) |

## Clean held-out episodes (false positives)

- Flagged (warn or fail): **2/25** (95% CI 2%-25%)
- Failed (quarantined): **2/25** (95% CI 2%-25%)
  - episode 31: fail (JOINT-002)
  - episode 36: fail (JOINT-002)
