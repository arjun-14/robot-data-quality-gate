"""Benchmark: calibrate on episodes 0-24, inject faults into copies of held-out episodes 25-49, score the gate."""
from __future__ import annotations
import json, math, os
import numpy as np
from qgate.dataset import load_dataset
from qgate.thresholds import calibrate
from qgate.checks import run_checks, PASS
from .inject import FAULTS, SEVERITIES, inject


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def run_benchmark(dataset, out_dir, seed=0, per_cell=10, cal=(0, 25), test=(25, 50)):
    os.makedirs(out_dir, exist_ok=True)
    _, _, eps = load_dataset(dataset)
    cal_eps = [e for e in eps if cal[0] <= e.idx < cal[1]]
    test_eps = [e for e in eps if test[0] <= e.idx < test[1]]
    th = calibrate(cal_eps)

    clean = []
    for e in test_eps:
        v, fs = run_checks(e, th)
        clean.append(dict(episode=e.idx, verdict=v, codes=[f.code for f in fs if f.status != PASS]))
    flagged = [c for c in clean if c["verdict"] != PASS]
    failed = [c for c in clean if c["verdict"] == "fail"]

    clean_ids = {c['episode'] for c in clean if c['verdict'] == PASS}
    pool = [e for e in test_eps if e.idx in clean_ids]   # inject only into episodes that pass clean, so detection is unambiguous
    truth, results = [], []
    for fi, (fault, (fn, params, expected, unit)) in enumerate(FAULTS.items()):
        for si, sev in enumerate(SEVERITIES):
            rng = np.random.default_rng([seed, fi, si])
            pick = rng.choice(len(pool), per_cell, replace=False)
            for p in pick:
                src = pool[int(p)]
                e = inject(src, fault, si, th, rng)
                v, fs = run_checks(e, th)
                codes = [f.code for f in fs if f.status != PASS]
                new = codes   # pool episodes are clean, so every finding is new
                rec = dict(fault=fault, severity=sev, param=params[si], unit=unit, episode=src.idx, expected=sorted(expected))
                truth.append(rec)
                results.append(dict(rec, flagged=v != PASS, codes=codes, new_codes=new,
                                    attributed=bool(set(new) & expected)))
    json.dump(truth, open(os.path.join(out_dir, "ground_truth.json"), "w"), indent=1)

    table = {}
    for r in results:
        t = table.setdefault(r["fault"], {})
        c = t.setdefault(r["severity"], dict(n=0, flagged=0, attributed=0, param=r["param"], unit=r["unit"]))
        c["n"] += 1; c["flagged"] += r["flagged"]; c["attributed"] += r["attributed"]
    n_clean = len(clean)
    fp = dict(n=n_clean, flagged=len(flagged), flagged_ci=wilson(len(flagged), n_clean), failed=len(failed), failed_ci=wilson(len(failed), n_clean), episodes=clean)
    out = dict(seed=seed, per_cell=per_cell, calibration=list(cal), test=list(test), thresholds=th, clean_heldout=fp, table=table, results=results)
    json.dump(out, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    md = render_markdown(out)
    open(os.path.join(out_dir, "results.md"), "w").write(md)
    print(md)
    return out


def render_markdown(o):
    L = ["# Benchmark results", "",
         f"Calibrated on episodes {o['calibration'][0]}-{o['calibration'][1]-1}; faults injected into copies of held-out episodes {o['test'][0]}-{o['test'][1]-1}; "
         f"{o['per_cell']} episodes per cell, drawn from the held-out episodes that pass clean; seed {o['seed']}.", "",
         "Cells show `detected/attributed` out of N: *detected* = the injected copy was not a pass; *attributed* = one of the expected taxonomy codes was among them.", "",
         "| Fault | Subtle | Moderate | Obvious |", "|---|---|---|---|"]
    for fault, t in o["table"].items():
        cells = []
        for sev in ("subtle", "moderate", "obvious"):
            c = t[sev]; cells.append(f"{c['flagged']}/{c['attributed']} of {c['n']} (`{c['param']}` {c['unit']})")
        L.append(f"| {fault} | " + " | ".join(cells) + " |")
    f = o["clean_heldout"]
    L += ["", "## Clean held-out episodes (false positives)", "",
          f"- Flagged (warn or fail): **{f['flagged']}/{f['n']}** (95% CI {f['flagged_ci'][0]:.0%}-{f['flagged_ci'][1]:.0%})",
          f"- Failed (quarantined): **{f['failed']}/{f['n']}** (95% CI {f['failed_ci'][0]:.0%}-{f['failed_ci'][1]:.0%})"]
    for c in f["episodes"]:
        if c["verdict"] != "pass":
            L.append(f"  - episode {c['episode']}: {c['verdict']} ({', '.join(c['codes'])})")
    return "\n".join(L) + "\n"
