"""qgate command line: calibrate, check, bench."""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter
from .dataset import load_dataset
from .checks import run_checks, RANK
from .thresholds import calibrate


def _split(s, n):
    a, b = s.split(":"); return int(a), int(b)


def cmd_calibrate(a):
    _, _, eps = load_dataset(a.dataset)
    lo, hi = _split(a.on, len(eps))
    th = calibrate([e for e in eps if lo <= e.idx < hi])
    json.dump(th, open(a.out, "w"), indent=2); print(f"wrote {a.out} (calibrated on episodes {lo}-{hi - 1})")


def cmd_check(a):
    _, _, eps = load_dataset(a.dataset)
    if a.thresholds:
        th = json.load(open(a.thresholds))
    else:
        lo, hi = _split(a.calibrate_on, len(eps))
        th = calibrate([e for e in eps if lo <= e.idx < hi])
    rows = []
    for e in eps:
        verdict, fs = run_checks(e, th)
        rows.append(dict(episode=e.idx, verdict=verdict, findings=[f.to_dict() for f in fs if f.status != "pass"]))
    cnt = Counter(r["verdict"] for r in rows)
    report = dict(dataset=a.dataset, n_episodes=len(rows), summary=dict(cnt), thresholds=th, episodes=rows)
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
    print(f"{len(rows)} episodes: {cnt.get('pass', 0)} pass, {cnt.get('warn', 0)} warn, {cnt.get('fail', 0)} fail")
    for r in rows:
        if r["verdict"] != "pass":
            print(f"  ep {r['episode']:>3} {r['verdict'].upper():<4} " + "; ".join(f"{f['code']} {f['detail']}" for f in r["findings"]))
    return 1 if cnt.get("fail") else 0


def cmd_bench(a):
    from bench.run import run_benchmark
    run_benchmark(a.dataset, a.out_dir, seed=a.seed, per_cell=a.per_cell)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="qgate", description="Quality gate for robot manipulation datasets")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate"); c.add_argument("dataset"); c.add_argument("--on", default="0:25"); c.add_argument("--out", default="thresholds.json"); c.set_defaults(fn=cmd_calibrate)
    k = sub.add_parser("check", help="exit code 1 if any episode fails"); k.add_argument("dataset"); k.add_argument("--thresholds"); k.add_argument("--calibrate-on", default="0:25"); k.add_argument("--out"); k.set_defaults(fn=cmd_check)
    b = sub.add_parser("bench"); b.add_argument("dataset"); b.add_argument("--out-dir", default="bench/results"); b.add_argument("--seed", type=int, default=0); b.add_argument("--per-cell", type=int, default=10); b.set_defaults(fn=cmd_bench)
    a = p.parse_args(argv)
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
