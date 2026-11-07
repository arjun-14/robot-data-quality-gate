"""Precompute everything the hosted dashboard needs (so it never downloads the dataset or video).

Writes app/data/{report.json, signals.parquet, meta.json}. Run after fetch_data.py.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qgate.dataset import load_dataset
from qgate.checks import run_checks
from qgate.thresholds import calibrate

root = sys.argv[1] if len(sys.argv) > 1 else "data/so100"
out = "app/data"; os.makedirs(out, exist_ok=True)
info, names, eps = load_dataset(root)
th = calibrate([e for e in eps if e.idx < 25])
rows, frames = [], []
for e in eps:
    v, fs = run_checks(e, th)
    rows.append(dict(episode=e.idx, verdict=v, n=e.n, checks=[f.to_dict() for f in fs]))
    d = pd.DataFrame(e.state.astype("float32"), columns=[f"s{j}" for j in range(e.state.shape[1])])
    for j in range(e.action.shape[1]): d[f"a{j}"] = e.action[:, j].astype("float32")
    d.insert(0, "timestamp", e.ts.astype("float64")); d.insert(0, "frame_index", e.frame_index); d.insert(0, "episode", e.idx)
    frames.append(d)
json.dump(dict(thresholds=th, episodes=rows), open(f"{out}/report.json", "w"))
pd.concat(frames).to_parquet(f"{out}/signals.parquet", index=False)
# Where each episode's clip sits in the Hub's concatenated mp4s, so the Live page can stream it (no video is downloaded here).
HUB = "https://huggingface.co/datasets/lerobot/svla_so100_pickplace/resolve/main/"
em = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/**/*.parquet", recursive=True))]).set_index("episode_index")
cams = sorted({c.split("/")[1] for c in em.columns if c.startswith("videos/") and c.endswith("/from_timestamp")})
videos = {str(e.idx): {cam.split(".")[-1]: dict(
    url=HUB + info["video_path"].format(video_key=cam, chunk_index=int(em.loc[e.idx, f"videos/{cam}/chunk_index"]),
                                        file_index=int(em.loc[e.idx, f"videos/{cam}/file_index"])),
    start=float(em.loc[e.idx, f"videos/{cam}/from_timestamp"]), end=float(em.loc[e.idx, f"videos/{cam}/to_timestamp"])) for cam in cams}
    for e in eps}
json.dump(dict(dataset=info.get("robot_type", "") + " / lerobot/svla_so100_pickplace", fps=info["fps"], joints=[n.replace("main_", "") for n in names],
               video_frames={str(e.idx): e.video_frames for e in eps}, videos=videos), open(f"{out}/meta.json", "w"))
print("wrote", out, {f: os.path.getsize(f"{out}/{f}") // 1024 for f in os.listdir(out)}, "KB")
