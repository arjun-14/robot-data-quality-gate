"""Load a LeRobot v3 dataset (metadata + parquet; video is only checked via metadata)."""
from __future__ import annotations
import copy, glob, json, os
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class Episode:
    idx: int
    fps: float
    state: np.ndarray        # (n, d) observation.state
    action: np.ndarray       # (n, d) action
    ts: np.ndarray           # (n,) timestamp, seconds
    frame_index: np.ndarray  # (n,)
    video_frames: dict = field(default_factory=dict)  # camera key -> frame count implied by metadata

    @property
    def n(self) -> int:
        return len(self.ts)

    def copy(self) -> "Episode":
        return copy.deepcopy(self)


def load_dataset(root: str):
    """Return (info dict, joint names, list[Episode])."""
    info = json.load(open(os.path.join(root, "meta", "info.json")))
    fps = float(info["fps"])
    names = info["features"]["observation.state"]["names"]
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root}/data (run scripts/fetch_data.py)")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    meta_files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"), recursive=True))
    meta = pd.concat([pd.read_parquet(f) for f in meta_files]).set_index("episode_index")
    cams = sorted({c.split("/")[1] for c in meta.columns if c.startswith("videos/") and c.endswith("/from_timestamp")})
    episodes = []
    for idx, g in df.groupby("episode_index"):
        vf = {}
        for cam in cams:
            a = meta.loc[idx, f"videos/{cam}/from_timestamp"]
            b = meta.loc[idx, f"videos/{cam}/to_timestamp"]
            vf[cam] = int(round((b - a) * fps))
        episodes.append(Episode(
            idx=int(idx), fps=fps,
            state=np.stack(g["observation.state"].values).astype(float),
            action=np.stack(g["action"].values).astype(float),
            ts=g["timestamp"].values.astype(float),
            frame_index=g["frame_index"].values.astype(np.int64),
            video_frames=vf))
    return info, names, episodes
