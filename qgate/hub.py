"""Read any LeRobot dataset on the Hugging Face Hub without downloading it.

Everything goes through an fsspec filesystem: `HfFileSystem()` with base "datasets" in the app, or a
LocalFileSystem in tests. Reads land in memory; nothing is written to disk.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from .dataset import Episode

SUPPORTED = ("v2.0", "v2.1", "v3.0")
REQUIRED = ("observation.state", "action", "frame_index", "timestamp")


@dataclass
class Inspection:
    ok: bool
    step: str = ""          # which validation step failed ("" when ok)
    message: str = ""
    info: dict = field(default_factory=dict)
    episodes: dict = field(default_factory=dict)     # episode index -> length (frames)
    meta: object = None                              # v3: per-episode metadata DataFrame (data file, video offsets)
    repo: str = ""

    @property
    def version(self):
        return self.info.get("codebase_version")

    @property
    def fps(self):
        return self.info.get("fps")

    @property
    def robot(self):
        return self.info.get("robot_type")

    @property
    def state_names(self):
        return _names(self.info["features"]["observation.state"])

    @property
    def action_names(self):
        return _names(self.info["features"]["action"])

    @property
    def cameras(self):
        return [k for k, v in self.info.get("features", {}).items() if v.get("dtype") == "video"]


def _names(feature):
    names = feature.get("names")
    if isinstance(names, dict):              # older datasets: {"motors": [...]}
        names = next(iter(names.values()), None)
    n = (feature.get("shape") or [0])[0]
    return list(names) if names else [f"ch{i}" for i in range(n)]


def inspect_dataset(repo: str, fs, base: str = "datasets") -> Inspection:
    root = f"{base}/{repo}"
    if not fs.exists(root):
        return Inspection(False, "not found", f"No public dataset called {repo} on the Hub.")
    if not fs.exists(f"{root}/meta/info.json"):
        return Inspection(False, "not LeRobot", "This dataset has no meta/info.json, so it is not in LeRobot format.")
    try:
        info = json.loads(fs.cat_file(f"{root}/meta/info.json"))
    except Exception as ex:   # HfFileSystem raises GatedRepoError / 401s; anything refusing access is the same answer here
        if isinstance(ex, PermissionError) or "Gated" in type(ex).__name__ or "401" in str(ex) or "403" in str(ex):
            return Inspection(False, "gated or private", f"{repo} needs a login or accepted terms; that is not supported yet.")
        raise
    version = str(info.get("codebase_version", "?"))
    if version not in SUPPORTED:
        return Inspection(False, "unsupported version", f"LeRobot {version}; only {', '.join(SUPPORTED)} are supported.", info)
    missing = [c for c in REQUIRED if c not in info.get("features", {})]
    if missing:
        return Inspection(False, "missing columns", f"LeRobot {version}, but no {', '.join(missing)} to check.", info)
    meta = _episode_meta(fs, root)
    episodes = {int(i): int(n) for i, n in zip(meta["episode_index"], meta["length"])}
    return Inspection(True, info=info, episodes=episodes, meta=meta.set_index("episode_index", drop=False), repo=repo)


MAX_BYTES = 50_000_000
COLUMNS = ["episode_index", "frame_index", "timestamp", "observation.state", "action"]


def data_file(insp: Inspection, idx: int) -> str:
    """Path of the parquet file holding episode `idx`, relative to the repo root."""
    info = insp.info
    if insp.version == "v3.0":
        m = insp.meta.loc[idx]
        return info["data_path"].format(chunk_index=int(m["data/chunk_index"]), file_index=int(m["data/file_index"]))
    return info["data_path"].format(episode_chunk=idx // int(info.get("chunks_size", 1000)), episode_index=idx)


RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/"


def video_source(insp: Inspection, idx: int, cam: str) -> tuple[str, float, float]:
    """(url, start s, end s) of episode `idx` for camera `cam`. v3 packs episodes into shared files at an offset."""
    info = insp.info
    if insp.version == "v3.0":
        m = insp.meta.loc[idx]
        rel = info["video_path"].format(video_key=cam, chunk_index=int(m[f"videos/{cam}/chunk_index"]),
                                        file_index=int(m[f"videos/{cam}/file_index"]))
        start, end = float(m[f"videos/{cam}/from_timestamp"]), float(m[f"videos/{cam}/to_timestamp"])
    else:
        rel = info["video_path"].format(video_key=cam, episode_index=idx, episode_chunk=idx // int(info.get("chunks_size", 1000)))
        start, end = 0.0, insp.episodes[idx] / float(insp.fps)
    return RESOLVE.format(repo=insp.repo) + rel, start, end


class TooLarge(Exception):
    pass


def load_episode(insp: Inspection, idx: int, fs, base: str = "datasets", max_bytes: int = MAX_BYTES) -> Episode:
    """Read one episode's frames into memory. Only the needed columns are fetched (byte-range reads on the Hub)."""
    import numpy as np
    import pyarrow.parquet as pq
    path = f"{base}/{insp.repo}/{data_file(insp, idx)}"
    size = fs.size(path)
    if size > max_bytes:
        raise TooLarge(f"Episode {idx} sits in a {size / 1e6:.0f} MB data file; the limit is {max_bytes / 1e6:.0f} MB.")
    with fs.open(path, "rb") as fh:
        # rows stay in stored order: sorting by frame_index would hide exactly the faults TIME-002 looks for
        t = pq.read_table(fh, columns=COLUMNS, filters=[("episode_index", "=", idx)]).to_pandas()
    return Episode(idx=idx, fps=float(insp.fps),
                   state=np.stack(t["observation.state"].to_numpy()).astype(float),
                   action=np.stack(t["action"].to_numpy()).astype(float),
                   ts=t["timestamp"].to_numpy(float), frame_index=t["frame_index"].to_numpy(np.int64))


def _episode_meta(fs, root):
    """Per-episode metadata: v3 keeps it in meta/episodes/**/*.parquet, v2.x in meta/episodes.jsonl."""
    import io
    import pandas as pd
    import pyarrow.parquet as pq
    files = sorted(fs.glob(f"{root}/meta/episodes/**/*.parquet"))
    if files:
        parts = []
        for f in files:   # only the columns used here: v3 episode metadata also carries per-episode stats, often MBs
            with fs.open(f, "rb") as fh:
                pf = pq.ParquetFile(fh)
                cols = [c for c in pf.schema_arrow.names
                        if c in ("episode_index", "length") or c.startswith("data/") or (c.startswith("videos/") and c.count("/") == 2)]
                parts.append(pf.read(columns=cols).to_pandas())
        return pd.concat(parts, ignore_index=True)
    lines = fs.cat_file(f"{root}/meta/episodes.jsonl").decode().splitlines()
    return pd.DataFrame([json.loads(x) for x in lines if x.strip()])
