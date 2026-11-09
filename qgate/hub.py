"""Read any LeRobot dataset on the Hugging Face Hub without downloading it.

Everything goes through an fsspec filesystem: `HfFileSystem()` with base "datasets" in the app, or a
LocalFileSystem in tests. Reads land in memory; nothing is written to disk.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from .dataset import Episode

SUPPORTED = ("v2.0", "v2.1", "v3.0")
ALIASES = {"observation.state": ("observation.state", "state"), "action": ("action", "actions")}
REQUIRED = ("observation.state", "action")   # frame_index / timestamp are optional: no universal check needs them


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
        return _names(self.info["features"][column(self.info["features"], "observation.state")])

    @property
    def action_names(self):
        return _names(self.info["features"][column(self.info["features"], "action")])

    @property
    def cameras(self):
        return [k for k, v in self.info.get("features", {}).items() if v.get("dtype") == "video"]


def column(features, name):
    """The dataset's own key for the canonical column `name` ("observation.state", "action"), or None."""
    return next((k for k in ALIASES[name] if k in features), None)


def _names(feature):
    names = feature.get("names")
    if isinstance(names, dict):              # older datasets: {"motors": [...]}
        names = next(iter(names.values()), None)
    n = (feature.get("shape") or [0])[0]
    return list(names) if names else [f"ch{i}" for i in range(n)]


def _exists(fs, path: str) -> bool:
    """Like fs.exists, but only a real "no such path" is False; rate limits (429), outages etc. raise instead of
    masquerading as "not found" (and being cached as such)."""
    try:
        fs.info(path)
        return True
    except FileNotFoundError:
        return False


def inspect_dataset(repo: str, fs, base: str = "datasets") -> Inspection:
    root = f"{base}/{repo}"
    if not _exists(fs, root):
        return Inspection(False, "not found", f"No public dataset called {repo} on the Hub.")
    if not _exists(fs, f"{root}/meta/info.json"):
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
    missing = [c for c in REQUIRED if column(info.get("features", {}), c) is None]
    if missing:
        return Inspection(False, "missing columns", f"LeRobot {version}, but no {', '.join(missing)} to check.", info)
    meta = _episode_meta(fs, root)
    if meta.empty or "episode_index" not in meta:
        return Inspection(False, "no episodes", f"LeRobot {version}, but the dataset lists no episodes to check.", info)
    episodes = {int(i): int(n) for i, n in zip(meta["episode_index"], meta["length"])}
    return Inspection(True, info=info, episodes=episodes, meta=meta.set_index("episode_index", drop=False), repo=repo)


MAX_BYTES = 50_000_000
COLUMNS = ["episode_index", "frame_index", "timestamp", "observation.state", "action"]
OPTIONAL = ("frame_index", "timestamp")


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


def _project(insp: Inspection, pf):
    """(state key, action key, columns to read) for a parquet file: aliases resolved, optional columns only if present."""
    feats = insp.info["features"]
    sk, ak = column(feats, "observation.state"), column(feats, "action")
    have = set(pf.schema_arrow.names)
    cols = [c for c in COLUMNS if c not in OPTIONAL or c in have]
    return sk, ak, [sk if c == "observation.state" else ak if c == "action" else c for c in cols]


def _projected_bytes(pf, cols) -> int:
    md = pf.metadata
    return sum(md.row_group(g).column(j).total_compressed_size for g in range(md.num_row_groups)
               for j in range(md.num_columns) if md.row_group(g).column(j).path_in_schema.split(".")[0] in cols)


def _episode(insp: Inspection, idx: int, t, sk: str, ak: str) -> Episode:
    import numpy as np
    n, fps = len(t), float(insp.fps)
    return Episode(idx=idx, fps=fps,
                   state=np.stack(t[sk].to_numpy()).astype(float),
                   action=np.stack(t[ak].to_numpy()).astype(float),
                   ts=t["timestamp"].to_numpy(float) if "timestamp" in t else np.arange(n) / fps,
                   frame_index=t["frame_index"].to_numpy(np.int64) if "frame_index" in t else np.arange(n, dtype=np.int64))


def load_episode(insp: Inspection, idx: int, fs, base: str = "datasets", max_bytes: int = MAX_BYTES) -> Episode:
    """Read one episode's frames into memory. Only the needed columns are fetched (byte-range reads on the Hub)."""
    import pyarrow.parquet as pq
    path = f"{base}/{insp.repo}/{data_file(insp, idx)}"
    with fs.open(path, "rb") as fh:
        pf = pq.ParquetFile(fh)
        sk, ak, cols = _project(insp, pf)
        # the cap is on what is actually fetched (the columns read), not on the whole data file
        size = _projected_bytes(pf, cols)
        if size > max_bytes:
            raise TooLarge(f"Episode {idx}: its columns sit in {size / 1e6:.0f} MB of a shared data file; the limit is {max_bytes / 1e6:.0f} MB.")
        # rows stay in stored order: sorting by frame_index would hide exactly the faults TIME-002 looks for
        t = pq.read_table(fh, columns=cols, filters=[("episode_index", "=", idx)]).to_pandas()
    return _episode(insp, idx, t, sk, ak)


@dataclass
class EpisodeResult:
    idx: int
    n: int
    verdict: str
    findings: list


@dataclass
class Scan:
    total: int                                        # episodes in the dataset
    episodes: list = field(default_factory=list)      # EpisodeResult per checked episode
    unreadable: dict = field(default_factory=dict)    # episode index -> why it could not be checked

    @property
    def scanned(self):
        return len(self.episodes)

    @property
    def sampled(self):
        return self.scanned < self.total

    def _count(self, v):
        return sum(e.verdict == v for e in self.episodes)

    passed = property(lambda self: self._count("pass"))
    warned = property(lambda self: self._count("warn"))
    failed = property(lambda self: self._count("fail"))

    @property
    def verdict(self):
        return "fail" if self.failed else "warn" if self.warned else "pass"

    @property
    def by_code(self):
        """code -> episode indices where that check warned or failed."""
        out = {}
        for e in self.episodes:
            for f in e.findings:
                if f.status != "pass":
                    out.setdefault(f.code, []).append(e.idx)
        return out


MAX_SCAN_EPISODES = 200     # one file per episode on v2.x: stay well under the Hub's anonymous 500 requests per 5 minutes


def scan_dataset(insp: Inspection, fs, base: str = "datasets", max_episodes: int = MAX_SCAN_EPISODES,
                 max_bytes: int = MAX_BYTES) -> Scan:
    """Run the universal checks on every episode (an evenly spaced sample past `max_episodes`), reading each data file
    once and only its state/action columns."""
    import numpy as np
    import pyarrow.parquet as pq
    from .checks import universal_verdict
    ids = sorted(insp.episodes)
    if len(ids) > max_episodes:
        ids = [ids[i] for i in np.unique(np.linspace(0, len(ids) - 1, max_episodes).round().astype(int))]
    by_file = {}
    for idx in ids:
        by_file.setdefault(data_file(insp, idx), []).append(idx)
    scan = Scan(total=len(insp.episodes))
    for rel, idxs in by_file.items():
        try:
            with fs.open(f"{base}/{insp.repo}/{rel}", "rb") as fh:
                pf = pq.ParquetFile(fh)
                sk, ak, cols = _project(insp, pf)
                size = _projected_bytes(pf, cols)
                if size > max_bytes:
                    raise TooLarge(f"its columns sit in {size / 1e6:.0f} MB of a data file; the limit is {max_bytes / 1e6:.0f} MB")
                df = pf.read(columns=cols).to_pandas()
        except Exception as ex:      # one bad file must not hide the verdict on the rest
            scan.unreadable.update({i: str(ex) or type(ex).__name__ for i in idxs})
            continue
        for idx in idxs:
            verdict, findings = universal_verdict(_episode(insp, idx, df[df["episode_index"] == idx], sk, ak))
            scan.episodes.append(EpisodeResult(idx, insp.episodes[idx], verdict, findings))
    return scan


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
