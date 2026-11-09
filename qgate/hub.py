"""Read any LeRobot dataset on the Hugging Face Hub without downloading it.

Everything goes through an fsspec filesystem: `HfFileSystem()` with base "datasets" in the app, or a
LocalFileSystem in tests. Reads land in memory; nothing is written to disk.
"""
from __future__ import annotations
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from .checks import dataset_conventions, relax
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


MAX_BYTES = 200_000_000
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


def _projected_bytes(pf, cols, groups=None) -> int:
    """Compressed bytes of `cols` in the given row groups (default: all of them)."""
    md = pf.metadata
    return sum(md.row_group(g).column(j).total_compressed_size for g in (range(md.num_row_groups) if groups is None else groups)
               for j in range(md.num_columns) if md.row_group(g).column(j).path_in_schema.split(".")[0] in cols)


def row_groups_by_episode(lengths: dict, group_rows: list):
    """episode index -> ids of the row groups whose rows overlap it, for one data file.

    `lengths` is the file's episodes in stored order (episode index -> frames), `group_rows` the rows in each row group.
    None when the two disagree on the file's row count: the layout cannot be trusted, so the caller reads everything."""
    import bisect
    import itertools
    if sum(lengths.values()) != sum(group_rows):
        return None
    ends = list(itertools.accumulate(group_rows))          # row after each group's last row
    out, start = {}, 0
    for idx, n in lengths.items():
        if n:
            out[idx] = list(range(bisect.bisect_right(ends, start), bisect.bisect_left(ends, start + n) + 1))
        else:
            out[idx] = []
        start += n
    return out


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
        md = pf.metadata
        rows = [md.row_group(g).num_rows for g in range(md.num_row_groups)]
        rel = data_file(insp, idx)
        mapping = row_groups_by_episode(_file_episodes(insp, only=rel)[rel], rows)
        groups = list(range(md.num_row_groups)) if mapping is None else mapping[idx]      # layout not trusted: every group
        size = _projected_bytes(pf, cols, groups)
        if size > max_bytes:
            raise TooLarge(f"Episode {idx}: its columns sit in {size / 1e6:.0f} MB of a shared data file; the limit is {max_bytes / 1e6:.0f} MB.")
        # rows stay in stored order: sorting by frame_index would hide exactly the faults TIME-002 looks for
        t = pf.read_row_groups(groups, columns=cols).to_pandas()
        t = t[t["episode_index"] == idx]
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
    deliberate: tuple = ()                            # codes that fail in every episode checked: a convention, not a fault

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


SCAN_WORKERS = 8
MAX_SCAN_EPISODES = 200     # default sample; the page's "Scan all" button lifts it


def _file_episodes(insp: Inspection, only: str | None = None) -> dict:
    """data file (relative path) -> {episode index: frames} in stored order, for every episode of the dataset
    (or just the file `only`)."""
    out = {}
    if insp.version == "v3.0":
        m = insp.meta.sort_index()
        if only is not None:      # skip the other files' rows: the dataset can list tens of thousands of episodes
            wanted = {tuple(int(x) for x in re.findall(r"\d+", only))}
            keep = [(int(c), int(f)) in wanted for c, f in zip(m["data/chunk_index"], m["data/file_index"])]
            m = m[keep]
        for idx, chunk, file in zip(m["episode_index"], m["data/chunk_index"], m["data/file_index"]):
            rel = insp.info["data_path"].format(chunk_index=int(chunk), file_index=int(file))
            out.setdefault(rel, {})[int(idx)] = insp.episodes[int(idx)]
    else:
        for idx in sorted(insp.episodes):
            out[data_file(insp, idx)] = {idx: insp.episodes[idx]}
    return out


def _scan_file(insp: Inspection, fs, base: str, rel: str, in_file: dict, idxs: list, max_bytes: int):
    """Universal verdict for episodes `idxs` of one data file -> (EpisodeResults, {episode index: why unreadable}).

    Reads only the row groups those episodes sit in, in batches of at most `max_bytes`; an episode that does not fit
    on its own is unreadable, the rest are still checked."""
    import pyarrow.parquet as pq
    from .checks import universal_verdict
    results, unreadable = [], {}
    with fs.open(f"{base}/{insp.repo}/{rel}", "rb") as fh:
        pf = pq.ParquetFile(fh)
        sk, ak, cols = _project(insp, pf)
        md = pf.metadata
        mapping = row_groups_by_episode(in_file, [md.row_group(g).num_rows for g in range(md.num_row_groups)])
        everything = list(range(md.num_row_groups))
        groups = {i: everything if mapping is None else mapping[i] for i in idxs}     # layout not trusted: every group
        batches, current, held = [], [], set()
        for i in idxs:
            need = held | set(groups[i])
            if current and _projected_bytes(pf, cols, need) > max_bytes:
                batches.append(current); current, need = [], set(groups[i])
            current.append(i); held = need
        batches.append(current)
        for batch in batches:
            wanted = sorted({g for i in batch for g in groups[i]})
            size = _projected_bytes(pf, cols, wanted)
            if size > max_bytes:
                why = f"its columns sit in {size / 1e6:.0f} MB of a data file; the limit is {max_bytes / 1e6:.0f} MB"
                unreadable.update({i: why for i in batch})
                continue
            df = pf.read_row_groups(wanted, columns=cols).to_pandas()
            for idx in batch:
                verdict, findings = universal_verdict(_episode(insp, idx, df[df["episode_index"] == idx], sk, ak))
                results.append(EpisodeResult(idx, insp.episodes[idx], verdict, findings))
    return results, unreadable


def scan_dataset(insp: Inspection, fs, base: str = "datasets", max_episodes: int | None = None,
                 max_bytes: int = MAX_BYTES) -> Scan:
    """Run the universal checks on every episode (an evenly spaced sample past `max_episodes`), reading each data file
    once and only its state/action columns."""
    import numpy as np
    import pyarrow.parquet as pq
    from .checks import universal_verdict
    max_episodes = MAX_SCAN_EPISODES if max_episodes is None else max_episodes
    ids = sorted(insp.episodes)
    if len(ids) > max_episodes:
        ids = [ids[i] for i in np.unique(np.linspace(0, len(ids) - 1, max_episodes).round().astype(int))]
    by_file = {}
    for idx in ids:
        by_file.setdefault(data_file(insp, idx), []).append(idx)
    in_files = _file_episodes(insp)
    scan = Scan(total=len(insp.episodes))

    def one(item):
        rel, idxs = item
        try:
            return _scan_file(insp, fs, base, rel, in_files[rel], idxs, max_bytes)
        except Exception as ex:      # one bad file must not hide the verdict on the rest
            return [], {i: str(ex) or type(ex).__name__ for i in idxs}

    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as pool:      # network bound: fetch files side by side
        for results, unreadable in pool.map(one, by_file.items()):
            scan.episodes.extend(results)
            scan.unreadable.update(unreadable)
    scan.episodes.sort(key=lambda e: e.idx)
    scan.deliberate = dataset_conventions([e.findings for e in scan.episodes])
    for e in scan.episodes:
        e.verdict, e.findings = relax(e.findings, scan.deliberate)
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
