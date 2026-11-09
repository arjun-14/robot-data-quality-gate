"""Hub dataset inspection, episode loading and video lookup, tested against tiny LeRobot layouts on local disk.

The Hub is reached through an fsspec filesystem (HfFileSystem in the app); tests use LocalFileSystem
rooted at a temp dir, so the same code paths run without the network.
"""
import json
import pytest
from fsspec.implementations.local import LocalFileSystem
from qgate.hub import TooLarge, inspect_dataset, load_episode, video_source


@pytest.fixture
def hub(tmp_path):
    return LocalFileSystem(), str(tmp_path)


def make_repo(base, repo, info=None, files=()):
    """A repo dir with an optional meta/info.json and any extra (empty) files."""
    root = f"{base}/{repo}"
    import os
    os.makedirs(root, exist_ok=True)
    if info is not None:
        os.makedirs(f"{root}/meta", exist_ok=True)
        json.dump(info, open(f"{root}/meta/info.json", "w"))
    for f in files:
        os.makedirs(os.path.dirname(f"{root}/{f}"), exist_ok=True)
        open(f"{root}/{f}", "w").close()
    return repo


def test_repo_without_lerobot_metadata_is_rejected(hub):
    fs, base = hub
    repo = make_repo(base, "someone/images", files=["README.md", "train/0001.png"])
    r = inspect_dataset(repo, fs, base)
    assert not r.ok and r.step == "not LeRobot"


FEATURES = {
    "observation.state": {"dtype": "float32", "shape": [3], "names": ["a", "b", "gripper"]},
    "action": {"dtype": "float32", "shape": [3], "names": ["a", "b", "gripper"]},
    "timestamp": {"dtype": "float32", "shape": [1]},
    "frame_index": {"dtype": "int64", "shape": [1]},
    "episode_index": {"dtype": "int64", "shape": [1]},
}


def v3_info(**over):
    return {"codebase_version": "v3.0", "robot_type": "so100", "fps": 30, "total_episodes": 2, "total_frames": 50,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": dict(FEATURES), **over}


def test_old_lerobot_version_is_rejected(hub):
    fs, base = hub
    repo = make_repo(base, "old/v1", info=v3_info(codebase_version="v1.6"))
    r = inspect_dataset(repo, fs, base)
    assert not r.ok and r.step == "unsupported version" and "v1.6" in r.message


@pytest.mark.parametrize("missing", ["observation.state", "action"])
def test_lerobot_dataset_without_required_columns_is_rejected(hub, missing):
    fs, base = hub
    feats = {k: v for k, v in FEATURES.items() if k != missing}
    repo = make_repo(base, f"x/no-{missing}", info=v3_info(features=feats))
    r = inspect_dataset(repo, fs, base)
    assert not r.ok and r.step == "missing columns" and missing in r.message


class GatedFS(LocalFileSystem):
    """The Hub lists a gated repo but refuses to serve its files without accepted terms."""
    def cat_file(self, path, *a, **k):
        raise PermissionError("gated")


def test_gated_repo_is_rejected_as_gated(hub):
    _, base = hub
    repo = make_repo(base, "corp/gated", info=v3_info())
    r = inspect_dataset(repo, GatedFS(), base)
    assert not r.ok and r.step == "gated or private"


def frames(ep, n, fps=30, offset=0, kind="move"):
    """kind: "move" (clean), "static" (nothing moves) or "idle" (robot holds still for the first 4 s)."""
    import numpy as np
    t = np.arange(n)
    st = np.stack([np.sin(t / 10 + ep), np.cos(t / 10), (t > n / 2) * 30.0], axis=1).astype("float32")
    if kind == "static":
        st[:] = st[0]
    elif kind == "idle":
        st[:120] = st[0]
    action = st if kind == "copy" else st + 1          # "copy": the exporter wrote the command into the state column
    return dict(episode_index=[ep] * n, frame_index=t.tolist(), timestamp=(t / fps).astype("float32").tolist(),
                index=(t + offset).tolist(), **{"observation.state": st.tolist(), "action": action.tolist()})


def write_v3(base, repo, lengths=(20, 30), cam="observation.images.top", drop=(), rename=None, kinds=None, groups=False):
    kinds = kinds or {}
    rename = rename or {}
    """Two episodes packed into one data file, as LeRobot v3.0 stores them."""
    import pandas as pd
    info = v3_info(total_episodes=len(lengths), total_frames=sum(lengths),
                   features={**{rename.get(k, k): v for k, v in FEATURES.items() if k not in drop}, cam: {"dtype": "video", "shape": [480, 640, 3], "info": {"video.codec": "av1"}}})
    make_repo(base, repo, info=info)
    root = f"{base}/{repo}"
    rows, eps, off = [], [], 0
    for i, n in enumerate(lengths):
        rows.append(pd.DataFrame(frames(i, n, offset=off, kind=kinds.get(i, "move"))))
        eps.append({"episode_index": i, "length": n, "data/chunk_index": 0, "data/file_index": 0,
                    f"videos/{cam}/chunk_index": 0, f"videos/{cam}/file_index": 0,
                    f"videos/{cam}/from_timestamp": off / 30, f"videos/{cam}/to_timestamp": (off + n) / 30})
        off += n
    import os
    os.makedirs(f"{root}/data/chunk-000", exist_ok=True); os.makedirs(f"{root}/meta/episodes/chunk-000", exist_ok=True)
    data_path = f"{root}/data/chunk-000/file-000.parquet"
    if groups:    # one row group per episode, as FineART is written
        import pyarrow as pa, pyarrow.parquet as pq
        tables = [pa.Table.from_pandas(r.drop(columns=list(drop)).rename(columns=rename), preserve_index=False) for r in rows]
        with pq.ParquetWriter(data_path, tables[0].schema) as w:
            for t in tables:
                w.write_table(t, row_group_size=len(t) or 1)
    else:
        pd.concat(rows or [pd.DataFrame(frames(0, 0))]).drop(columns=list(drop)).rename(columns=rename).to_parquet(data_path, index=False)
    pd.DataFrame(eps).to_parquet(f"{root}/meta/episodes/chunk-000/file-000.parquet", index=False)
    return repo


def test_valid_v3_dataset_is_summarised_with_its_episodes(hub):
    fs, base = hub
    repo = write_v3(base, "lab/arm-v3", lengths=(20, 30))
    r = inspect_dataset(repo, fs, base)
    assert r.ok
    assert r.version == "v3.0" and r.fps == 30 and r.robot == "so100"
    assert r.episodes == {0: 20, 1: 30}
    assert r.state_names == ["a", "b", "gripper"] and r.cameras == ["observation.images.top"]


def write_v2(base, repo, lengths=(20, 30), version="v2.1", cam="observation.images.top", motors_dict=False):
    """One parquet and one video per episode, as LeRobot v2.x stores them."""
    import os, pandas as pd
    feats = {**FEATURES, cam: {"dtype": "video", "shape": [480, 640, 3], "info": {"video.codec": "h264"}}}
    if motors_dict:
        for k in ("observation.state", "action"):
            feats[k] = {**feats[k], "names": {"motors": ["a", "b", "gripper"]}}
    info = v3_info(codebase_version=version, total_episodes=len(lengths), chunks_size=1000, features=feats,
                   data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                   video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    make_repo(base, repo, info=info)
    root = f"{base}/{repo}"
    os.makedirs(f"{root}/data/chunk-000", exist_ok=True)
    with open(f"{root}/meta/episodes.jsonl", "w") as f:
        for i, n in enumerate(lengths):
            pd.DataFrame(frames(i, n)).to_parquet(f"{root}/data/chunk-000/episode_{i:06d}.parquet", index=False)
            f.write(json.dumps({"episode_index": i, "tasks": ["pick"], "length": n}) + "\n")
    return repo


def test_valid_v2_dataset_is_summarised_with_its_episodes(hub):
    fs, base = hub
    repo = write_v2(base, "lab/arm-v2", lengths=(12, 40, 25), motors_dict=True)
    r = inspect_dataset(repo, fs, base)
    assert r.ok and r.version == "v2.1"
    assert r.episodes == {0: 12, 1: 40, 2: 25}
    assert r.state_names == ["a", "b", "gripper"]


def test_v3_episode_is_cut_out_of_the_shared_data_file(hub):
    import numpy as np
    fs, base = hub
    repo = write_v3(base, "lab/arm-v3", lengths=(20, 30))
    ep = load_episode(inspect_dataset(repo, fs, base), 1, fs, base)
    want = frames(1, 30)
    assert ep.idx == 1 and ep.n == 30 and ep.fps == 30
    assert ep.frame_index.tolist() == list(range(30))
    np.testing.assert_allclose(ep.state, np.array(want["observation.state"]), rtol=1e-6)
    np.testing.assert_allclose(ep.action, np.array(want["action"]), rtol=1e-6)


def test_v2_episode_is_read_from_its_own_file(hub):
    import numpy as np
    fs, base = hub
    repo = write_v2(base, "lab/arm-v2", lengths=(12, 40, 25), version="v2.0")
    ep = load_episode(inspect_dataset(repo, fs, base), 2, fs, base)
    assert ep.idx == 2 and ep.n == 25
    np.testing.assert_allclose(ep.state, np.array(frames(2, 25)["observation.state"]), rtol=1e-6)


def test_episode_in_an_oversized_data_file_is_refused(hub):
    fs, base = hub
    repo = write_v3(base, "lab/huge", lengths=(20, 30))
    insp = inspect_dataset(repo, fs, base)
    with pytest.raises(TooLarge, match="MB"):
        load_episode(insp, 0, fs, base, max_bytes=100)


def test_v3_video_is_a_window_into_the_shared_file(hub):
    fs, base = hub
    repo = write_v3(base, "lab/arm-v3", lengths=(20, 30))
    url, start, end = video_source(inspect_dataset(repo, fs, base), 1, "observation.images.top")
    assert url == "https://huggingface.co/datasets/lab/arm-v3/resolve/main/videos/observation.images.top/chunk-000/file-000.mp4"
    assert start == pytest.approx(20 / 30) and end == pytest.approx(50 / 30)


def test_v2_video_is_the_episode_file(hub):
    fs, base = hub
    repo = write_v2(base, "lab/arm-v2", lengths=(12, 45))
    url, start, end = video_source(inspect_dataset(repo, fs, base), 1, "observation.images.top")
    assert url == "https://huggingface.co/datasets/lab/arm-v2/resolve/main/videos/chunk-000/observation.images.top/episode_000001.mp4"
    assert start == 0 and end == pytest.approx(1.5)


def test_unknown_repo_is_rejected_as_not_found(hub):
    fs, base = hub
    r = inspect_dataset("nobody/nothing", fs, base)
    assert not r.ok and r.step == "not found"


def test_dataset_with_no_episodes_is_rejected_not_crashed(hub):
    fs, base = hub
    repo = write_v2(base, "lab/empty-v2", lengths=())
    r = inspect_dataset(repo, fs, base)
    assert not r.ok and r.step == "no episodes"
    repo = write_v3(base, "lab/empty-v3", lengths=())
    r = inspect_dataset(repo, fs, base)
    assert not r.ok and r.step == "no episodes"


def test_dataset_without_frame_index_or_timestamp_is_still_checkable(hub):
    import numpy as np
    fs, base = hub
    repo = write_v3(base, "lab/bare", lengths=(20, 30), drop=("frame_index", "timestamp"))
    insp = inspect_dataset(repo, fs, base)
    assert insp.ok
    ep = load_episode(insp, 1, fs, base)
    assert ep.n == 30
    assert ep.frame_index.tolist() == list(range(30))
    np.testing.assert_allclose(ep.ts, np.arange(30) / 30)


def test_non_standard_state_and_action_names_are_understood(hub):
    import numpy as np
    fs, base = hub
    repo = write_v3(base, "lab/aliased", lengths=(20, 30), rename={"observation.state": "state", "action": "actions"})
    insp = inspect_dataset(repo, fs, base)
    assert insp.ok
    assert insp.state_names == ["a", "b", "gripper"] and insp.action_names == ["a", "b", "gripper"]
    ep = load_episode(insp, 1, fs, base)
    np.testing.assert_allclose(ep.state, np.array(frames(1, 30)["observation.state"]), rtol=1e-6)
    np.testing.assert_allclose(ep.action, np.array(frames(1, 30)["action"]), rtol=1e-6)


def test_size_cap_counts_only_the_columns_read(hub):
    import numpy as np, pandas as pd, os
    fs, base = hub
    repo = write_v3(base, "lab/wide", lengths=(20, 30))
    path = f"{base}/{repo}/data/chunk-000/file-000.parquet"
    df = pd.read_parquet(path)
    rng = np.random.default_rng(0)
    df["junk"] = [rng.random(2000).tolist() for _ in range(len(df))]    # incompressible bulk the gate never reads
    df.to_parquet(path, index=False)
    insp = inspect_dataset(repo, fs, base)
    assert os.path.getsize(path) > 500_000
    assert load_episode(insp, 0, fs, base, max_bytes=50_000).n == 20
    with pytest.raises(TooLarge):
        load_episode(insp, 0, fs, base, max_bytes=100)


def test_scan_counts_passed_warned_and_failed_episodes(hub):
    from qgate.hub import scan_dataset
    fs, base = hub
    repo = write_v3(base, "lab/mixed", lengths=(300, 300, 300, 300), kinds={1: "static", 2: "idle"})
    scan = scan_dataset(inspect_dataset(repo, fs, base), fs, base)
    assert scan.total == 4 and scan.scanned == 4
    assert (scan.passed, scan.warned, scan.failed) == (2, 1, 1)
    assert scan.verdict == "fail"
    assert scan.by_code["JOINT-005"] == [1] and scan.by_code["EPIS-004"] == [2]
    assert {e.idx: e.verdict for e in scan.episodes} == {0: "pass", 1: "fail", 2: "warn", 3: "pass"}


def test_scan_of_a_large_dataset_samples_evenly_and_says_so(hub):
    from qgate.hub import scan_dataset
    fs, base = hub
    repo = write_v2(base, "lab/many", lengths=(300,) * 10)
    scan = scan_dataset(inspect_dataset(repo, fs, base), fs, base, max_episodes=4)
    assert scan.total == 10 and scan.scanned == 4
    assert [e.idx for e in scan.episodes] == [0, 3, 6, 9]
    assert scan.sampled
    assert not scan_dataset(inspect_dataset(repo, fs, base), fs, base).sampled


def test_scan_reports_unreadable_episodes_instead_of_dropping_them(hub):
    import os
    from qgate.hub import scan_dataset
    fs, base = hub
    repo = write_v2(base, "lab/broken", lengths=(300, 300, 300))
    os.remove(f"{base}/{repo}/data/chunk-000/episode_000001.parquet")
    scan = scan_dataset(inspect_dataset(repo, fs, base), fs, base)
    assert [e.idx for e in scan.episodes] == [0, 2]
    assert list(scan.unreadable) == [1] and scan.unreadable[1]
    assert scan.total == 3


def test_scan_skips_data_files_over_the_size_cap_and_says_why(hub):
    from qgate.hub import scan_dataset
    fs, base = hub
    repo = write_v3(base, "lab/huge", lengths=(300, 300))
    scan = scan_dataset(inspect_dataset(repo, fs, base), fs, base, max_bytes=100)
    assert scan.scanned == 0 and sorted(scan.unreadable) == [0, 1]
    assert "MB" in scan.unreadable[0]


# ---- which row groups of a shared data file hold which episodes

def test_each_episode_maps_to_the_row_groups_that_overlap_its_rows():
    from qgate.hub import row_groups_by_episode
    # row groups line up with episodes (as in FineART): one group each
    assert row_groups_by_episode({0: 3, 1: 2, 2: 4}, [3, 2, 4]) == {0: [0], 1: [1], 2: [2]}
    # row groups do not line up: episode 1 (rows 3-4) straddles groups 0 (rows 0-3) and 1 (rows 4-8)
    assert row_groups_by_episode({0: 3, 1: 2, 2: 4}, [4, 5]) == {0: [0], 1: [0, 1], 2: [1]}


def test_episode_lengths_that_disagree_with_the_file_map_to_nothing():
    from qgate.hub import row_groups_by_episode
    assert row_groups_by_episode({0: 3, 1: 2}, [3, 3]) is None


# ---- scanning a dataset whose episodes share large data files

def test_scan_checks_episodes_whose_row_groups_fit_the_cap_though_the_file_does_not(hub):
    import os
    from qgate.hub import scan_dataset
    fs, base = hub
    repo = write_v3(base, "lab/big-file", lengths=(300,) * 6, groups=True)
    insp = inspect_dataset(repo, fs, base)
    cap = os.path.getsize(f"{base}/{repo}/data/chunk-000/file-000.parquet") // 3     # under the file, over any one episode
    scan = scan_dataset(insp, fs, base, max_episodes=2, max_bytes=cap)
    assert scan.unreadable == {}
    assert sorted(e.idx for e in scan.episodes) == [0, 5]


def test_scan_of_every_episode_in_a_file_over_the_cap_is_read_in_batches(hub):
    import os
    from qgate.hub import scan_dataset
    fs, base = hub
    repo = write_v3(base, "lab/big-file", lengths=(300,) * 6, groups=True)
    insp = inspect_dataset(repo, fs, base)
    cap = os.path.getsize(f"{base}/{repo}/data/chunk-000/file-000.parquet") // 3
    scan = scan_dataset(insp, fs, base, max_bytes=cap)
    assert scan.unreadable == {} and sorted(e.idx for e in scan.episodes) == list(range(6))


def test_an_episode_too_big_for_the_cap_does_not_hide_the_others(hub):
    import os
    from qgate.hub import scan_dataset
    fs, base = hub
    repo = write_v3(base, "lab/lopsided", lengths=(100, 100, 3000), groups=True)
    insp = inspect_dataset(repo, fs, base)
    cap = os.path.getsize(f"{base}/{repo}/data/chunk-000/file-000.parquet") // 4
    scan = scan_dataset(insp, fs, base, max_bytes=cap)
    assert sorted(e.idx for e in scan.episodes) == [0, 1]
    assert list(scan.unreadable) == [2] and "MB" in scan.unreadable[2]


def test_scan_verdicts_do_not_depend_on_how_the_file_is_split_into_row_groups(hub):
    from qgate.hub import scan_dataset
    fs, base = hub
    kinds = {0: "move", 1: "static", 2: "idle", 3: "move"}
    out = {}
    for name, groups in (("one-group", False), ("per-episode", True)):
        repo = write_v3(base, f"lab/{name}", lengths=(200, 250, 300, 220), kinds=kinds, groups=groups)
        scan = scan_dataset(inspect_dataset(repo, fs, base), fs, base, max_episodes=3)
        assert scan.unreadable == {}
        out[name] = {e.idx: (e.verdict, [(f.code, f.status) for f in e.findings]) for e in scan.episodes}
    assert len(out["one-group"]) == 3 and out["one-group"] == out["per-episode"]
    assert len({v for v, _ in out["per-episode"].values()}) > 1       # the sample really contains good and bad episodes


def test_scan_falls_back_to_reading_everything_when_episode_lengths_disagree_with_the_file(hub):
    from qgate.hub import scan_dataset
    fs, base = hub
    repo = write_v3(base, "lab/liar", lengths=(200, 250, 300), groups=True)
    insp = inspect_dataset(repo, fs, base)
    insp.episodes[0] = 999                                              # metadata no longer matches the data file
    scan = scan_dataset(insp, fs, base)
    assert scan.unreadable == {} and sorted(e.idx for e in scan.episodes) == [0, 1, 2]


def test_scan_reads_data_files_concurrently(tmp_path):
    import threading
    from qgate.hub import scan_dataset

    class MeetingPoint(LocalFileSystem):
        """Each data file waits for another to be opened at the same time: a sequential scan times out."""
        gate = threading.Barrier(2, timeout=3)

        def open(self, path, *a, **k):
            if "/episode_" in str(path):
                self.gate.wait()
            return super().open(path, *a, **k)

    fs, base = MeetingPoint(), str(tmp_path)
    repo = write_v2(base, "lab/many-files", lengths=(50, 60))
    scan = scan_dataset(inspect_dataset(repo, LocalFileSystem(), base), fs, base)
    assert scan.unreadable == {} and sorted(e.idx for e in scan.episodes) == [0, 1]


def test_a_corrupt_data_file_leaves_the_other_episodes_checked(hub):
    from qgate.hub import scan_dataset
    fs, base = hub
    repo = write_v2(base, "lab/one-bad", lengths=(50, 60, 70))
    open(f"{base}/{repo}/data/chunk-000/episode_000001.parquet", "wb").write(b"not parquet")
    scan = scan_dataset(inspect_dataset(repo, fs, base), fs, base)
    assert sorted(e.idx for e in scan.episodes) == [0, 2] and list(scan.unreadable) == [1]


def test_episode_is_loaded_from_its_own_row_group_when_the_whole_file_is_over_the_cap(hub):
    import os
    import numpy as np
    fs, base = hub
    repo = write_v3(base, "lab/big-file", lengths=(300, 300, 300, 300), groups=True)
    insp = inspect_dataset(repo, fs, base)
    cap = os.path.getsize(f"{base}/{repo}/data/chunk-000/file-000.parquet") // 2
    ep = load_episode(insp, 2, fs, base, max_bytes=cap)
    assert ep.n == 300
    np.testing.assert_allclose(ep.state, np.array(frames(2, 300)["observation.state"]), rtol=1e-6)


def test_episode_whose_own_row_groups_exceed_the_cap_is_still_refused(hub):
    fs, base = hub
    repo = write_v3(base, "lab/big-file", lengths=(300, 300), groups=True)
    with pytest.raises(TooLarge, match="MB"):
        load_episode(inspect_dataset(repo, fs, base), 0, fs, base, max_bytes=100)


def test_scan_samples_up_to_the_default_limit_unless_asked_for_more(hub, monkeypatch):
    import qgate.hub
    from qgate.hub import scan_dataset
    fs, base = hub
    repo = write_v2(base, "lab/eight", lengths=(30,) * 8)
    insp = inspect_dataset(repo, fs, base)
    monkeypatch.setattr(qgate.hub, "MAX_SCAN_EPISODES", 3)
    assert scan_dataset(insp, fs, base).scanned == 3
    assert scan_dataset(insp, fs, base, max_episodes=8).scanned == 8


def test_the_default_scan_limit_is_two_hundred_episodes():
    import qgate.hub
    assert qgate.hub.MAX_SCAN_EPISODES == 200


# ---- state equal to action: a fault in one episode, a recording convention when every episode does it

def scan_of(hub, name, kinds, lengths=(200, 250, 300)):
    from qgate.hub import scan_dataset
    fs, base = hub
    repo = write_v3(base, f"lab/{name}", lengths=lengths, kinds=kinds)
    return scan_dataset(inspect_dataset(repo, fs, base), fs, base)


def test_state_equal_to_action_in_every_episode_is_a_convention_not_a_failure(hub):
    scan = scan_of(hub, "all-copy", {0: "copy", 1: "copy", 2: "copy"})
    assert scan.failed == 0 and scan.verdict == "pass" and scan.scanned == 3
    assert scan.by_code.get("XSIG-002") is None
    assert scan.deliberate == ("XSIG-002",)
    detail = next(f.detail for f in scan.episodes[0].findings if f.code == "XSIG-002")
    assert "every episode" in detail and "deliberate" in detail


def test_state_equal_to_action_in_only_some_episodes_still_fails_those(hub):
    scan = scan_of(hub, "some-copy", {0: "copy"})
    assert scan.deliberate == ()
    assert scan.by_code["XSIG-002"] == [0] and scan.failed == 1 and scan.passed == 2


def test_one_episode_alone_cannot_show_that_state_equal_to_action_is_deliberate(hub):
    scan = scan_of(hub, "one-copy", {0: "copy"}, lengths=(200,))
    assert scan.deliberate == () and scan.failed == 1
