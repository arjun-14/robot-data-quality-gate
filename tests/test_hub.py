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


@pytest.mark.parametrize("missing", ["observation.state", "action", "frame_index", "timestamp"])
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


def frames(ep, n, fps=30, offset=0):
    import numpy as np
    t = np.arange(n)
    st = np.stack([np.sin(t / 10 + ep), np.cos(t / 10), (t > n / 2) * 30.0], axis=1).astype("float32")
    return dict(episode_index=[ep] * n, frame_index=t.tolist(), timestamp=(t / fps).astype("float32").tolist(),
                index=(t + offset).tolist(), **{"observation.state": st.tolist(), "action": (st + 1).tolist()})


def write_v3(base, repo, lengths=(20, 30), cam="observation.images.top"):
    """Two episodes packed into one data file, as LeRobot v3.0 stores them."""
    import pandas as pd
    info = v3_info(total_episodes=len(lengths), total_frames=sum(lengths),
                   features={**FEATURES, cam: {"dtype": "video", "shape": [480, 640, 3], "info": {"video.codec": "av1"}}})
    make_repo(base, repo, info=info)
    root = f"{base}/{repo}"
    rows, eps, off = [], [], 0
    for i, n in enumerate(lengths):
        rows.append(pd.DataFrame(frames(i, n, offset=off)))
        eps.append({"episode_index": i, "length": n, "data/chunk_index": 0, "data/file_index": 0,
                    f"videos/{cam}/chunk_index": 0, f"videos/{cam}/file_index": 0,
                    f"videos/{cam}/from_timestamp": off / 30, f"videos/{cam}/to_timestamp": (off + n) / 30})
        off += n
    import os
    os.makedirs(f"{root}/data/chunk-000", exist_ok=True); os.makedirs(f"{root}/meta/episodes/chunk-000", exist_ok=True)
    pd.concat(rows).to_parquet(f"{root}/data/chunk-000/file-000.parquet", index=False)
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
