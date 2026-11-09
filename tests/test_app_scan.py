"""The Any dataset page's verdict block, driven through Streamlit's AppTest against a tiny dataset on local disk."""
import pytest
import huggingface_hub
import streamlit as st
from fsspec.implementations.local import LocalFileSystem
from streamlit.testing.v1 import AppTest
from test_hub import write_v2, write_v3

APP = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app" / "app.py")


def elements(node):
    """Every element under `node`, in page order."""
    for child in getattr(node, "children", {}).values():
        yield child
        yield from elements(child)


def verdict_page(tmp_path, monkeypatch, corrupt=(), kinds=None, open_episode=None):
    """Run the page on a 4-episode dataset (one data file each) in which the episodes in `corrupt` cannot be read."""
    st.cache_data.clear()      # the page caches scans by repo name; each test builds a different "lab/x"
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "datasets").mkdir()
    if kinds is None:
        write_v2(str(tmp_path / "datasets"), "lab/x", lengths=(50, 60, 70, 80))
    else:      # v3, with chosen episodes' state copied from action
        write_v3(str(tmp_path / "datasets"), "lab/x", lengths=(200, 250, 300), kinds=kinds)
    for i in corrupt:
        (tmp_path / f"datasets/lab/x/data/chunk-000/episode_{i:06d}.parquet").write_bytes(b"not parquet")
    monkeypatch.setattr(huggingface_hub, "HfFileSystem", lambda token=None: LocalFileSystem())     # the Hub is a local dir
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["ds_repo"] = "lab/x"
    if open_episode is not None:
        at.session_state["ds_want"] = open_episode      # what a ?ep= in the URL leaves for the page to open
    at.run()
    assert not at.exception
    return at


def test_unreadable_episodes_are_flagged_above_the_verdict_numbers(tmp_path, monkeypatch):
    at = verdict_page(tmp_path, monkeypatch, corrupt=(1,))
    order = [type(e).__name__ + ":" + str(getattr(e, "value", "")) for e in elements(at.main)]
    warning = next(i for i, e in enumerate(order) if e.startswith("Warning:") and "could not be checked" in e)
    tiles = next(i for i, e in enumerate(order) if e.startswith("Html:") and ">Passed<" in e)
    assert warning < tiles


def test_a_verdict_resting_on_mostly_unreadable_episodes_is_called_unreliable(tmp_path, monkeypatch):
    at = verdict_page(tmp_path, monkeypatch, corrupt=(1, 2, 3))
    assert any("unreliable" in w.value for w in at.warning)


def test_a_mostly_readable_dataset_is_not_called_unreliable(tmp_path, monkeypatch):
    at = verdict_page(tmp_path, monkeypatch, corrupt=(1,))
    assert at.warning and not any("unreliable" in w.value for w in at.warning)


def checked_text(at):
    """The 'N checked' caption under the Episodes tile."""
    return next(e.value for e in elements(at.main) if type(e).__name__ == "Html" and ">Episodes<" in str(e.value))


def test_a_sampled_scan_offers_to_check_every_episode_and_does_so_on_click(tmp_path, monkeypatch):
    import qgate.hub
    monkeypatch.setattr(qgate.hub, "MAX_SCAN_EPISODES", 2)
    at = verdict_page(tmp_path, monkeypatch)
    assert "2 checked" in checked_text(at)
    at.button(key="ds_scan_all").click().run()
    assert not at.exception
    assert "4 checked" in checked_text(at)
    assert not [b for b in at.button if b.key == "ds_scan_all"]        # nothing left to scan


def test_a_dataset_scanned_in_full_has_no_scan_all_button(tmp_path, monkeypatch):
    at = verdict_page(tmp_path, monkeypatch)
    assert not [b for b in at.button if b.key == "ds_scan_all"]


def html_of(at):
    return "\n".join(str(e.value) for e in elements(at.main) if type(e).__name__ == "Html")


def test_state_equal_to_action_throughout_is_shown_as_deliberate_not_failed(tmp_path, monkeypatch):
    at = verdict_page(tmp_path, monkeypatch, kinds={0: "copy", 1: "copy", 2: "copy"})
    page = html_of(at)
    assert ">Failed<" in page                                                                     # the tiles rendered
    row = next(r for r in page.split("<span class=\"s ") if "state copied from action" in r)
    assert row.startswith("pass") and "deliberate" in row


def test_state_equal_to_action_in_some_episodes_stays_a_failure(tmp_path, monkeypatch):
    at = verdict_page(tmp_path, monkeypatch, kinds={0: "copy"})
    row = next(r for r in html_of(at).split("<span class=\"s ") if "state copied from action" in r)
    assert row.startswith("fail") and "deliberate" not in row


def test_opening_an_episode_agrees_with_the_dataset_verdict_on_a_convention(tmp_path, monkeypatch):
    all_copy = verdict_page(tmp_path / "a", monkeypatch, kinds={0: "copy", 1: "copy", 2: "copy"}, open_episode=0)
    assert "<b>state copied from action.</b>" not in html_of(all_copy)
    some_copy = verdict_page(tmp_path / "b", monkeypatch, kinds={0: "copy"}, open_episode=0)
    assert "<b>state copied from action.</b>" in html_of(some_copy)


import re

STRIP_KINDS = {0: "idle", 1: "copy", 2: "move"}       # 0 warns (still for its first 4 s), 1 fails, 2 passes


# ---- the overview: every episode at a glance, one column each, one row per check

def matrix_rows(at):
    """row label -> [(episode, status)] for each row of the overview, read from its cells."""
    page = next(h for h in html_of(at).split("\n") if 'class="qg-mx' in h)
    labels = re.findall(r"<i>([^<]+)</i>", re.search(r'<div class="labs">(.*?)</div>', page).group(1))
    rows = re.findall(r'<div class="cells">(.*?)</div>', page)
    return {label: [(int(ep), st_) for st_, ep in re.findall(r'<i class="c (\w+)"[^>]*data-ep="(\d+)"', cells)]
            for label, cells in zip(labels, rows)}


def test_the_overview_has_a_row_per_check_marking_where_it_fired(tmp_path, monkeypatch):
    rows = matrix_rows(verdict_page(tmp_path, monkeypatch, kinds=STRIP_KINDS))
    assert rows["idle start"] == [(0, "warn"), (1, "pass"), (2, "pass")]
    assert rows["state copied from action"] == [(0, "pass"), (1, "fail"), (2, "pass")]
    assert set(rows) == {"no signal", "too short", "idle start", "idle end", "state copied from action"}


def test_clicking_a_cell_opens_that_episode_and_plays_it(tmp_path, monkeypatch):
    at = verdict_page(tmp_path, monkeypatch, kinds=STRIP_KINDS)
    at.text_input(key="ds_jump").set_value("2:1731000000000").run()      # what the cell's click handler writes
    assert not at.exception
    assert at.session_state["ds_open"] == 2
    page = html_of(at)
    assert 'id="qgh"' in page and 'id="ds-ep"' in page                   # the player, and the anchor the page scrolls to
    assert "playing: prev ? prev.playing : true" in page                  # a newly opened episode plays without a click
    assert "dataset verdict" not in " ".join(b.label for b in at.button)  # no back button


@pytest.mark.parametrize("kinds, opened", [
    (STRIP_KINDS, 0),                  # 0 warns: the first flagged episode, warn or fail
    ({0: "move", 1: "copy"}, 1),       # 0 passes, 1 fails
    ({}, 0),                           # nothing flagged: the first episode
])
def test_entering_a_dataset_opens_the_first_flagged_episode_or_else_the_first(tmp_path, monkeypatch, kinds, opened):
    at = verdict_page(tmp_path, monkeypatch, kinds=kinds)
    assert at.session_state["ds_open"] == opened
    assert 'id="qgh"' in html_of(at) and f"<b>Episode {opened}</b>" in html_of(at)


def test_a_check_that_fires_in_every_episode_shows_as_passing_in_the_overview(tmp_path, monkeypatch):
    at = verdict_page(tmp_path, monkeypatch, kinds={0: "copy", 1: "copy", 2: "copy"})
    assert matrix_rows(at)["state copied from action"] == [(0, "pass"), (1, "pass"), (2, "pass")]
