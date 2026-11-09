"""qgate: check any public LeRobot dataset on the Hugging Face Hub, episode by episode, with the universal checks.

Run:  streamlit run app/app.py
"""
import json, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from qgate.checks import CODES, relax, universal_verdict, UNIVERSAL_CODES

st.set_page_config(page_title="qgate", layout="wide")

# Colour is spent only on problems. Values chosen to read on both the light and dark theme.
ACCENT, CRIT, WARN, GOOD = "#3987e5", "#d03b3b", "#fab219", "#0ca30c"
NEUTRAL, NEUTRAL_LINE, MUTED_LINE = "rgba(128,140,155,0.22)", "rgba(128,140,155,0.45)", "rgba(140,150,162,0.85)"
FLAG_FILL = {"fail": "rgba(208,59,59,0.20)", "warn": "rgba(250,178,25,0.22)"}
STATUS = {"pass": ("Pass", "✓"), "warn": ("Review", "!"), "fail": ("Quarantined", "✕")}
FONT = dict(family="IBM Plex Sans, system-ui, sans-serif", size=11)

st.html(f"""<style>
[data-testid="stMainBlockContainer"] {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1680px; }}
.qg-brand {{ font-weight:600; font-size:17px; display:flex; align-items:center; gap:8px; }}
.qg-brand i {{ width:10px; height:10px; border-radius:2px; background:{ACCENT}; display:inline-block; }}
.qg-eyebrow {{ font-size:11px; text-transform:uppercase; letter-spacing:.08em; opacity:.65; font-weight:500; }}
.qg-v {{ font-family:'IBM Plex Mono', ui-monospace, monospace; font-size:28px; font-weight:500; line-height:1.15; margin-top:4px;
         display:flex; align-items:baseline; gap:8px; font-variant-numeric:tabular-nums; }}
.qg-v small {{ font-family:'IBM Plex Sans', sans-serif; font-size:12px; opacity:.65; font-weight:400; }}
.qg-v.alert b {{ color:{CRIT}; }}
.qg-v b {{ font-weight:500; }}
.qg-h {{ display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }}
.qg-h b {{ font-size:14px; font-weight:600; }}
.qg-h span {{ font-size:12px; opacity:.65; }}
.qg-pill {{ display:inline-flex; align-items:center; gap:6px; font-size:12px; font-weight:500; padding:1px 8px 1px 5px;
            border-radius:999px; border:1px solid {NEUTRAL_LINE}; white-space:nowrap; }}
.qg-pill i {{ width:14px; height:14px; border-radius:50%; display:inline-grid; place-items:center; font-size:9px;
              font-style:normal; font-weight:700; color:#fff; }}
.qg-pill.pass i {{ background:{GOOD}; }} .qg-pill.warn i {{ background:{WARN}; color:#3b2a00; }} .qg-pill.fail i {{ background:{CRIT}; }}
.qg-note {{ font-size:12.5px; opacity:.7; margin:6px 0 12px; }}
.qg-ckl {{ display:grid; grid-template-columns:auto max-content minmax(0, 1fr); gap:5px 10px; font-size:12.5px; align-items:center; margin-top:6px; }}
.qg-ckl .s {{ width:8px; height:8px; border-radius:50%; background:{NEUTRAL_LINE}; }}
.qg-ckl .s.fail {{ background:{CRIT}; }} .qg-ckl .s.warn {{ background:{WARN}; }}
.qg-sum {{ display:flex; flex-wrap:wrap; gap:6px 18px; font-size:12.5px; margin:2px 0 4px; }}
.qg-sum b {{ font-weight:600; }} .qg-sum span {{ opacity:.85; }}
.qg-ckl .d {{ font-family:'IBM Plex Mono', monospace; font-size:11.5px; opacity:.6; text-align:right; padding-left:12px; }}
.qg-ckl [title] {{ cursor:help; }}
.qg-mxw {{ container-type:inline-size; margin:8px 0 6px; }}
.qg-mx {{ --vis:min(75, var(--n)); --lab:170px; --cnt:84px; --g:3px; --rg:5px; --pad:3px; --mg:14px;
          --cs:max(9px, calc((100cqw - var(--lab) - var(--cnt) - 2 * var(--mg) - 2 * var(--pad) - (var(--vis) - 1) * var(--g) - 1px) / var(--vis)));
          --ch:min(var(--cs), 22px);   /* fewer episodes than fit: columns widen to fill the row, height stops at 22px */
          display:flex; gap:var(--mg); font-size:12.5px; }}
.qg-mx .labs, .qg-mx .cnts {{ display:flex; flex-direction:column; gap:var(--rg); padding:var(--pad) 0; flex:none; }}
.qg-mx .labs {{ width:var(--lab); }} .qg-mx .cnts {{ width:var(--cnt); }}
.qg-mx .labs span, .qg-mx .cnts span {{ height:var(--ch); display:flex; align-items:center; white-space:nowrap; }}
.qg-mx .labs i {{ font-style:normal; line-height:1.35; min-width:0; overflow:hidden; text-overflow:ellipsis; }}   /* taller than the row, so descenders show */
.qg-mx .cnts span {{ justify-content:flex-end; font-family:'IBM Plex Mono', monospace; font-size:11.5px; opacity:.7; font-variant-numeric:tabular-nums; }}
.qg-mx .scroll {{ flex:1; min-width:0; overflow-x:auto; overflow-y:hidden; padding:var(--pad); scrollbar-width:thin; }}
.qg-mx .rows {{ display:flex; flex-direction:column; gap:var(--rg); width:max-content; }}
.qg-mx .cells, .qg-mx .axis {{ display:grid; grid-auto-flow:column; grid-auto-columns:var(--cs); gap:var(--g); }}
.qg-mx .cells {{ height:var(--ch); }}
.qg-mx .axis {{ font-family:'IBM Plex Mono', monospace; font-size:10.5px; opacity:.55; padding-top:2px; }}
.qg-mx .c, .qg-mx-key .c {{ display:block; border-radius:2px; background:{NEUTRAL}; box-shadow:inset 0 0 0 1px rgba(127,127,127,.12); }}
.qg-mx .c {{ cursor:pointer; }}
.qg-mx .c.fail, .qg-mx-key .c.fail {{ background:{CRIT}; }} .qg-mx .c.warn, .qg-mx-key .c.warn {{ background:{WARN}; }}
.qg-mx .c:hover {{ outline:2px solid {ACCENT}; outline-offset:1px; }}
@container (max-width: 640px) {{ .qg-mx {{ --vis:min(15, var(--n)); --lab:96px; --cnt:0px; --mg:4px; font-size:11.5px; }} .qg-mx .cnts {{ display:none; }} }}
.qg-mx-key {{ display:flex; flex-wrap:wrap; gap:4px 16px; font-size:12px; opacity:.8; margin:2px 0 10px; }}
.qg-mx-key span {{ display:inline-flex; align-items:center; gap:6px; }}
.qg-mx-key .c {{ width:10px; height:10px; }}
.st-key-ds_jump {{ display:none; }}
</style>""")


# ------------------------------------------------------------------ small HTML pieces
def pill(v):
    label, glyph = STATUS[v]
    return f'<span class="qg-pill {v}"><i>{glyph}</i>{label}</span>'


TICK_EVERY = 25      # an episode number under every 25th column


def overview(scan, fps):
    """Every episode checked as one square per check, GitHub-style, 75 columns in view and the rest a sideways scroll away.
    Past 2,000 episodes a column stands for a run of consecutive episodes, to keep the page light."""
    from qgate.overview import columns
    by_idx = {e.idx: e for e in scan.episodes}
    cols = columns(by_idx, max_cols=2000)
    rank = {"pass": 0, "warn": 1, "fail": 2}

    def cell(run, status_of, what):
        worst = max(run, key=lambda i: rank[status_of(i)])
        st_ = status_of(worst)
        if len(run) == 1:
            e = by_idx[worst]
            tip = f"Episode {worst} · {e.n / float(fps):.0f} s · {what(worst)}"
            return f'<i class="c {st_}" data-ep="{worst}" title="{tip}"></i>'
        hit = sum(status_of(i) != "pass" for i in run)
        share = hit / len(run)
        tip = f"Episodes {run[0]}-{run[-1]}: {hit} of {len(run)} flagged"
        return f'<i class="c {st_}" data-ep="{worst}" title="{tip}" style="opacity:{0.35 + 0.65 * share:.2f}"></i>' if hit else \
               f'<i class="c pass" data-ep="{worst}" title="{tip}"></i>'

    labs, rows, cnts = [], [], []
    for code in UNIVERSAL_CODES:
        status = {e.idx: next((f.status for f in e.findings if f.code == code), "pass") for e in scan.episodes}
        detail = {e.idx: next((f.detail for f in e.findings if f.code == code), "") for e in scan.episodes}
        if code in scan.deliberate:      # fires in every episode: how the dataset was recorded, not a fault, so shown as a pass
            status = {i: "pass" for i in status}
        count = f"{sum(v != 'pass' for v in status.values()):,}"
        labs.append(f'<span title="{CODES[code]}"><i>{CODES[code]}</i></span>')
        rows.append('<div class="cells">' + "".join(cell(run, status.get, detail.get) for run in cols) + "</div>")
        cnts.append(f"<span>{count}</span>")
    ticks = "".join(f'<span style="grid-column:{k + 1} / span {min(TICK_EVERY, len(cols) - k)}">{cols[k][0]}</span>' for k in range(0, len(cols), TICK_EVERY))
    rows.append(f'<div class="axis">{ticks}</div>')
    key = ('<div class="qg-mx-key"><span><i class="c fail"></i>quarantined</span><span><i class="c warn"></i>review</span>'
           '<span><i class="c pass"></i>pass</span></div>')
    return (f'<div class="qg-mxw"><div class="qg-mx" style="--n:{len(cols)}"><div class="labs">{"".join(labs)}</div>'
            f'<div class="scroll"><div class="rows">{"".join(rows)}</div></div><div class="cnts">{"".join(cnts)}</div></div></div>{key}')


def head(title, sub=""):
    st.html(f'<div class="qg-h"><b>{title}</b><span>{sub}</span></div>')


def stat(label, value, sub="", alert=False):
    with st.container(border=True):
        st.html(f'<div class="qg-eyebrow">{label}</div><div class="qg-v {"alert" if alert else ""}"><b>{value}</b><small>{sub}</small></div>')


def base_layout(fig, height, **kw):
    kw.setdefault("margin", dict(l=0, r=8, t=8, b=0))
    fig.update_layout(height=height, font=FONT, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", **kw)
    return fig


# ------------------------------------------------------------------ a player the browser owns
# The <video> is the clock: play, pause, seek and speed happen in the browser with no rerun, and the Signals cursor reads the
# player's time. The HTML only changes with the episode, camera or flagged stretches, so reruns leave a playing video alone.
PLAYER_JS = """(() => {
  const C = __CFG__, root = document.getElementById("qgh"); if (!root) return;
  const v = root.querySelector("video"), $ = s => root.querySelector(s);
  const prev = window.__qgst && window.__qgst.key === C.key ? window.__qgst : null;      // same episode (e.g. camera switch): resume
  const S = window.__qgst = {key: C.key, t: prev ? prev.t : 0, playing: prev ? prev.playing : true, speed: prev ? prev.speed : 1};
  let t0 = performance.now();                    // anchor of the virtual clock used when there is no video
  const clamp = t => Math.min(Math.max(t, 0), C.dur);
  const P = window.__qgp = {
    t() { return v ? clamp(v.currentTime - C.start) : S.playing ? clamp(S.t + (performance.now() - t0) / 1000 * S.speed) : S.t; },
    play() { if (P.t() >= C.dur - 0.05) P.seek(0); S.playing = true; t0 = performance.now(); if (v) v.play().catch(() => {}); draw(); },
    pause() { S.t = P.t(); S.playing = false; if (v) v.pause(); draw(); },
    toggle() { S.playing ? P.pause() : P.play(); },
    seek(t) { S.t = clamp(t); t0 = performance.now(); if (v && v.readyState >= 1) v.currentTime = C.start + S.t; draw(); },
    rate(r) { S.t = P.t(); t0 = performance.now(); S.speed = r; if (v) v.playbackRate = r; draw(); },
  };
  // the sanitizer drops a script whose text looks like it holds a tag, so the icons spell "<" as \\u003c
  const ICON = {play: '\\u003csvg viewBox="0 0 24 24">\\u003cpath d="M8 5v14l11-7z"/>\\u003c/svg>',
                pause: '\\u003csvg viewBox="0 0 24 24">\\u003cpath d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z"/>\\u003c/svg>'};
  const fmt = t => Math.floor(t / 60) + ":" + String(Math.floor(t % 60)).padStart(2, "0");
  function draw() {
    const t = P.t(), f = C.dur ? t / C.dur * 100 : 0;
    $(".fill").style.width = f + "%"; $(".knob").style.left = f + "%";
    $(".time").textContent = fmt(t) + " / " + fmt(C.dur);
    $(".pp").innerHTML = S.playing ? ICON.pause : ICON.play;
    $(".spd").textContent = S.speed + "×";
    root.classList.toggle("paused", !S.playing);
  }
  if (v) {
    v.addEventListener("loadedmetadata", () => { v.currentTime = C.start + S.t; v.playbackRate = S.speed; if (S.playing) v.play().catch(() => {}); });
    v.addEventListener("error", () => { $(".err").style.display = "grid"; });
    v.addEventListener("click", P.toggle);
  }
  $(".pp").addEventListener("click", P.toggle);
  $(".spd").addEventListener("click", () => P.rate({1: 2, 2: 4, 4: 0.5, 0.5: 1}[S.speed] || 1));
  const bar = $(".bar"); let drag = false;
  const at = e => { const r = bar.getBoundingClientRect(); P.seek((e.clientX - r.left) / r.width * C.dur); };
  bar.addEventListener("pointerdown", e => { drag = true; bar.setPointerCapture(e.pointerId); at(e); });
  bar.addEventListener("pointermove", e => { if (drag) at(e); });
  bar.addEventListener("pointerup", () => { drag = false; });
  if (window.__qgpi) clearInterval(window.__qgpi);
  window.__qgpi = setInterval(() => {
    if (!document.body.contains(root)) return clearInterval(window.__qgpi);
    const t = P.t();
    if (v && v.readyState >= 1 && v.currentTime < C.start - 0.05) v.currentTime = C.start + S.t;    // packed file: stay in the clip
    if (S.playing && t >= C.dur - 0.03) { P.pause(); S.t = C.dur; if (v) v.currentTime = C.start + C.dur - 0.02; }   // stop at the clip's end
    else if (S.playing) S.t = t;
    draw();
  }, 100);
  draw();
})();"""


def hub_player(key, url, start, dur, ranges, aspect="4/3"):
    """YouTube-style controls over the episode's video, shown on hover: play/pause, seek bar (flagged stretches marked), time, speed.
    Without a video, the same bar runs a clock of its own so the Signals cursor still plays."""
    marks = "".join(f'<i style="left:{a / dur * 100:.2f}%;width:{max(b - a, dur / 200) / dur * 100:.2f}%;background:{CRIT if s == "fail" else WARN}"></i>'
                    for s, a, b in ranges if dur)
    media = (f'<video muted playsinline preload="auto" src="{url}"></video>'
             '<div class="err">Video unavailable in this browser.<br>The signals below still play.</div>') if url else ""
    cfg = json.dumps(dict(key=key, start=round(start, 4), dur=round(dur, 4)))
    st.html(f"""<style>
      #qgh {{ position:relative; border-radius:6px; overflow:hidden; background:#000; user-select:none; }}
      #qgh video {{ width:100%; aspect-ratio:{aspect}; display:block; object-fit:contain; cursor:pointer; }}
      #qgh.novid {{ height:44px; }}
      #qgh .err {{ display:none; position:absolute; inset:0; place-items:center; text-align:center; padding:24px; color:#e6eaf0; font-size:13px; }}
      #qgh .ctl {{ position:absolute; left:0; right:0; bottom:0; padding:22px 10px 6px; color:#fff;
                   background:linear-gradient(transparent, rgba(0,0,0,.65)); opacity:0; transition:opacity .2s; }}
      #qgh:hover .ctl, #qgh.paused .ctl, #qgh.novid .ctl {{ opacity:1; }}
      #qgh.novid .ctl {{ padding-top:6px; background:none; }}
      #qgh .bar {{ position:relative; height:14px; cursor:pointer; touch-action:none; }}
      #qgh .bar::before, #qgh .fill, #qgh .bar .marks i {{ content:""; position:absolute; top:5px; height:4px; border-radius:2px; transition:height .1s, top .1s; }}
      #qgh .bar::before {{ left:0; right:0; background:rgba(255,255,255,.3); }}
      #qgh .bar .marks i {{ opacity:.85; }}
      #qgh .fill {{ left:0; background:{ACCENT}; }}
      #qgh .bar:hover::before, #qgh .bar:hover .fill, #qgh .bar:hover .marks i {{ top:4px; height:6px; }}
      #qgh .knob {{ position:absolute; top:7px; width:12px; height:12px; margin:-6px 0 0 -6px; border-radius:50%; background:{ACCENT};
                    transform:scale(0); transition:transform .1s; }}
      #qgh .bar:hover .knob, #qgh .ctl:hover .knob {{ transform:scale(1); }}
      #qgh .row {{ display:flex; align-items:center; gap:10px; height:28px; }}
      #qgh button {{ all:unset; cursor:pointer; display:inline-flex; align-items:center; justify-content:center; }}
      #qgh .pp {{ width:28px; height:28px; }} #qgh .pp svg {{ width:22px; height:22px; fill:#fff; }}
      #qgh .time {{ font:12px 'IBM Plex Mono', monospace; font-variant-numeric:tabular-nums; }}
      #qgh .spd {{ margin-left:auto; font:600 12px 'IBM Plex Mono', monospace; padding:2px 6px; border-radius:4px; }}
      #qgh .spd:hover {{ background:rgba(255,255,255,.18); }}
    </style>
    <div id="qgh" class="{'' if url else 'novid'}">{media}
      <div class="ctl"><div class="bar"><span class="marks">{marks}</span><span class="fill"></span><span class="knob"></span></div>
        <div class="row"><button class="pp" title="Play / pause"></button><span class="time"></span>
          <button class="spd" title="Playback speed"></button></div></div></div>
    <script>{PLAYER_JS.replace("__CFG__", cfg)}</script>""", unsafe_allow_javascript=True)


# ------------------------------------------------------------------ page
st.html('<div class="qg-brand"><i></i>qgate</div>')

# Stream any public LeRobot dataset from the Hub: its frames are read into memory (nothing saved), replayed on the
# playback clock, and checked by the universal checks (no thresholds; each caught real faults on the Hub, ticket T8).
from huggingface_hub import HfFileSystem
from qgate.hub import TooLarge, inspect_dataset, load_episode, scan_dataset, video_source

# one per check, each from a different dataset (only that check fires, except too short: every sub-1 s episode
# in the Hub sample is also static)
EXAMPLES = ["lerobot/svla_so100_pickplace", "samsitol/so100_peg01", "cedricxie/robofac-lerobot-preview", "maux/so100_test"]

# why each flag matters for training a policy (primary sources: tickets/T8-research-state-action-checks.md)
WHY = {"JOINT-005": "A static episode teaches the policy to do nothing; no-op labels make policies freeze (OpenVLA, arXiv 2406.09246, App. C / E.1).",
       "EPIS-004": "Leading idle frames are training samples that map the start pose to \"stay still\" (Diffusion Policy, arXiv 2303.04137, §4.3, §6.1; OpenVLA App. C). Trim them.",
       "EPIS-005": "Trailing idle frames add no-op samples after the task is done (same sources as idle start). Trim them.",
       "EPIS-003": "Under one second is too short to hold a full demonstration; it is probably a mis-trigger of the recorder.",
       "XSIG-002": "Action should be the leader arm's command, recorded apart from the follower state (ACT, arXiv 2304.13705, §IV). Identical or shifted copies can mean the label was derived from state, though some datasets do this on purpose."}

def hub_error(ex_, what):
    """One message for a failed Hub call; a 429 says to wait (and how long) instead of dumping the exception."""
    resp = getattr(ex_, "response", None)
    if getattr(resp, "status_code", None) == 429 or "429" in str(ex_)[:200]:
        wait = (resp.headers.get("Retry-After") if resp is not None else None) or ""
        return ("The Hugging Face Hub is rate-limiting this connection (anonymous limit: 500 requests per 5 minutes). "
                f"Try again{f' in about {wait} s' if wait.isdigit() else ' in a minute or two'}; "
                "setting HF_TOKEN before launching the app raises the limit.")
    return f"Could not {what} on the Hub: {type(ex_).__name__}: {ex_}"

def hub_fs():
    """HF_TOKEN (env, or Streamlit secrets on a hosted app) lifts the anonymous 500-requests-per-5-minutes limit."""
    import os
    try:
        token = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN")
    except Exception:      # no secrets file
        token = None
    return HfFileSystem(token=token or None)

@st.cache_data(ttl=600, show_spinner=False)
def hub_inspect(repo):
    return inspect_dataset(repo, hub_fs())

@st.cache_data(ttl=600, max_entries=16, show_spinner=False)
def hub_episode(repo, idx):
    return load_episode(hub_inspect(repo), idx, hub_fs())

if "ds_repo" not in st.session_state:
    st.session_state["ds_repo"] = st.query_params.get("repo", "")      # nothing loads until a dataset is entered
    _qe = st.query_params.get("ep", "")
    st.session_state["ds_want"] = int(_qe) if _qe.isdigit() else None
H = st.session_state.setdefault("hub", dict(repo=None, ep=None, opened=0))

def _example(repo):
    st.session_state["ds_repo"], st.session_state["ds_open"] = repo, None   # examples show the dataset verdict

with st.container(border=True):
    st.text_input("LeRobot dataset on the Hugging Face Hub", key="ds_repo", placeholder="owner/name, e.g. lerobot/svla_so100_pickplace")
    ex = st.columns(len(EXAMPLES) + 1, vertical_alignment="center")
    ex[0].html('<div class="qg-note" style="margin:0">Examples:</div>')
    for col, name in zip(ex[1:], EXAMPLES):
        col.button(name, key=f"ex_{name}", on_click=_example, args=(name,), width="stretch")

repo = st.session_state["ds_repo"].strip().strip("/").removeprefix("https://huggingface.co/datasets/")
if not repo:
    st.stop()
try:
    with st.spinner(f"Checking {repo} on the Hub…"):
        insp = hub_inspect(repo)
except Exception as ex_:
    st.error(hub_error(ex_, f"reach {repo}"))
    st.stop()
if not insp.ok:
    st.warning(f"**{insp.step[0].upper() + insp.step[1:]}.** {insp.message}")
    st.stop()

eps = sorted(insp.episodes)
if st.session_state.get("ds_seen") != repo:      # new dataset: only a ?ep= in the URL opens an episode straight away
    st.session_state["ds_seen"] = repo
    st.session_state["ds_open"] = st.session_state.pop("ds_want", None)
cams = insp.cameras
sn, an = insp.state_names, insp.action_names
st.html('<div class="qg-sum">' + "".join(f"<span>{k} <b>{v}</b></span>" for k, v in (
    ("LeRobot", insp.version), ("robot", insp.robot or "unknown"), ("fps", insp.fps), ("episodes", f"{len(eps):,}"),
    ("frames", f"{insp.info.get('total_frames', sum(insp.episodes.values())):,}"), ("state", len(sn)), ("action", len(an)),
    ("cameras", len(cams)))) + "</div>")

@st.cache_data(ttl=600, max_entries=8, show_spinner=False)
def hub_scan(repo, full=False):
    insp_ = hub_inspect(repo)
    return scan_dataset(insp_, hub_fs(), max_episodes=len(insp_.episodes) if full else None)

try:
    with st.spinner(f"Checking {len(eps):,} episodes of {repo}…"):
        scan = hub_scan(repo, st.session_state.get("ds_full") == repo)
except Exception as ex_:
    st.error(hub_error(ex_, f"scan {repo}")); st.stop()

def _open(idx):
    st.session_state["ds_open"] = idx
    st.session_state["ds_scroll"] = True            # pan down to the episode once it renders
    H["opened"] = H.get("opened", 0) + 1            # a fresh player key: starts from the top and plays

def _jump():
    ep = st.session_state["ds_jump"].split(":")[0]
    if ep.isdigit():
        _open(int(ep))

# ---- the dataset verdict: what a user gets on entering a dataset
flagged = {}                                         # code -> worst status over episodes, and which episodes
for e in scan.episodes:
    for f in e.findings:
        if f.status != "pass":
            flagged[f.code] = "fail" if "fail" in (f.status, flagged.get(f.code)) else "warn"
with st.container(border=True):
    st.html(f'<div class="qg-h"><b>{repo}</b></div>')
    if scan.unreadable:      # before the numbers: they only describe the episodes that could be read
        why = next(iter(scan.unreadable.values()))
        tried = scan.scanned + len(scan.unreadable)
        note = (f" The verdict below rests on {scan.scanned:,} of the {tried:,} episodes sampled, so treat it as unreliable."
                if len(scan.unreadable) > scan.scanned else "")
        st.warning(f"{len(scan.unreadable):,} episodes could not be checked ({why}).{note}")
    cc = st.columns(4)
    with cc[0]: stat("Episodes", f"{scan.total:,}", f"{scan.scanned:,} checked")
    with cc[1]: stat("Passed", f"{scan.passed:,}", f"{scan.passed / max(scan.scanned, 1):.0%} of checked")
    with cc[2]: stat("Warned", f"{scan.warned:,}", "trim or review")
    with cc[3]: stat("Failed", f"{scan.failed:,}", "keep out of training", alert=scan.failed > 0)
    rows = []
    for code in UNIVERSAL_CODES:
        n_ep = len(scan.by_code.get(code, []))
        what = ("every episode checked · looks deliberate, not a fault" if code in scan.deliberate
                else f"{n_ep} episode{'s' if n_ep != 1 else ''}")
        rows.append(f'<span class="s {flagged.get(code, "pass")}"></span><span>{CODES[code]}</span>'
                    f'<span class="d">{what}</span>')
    st.html(f'<div class="qg-ckl">{"".join(rows)}</div>')
    if scan.scanned + len(scan.unreadable) < scan.total:      # an evenly spaced sample, not every episode
        st.html(f'<div class="qg-note">Checked an evenly spaced sample of {scan.scanned + len(scan.unreadable):,} of {scan.total:,} episodes.</div>')
        st.button(f"Scan all {scan.total:,} episodes", key="ds_scan_all", on_click=st.session_state.__setitem__, args=("ds_full", repo),
                  help="Reads every episode's state and action columns. On a large dataset this can take minutes and download a lot of data.")

with st.container(border=True):
    head("Episodes", "click a cell to open that episode below: its video, signals and which checks it tripped")
    if scan.episodes:
        st.html(overview(scan, insp.fps))
    # a click on a cell writes "<episode>:<nonce>" into a hidden input, which opens that episode below
    st.text_input("Open episode", key="ds_jump", label_visibility="collapsed", on_change=_jump)
    st.html("""<script>(() => {
      if (window.__qgmx) return; window.__qgmx = true;
      document.addEventListener("click", ev => {
        const c = ev.target.closest(".qg-mx .c[data-ep]"); if (!c) return;
        const inp = document.querySelector(".st-key-ds_jump input"); if (!inp) return;
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(inp, c.dataset.ep + ":" + Date.now());
        inp.dispatchEvent(new Event("input", {bubbles: true}));
        inp.dispatchEvent(new KeyboardEvent("keydown", {key: "Enter", code: "Enter", keyCode: 13, which: 13, bubbles: true}));
      });
    })();</script>""", unsafe_allow_javascript=True)

ep_idx = st.session_state.get("ds_open")
if ep_idx is not None and ep_idx not in insp.episodes:
    st.warning(f"Episode {ep_idx} is not in this dataset."); ep_idx = None
if ep_idx is None:      # nothing picked yet: open the first episode that warned or failed, else the first episode
    ep_idx = next((e.idx for e in sorted(scan.episodes, key=lambda e: e.idx) if e.verdict != "pass"),
                  min((e.idx for e in scan.episodes), default=eps[0]))
    st.session_state["ds_open"] = ep_idx
st.query_params["repo"], st.query_params["ep"] = repo, str(ep_idx)
st.html('<div id="ds-ep"></div>')

if (H["repo"], H["ep"], H.get("dl")) != (repo, ep_idx, scan.deliberate):
    try:
        with st.spinner(f"Streaming episode {ep_idx} from the Hub…"):
            src = hub_episode(repo, ep_idx)
    except TooLarge as ex_:
        st.warning(str(ex_)); st.stop()
    except Exception as ex_:
        st.error(hub_error(ex_, f"read episode {ep_idx}")); st.stop()
    verdict, findings = universal_verdict(src)     # decided on load: no waiting for the episode to play out
    verdict, findings = relax(findings, scan.deliberate)      # same reading of a dataset-wide convention as the verdict above
    H.update(repo=repo, ep=ep_idx, dl=scan.deliberate, src=src, verdict=verdict, findings={f.code: f for f in findings})
src, findings = H["src"], H["findings"]
dur = src.n / src.fps

# stretches of the episode the checks flagged (shaded on the plot, and jump targets for the video)
ranges, whole = [], None
for code, f in findings.items():
    if f.status == "pass":
        continue
    if code in ("EPIS-004", "EPIS-005") and f.span:
        ranges.append((code, f.status, *f.span))
    elif f.status == "fail":
        whole = code
if whole:
    ranges.insert(0, (whole, "fail", 0, src.n))

def _hjump(frame):
    st.session_state["ds_seek"] = frame / H["src"].fps        # applied in the browser, at the end of the page

pairable = sn == an
chans = list(range(min(8, len(sn))))                    # the first 8 channels; no picker
marks = [(status, a / src.fps, b / src.fps) for _, status, a, b in ranges if (a, b) != (0, src.n)]     # shown on the seek bar
v_col, p_col = st.columns([1.25, 2])
with v_col, st.container(border=True):
    if cams:
        cam = cams[0]
        if len(cams) > 1:
            cam = st.segmented_control("Camera", cams, default=cams[0], key=f"ds_cam_{repo}", label_visibility="collapsed",
                                       format_func=lambda k: k.split(".")[-1]) or cams[0]
        url, vstart, _ = video_source(insp, ep_idx, cam)
        shape = insp.info["features"][cam].get("shape") or [480, 640, 3]
        h, w = (shape[1], shape[2]) if len(shape) == 3 and shape[0] in (1, 3) else (shape[0], shape[1])
        hub_player(f"{repo}|{ep_idx}|{H['opened']}", url, vstart, dur, marks, aspect=f"{w}/{h}")
    else:
        hub_player(f"{repo}|{ep_idx}|{H['opened']}", None, 0.0, dur, marks)
        st.html('<div class="qg-note">This dataset has no video; the bar above plays the signals.</div>')

with p_col, st.container(border=True):
    st.html(f'<div class="qg-h"><b>Episode {ep_idx}</b>{pill(H["verdict"])}'
            f'<span>{src.n} frames · {dur:.1f} s</span></div>')
    rows = []
    for code, f in findings.items():
        why = WHY.get(code, "")
        rows.append(f'<span class="s {f.status}"></span><span title="{f.detail}">{CODES[code]}</span>'
                    f'<span class="d">{f.detail}</span>')
    st.html(f'<div class="qg-ckl">{"".join(rows)}</div>')
    for code, f in findings.items():
        if f.status != "pass" and code in WHY:
            st.html(f'<div class="qg-note"><b>{CODES[code]}.</b> {WHY[code]}</div>')
    if ranges:
        rc = st.columns(min(len(ranges), 3))
        for col, (code, status, a, b) in zip(rc, ranges):
            col.button(f"{CODES[code]} · {a / src.fps:.1f}–{b / src.fps:.1f} s", key=f"ds_rng_{code}", on_click=_hjump,
                       args=(a,), width="stretch")
    else:
        st.html('<div class="qg-note">Nothing flagged: no stretch to jump to.</div>')

def hub_fig():
    """State channels (solid) with the commanded action (dashed) when state and action share names."""
    k = len(chans)
    fig = make_subplots(rows=k, cols=1, shared_xaxes=True, vertical_spacing=0.012)
    t = np.arange(src.n) / src.fps
    for r, j in enumerate(chans):
        full = np.concatenate([src.state[:, j], src.action[:, j]] if pairable else [src.state[:, j]])
        full = full[np.isfinite(full)]
        if pairable:
            fig.add_trace(go.Scatter(x=t, y=src.action[:, j], line=dict(width=1, color=MUTED_LINE, dash="dot"),
                                     name=f"{sn[j]} (action)", hovertemplate="%{y:.3g}"), row=r + 1, col=1)
        fig.add_trace(go.Scatter(x=t, y=src.state[:, j], line=dict(width=1.5, color=ACCENT), name=sn[j],
                                 hovertemplate="%{y:.3g}"), row=r + 1, col=1)
        if len(full):
            span = (full.max() - full.min()) or 1.0
            fig.update_yaxes(range=[full.min() - .06 * span, full.max() + .42 * span], row=r + 1, col=1)
        fig.add_annotation(xref=f"x{r + 1 if r else ''} domain", yref=f"y{r + 1 if r else ''} domain", x=0, y=1, xanchor="left",
                           yanchor="top", text=f"<b>{sn[j]}</b>", showarrow=False, font=dict(size=11.5))
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=r + 1, col=1)
        for code, status, s0, e0 in ranges:
            fig.add_vrect(x0=s0 / src.fps, x1=max(e0, s0 + 1) / src.fps, fillcolor=FLAG_FILL[status], line_width=0,
                          layer="below", row=r + 1, col=1)
    base_layout(fig, 58 * k + 30, hovermode="x unified", hoversubplots="axis", margin=dict(l=0, r=8, t=4, b=0))
    fig.update_xaxes(range=[0, dur], showgrid=True, gridcolor=NEUTRAL, zeroline=False)
    fig.update_xaxes(ticksuffix=" s", row=k, col=1)
    return fig

# The chart is drawn once per episode change; the cursor is a line the browser slides over it, reading the player's
# time. Redrawing the figure every tick to move the cursor made the whole chart blink.
with st.container(border=True):
    head("Signals")
    st.plotly_chart(hub_fig(), key="ds_sig", config={"displayModeBar": False}, width="stretch")
st.html("""<script>(() => {
  if (window.__qgcur) clearInterval(window.__qgcur);
  window.__qgcur = setInterval(() => {
    const t = window.__qgp ? window.__qgp.t() : 0;
    const gd = document.querySelector(".st-key-ds_sig .js-plotly-plot"), fl = gd && gd._fullLayout;
    if (!fl || !fl.xaxis || !fl.xaxis.l2p) return;
    let line = gd.querySelector(".qg-cursor");
    if (!line) {
      line = document.createElement("div"); line.className = "qg-cursor";
      line.style.cssText = "position:absolute;width:1px;background:__ACCENT__;pointer-events:none;z-index:2";
      gd.style.position = "relative"; gd.appendChild(line);
    }
    line.style.left = (fl.xaxis._offset + fl.xaxis.l2p(t)) + "px";
    line.style.top = fl._size.t + "px"; line.style.height = fl._size.h + "px";
  }, 50);
})();</script>""".replace("__ACCENT__", ACCENT), unsafe_allow_javascript=True)

# one-off browser actions go last, so adding or dropping them never shifts (and remounts) the player above
if st.session_state.pop("ds_scroll", False):
    st.html(f"""<script>setTimeout(() => document.getElementById("ds-ep")?.scrollIntoView({{behavior: "smooth", block: "start"}}), 150)
            </script><!-- {time.time()} -->""", unsafe_allow_javascript=True)
if (seek := st.session_state.pop("ds_seek", None)) is not None:
    st.html(f"<script>window.__qgp && window.__qgp.seek({seek:.3f})</script><!-- {time.time()} -->", unsafe_allow_javascript=True)
