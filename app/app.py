"""qgate dashboard ("Board" layout): fleet health, check status history, threshold headroom, episode inspector;
Gate validation page with the benchmark and a live fault lab.

Reads precomputed files in app/data (see scripts/build_app_data.py) and bench/results/results.json.
Run:  streamlit run app/app.py
"""
import json, pathlib, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from numpy.lib.stride_tricks import sliding_window_view
from plotly.subplots import make_subplots

from bench.inject import FAULTS, SEVERITIES, inject
from qgate.checks import CODES, GATE_CODES, first_move, run_checks, universal_verdict, velocity, UNIVERSAL_CODES
from qgate.dataset import Episode
from qgate.live import PENDING, LiveGate

st.set_page_config(page_title="qgate", layout="wide")

# Colour is spent only on problems. Values chosen to read on both the light and dark theme.
ACCENT, CRIT, WARN, GOOD = "#3987e5", "#d03b3b", "#fab219", "#0ca30c"
NEUTRAL, NEUTRAL_LINE, MUTED_LINE = "rgba(128,140,155,0.22)", "rgba(128,140,155,0.45)", "rgba(140,150,162,0.85)"
FLAG_FILL = {"fail": "rgba(208,59,59,0.20)", "warn": "rgba(250,178,25,0.22)"}
SEQ = ["rgba(128,140,155,0.14)", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"]   # detection rate 0 -> 10/10
STATUS = {"pass": ("Pass", "✓"), "warn": ("Review", "!"), "fail": ("Quarantined", "✕")}
PROV_KEY = {"TIME-001": "ts_tol", "TIME-003": "vid_tol_frames", "JOINT-002": "range", "JOINT-003": "vel_limit", "JOINT-004": "stuck",
            "GRIP-001": "grip_min_range", "XSIG-001": "lag", "EPIS-001": "length", "EPIS-002": "idle"}
FONT = dict(family="IBM Plex Sans, system-ui, sans-serif", size=11)

st.html(f"""<style>
[data-testid="stMainBlockContainer"] {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1680px; }}
.qg-brand {{ font-weight:600; font-size:17px; display:flex; align-items:center; gap:8px; }}
.qg-brand i {{ width:10px; height:10px; border-radius:2px; background:{ACCENT}; display:inline-block; }}
.st-key-nav, .st-key-nav > div {{ display:flex; justify-content:flex-end; width:100%; }}
.st-key-nav [data-testid="stButtonGroup"] {{ justify-content:flex-end; margin-left:auto; }}
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
.qg-code {{ font-family:'IBM Plex Mono', monospace; font-size:11.5px; padding:1px 6px; border-radius:4px;
            border:1px solid {NEUTRAL_LINE}; background:{NEUTRAL}; white-space:nowrap; }}
.qg-f {{ border-left:3px solid {CRIT}; padding:2px 0 2px 10px; margin-bottom:10px; }}
.qg-f.warn {{ border-left-color:{WARN}; }}
.qg-f p {{ margin:4px 0 0; font-size:13px; opacity:.85; }}
.qg-note {{ font-size:12.5px; opacity:.7; margin:6px 0 12px; }}
.qg-ckl {{ display:grid; grid-template-columns:auto 1fr auto; gap:5px 10px; font-size:12.5px; align-items:center; margin-top:6px; }}
.qg-ckl .s {{ width:8px; height:8px; border-radius:50%; background:{NEUTRAL_LINE}; }}
.qg-ckl .s.fail {{ background:{CRIT}; }} .qg-ckl .s.warn {{ background:{WARN}; }}
.qg-ckl .s.pending {{ background:transparent; border:1.5px dashed {NEUTRAL_LINE}; box-sizing:border-box; }}
.qg-sum {{ display:flex; flex-wrap:wrap; gap:6px 18px; font-size:12.5px; margin:2px 0 4px; }}
.qg-sum b {{ font-weight:600; }} .qg-sum span {{ opacity:.85; }}
.qg-log {{ font-size:12.5px; display:grid; grid-template-columns:auto auto auto 1fr; gap:4px 10px; align-items:center; }}
.qg-log .t {{ font-family:'IBM Plex Mono', monospace; font-size:11.5px; opacity:.65; white-space:nowrap; }}
.qg-live {{ display:inline-flex; align-items:center; gap:6px; font-size:12px; font-weight:500; }}
.qg-live i {{ width:8px; height:8px; border-radius:50%; background:{CRIT}; animation:qgpulse 1.2s infinite; }}
.qg-live.off i {{ background:{NEUTRAL_LINE}; animation:none; }}
@keyframes qgpulse {{ 50% {{ opacity:.25; }} }}
.qg-ckl .d {{ font-family:'IBM Plex Mono', monospace; font-size:11.5px; opacity:.6; text-align:right; }}
.qg-ckl [title] {{ cursor:help; }}
.qg-miss {{ padding-left:18px; margin:0; font-size:13px; }} .qg-miss li {{ margin-bottom:6px; }}
.qg-outcome {{ font-size:15px; margin:4px 0 2px; }}
</style>""")


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    rep = json.load(open(ROOT / "app/data/report.json"))
    meta = json.load(open(ROOT / "app/data/meta.json"))
    sig = pd.read_parquet(ROOT / "app/data/signals.parquet")
    bench = json.load(open(ROOT / "bench/results/results.json"))
    return rep, meta, sig, bench


REP, META, SIG, BENCH = load()
TH = REP["thresholds"]
EPS = {e["episode"]: e for e in REP["episodes"]}
JOINTS = META["joints"]
JN = [j.replace("_", " ") for j in JOINTS]
FPS = float(META["fps"])
COUNTS = {v: sum(e["verdict"] == v for e in EPS.values()) for v in STATUS}


def episode_obj(idx) -> Episode:
    d = SIG[SIG.episode == idx]
    return Episode(idx, FPS, d[[f"s{j}" for j in range(6)]].to_numpy(float), d[[f"a{j}" for j in range(6)]].to_numpy(float),
                   d.timestamp.to_numpy(float), d.frame_index.to_numpy(np.int64), dict(META["video_frames"][str(idx)]))


def unit(j):
    return "" if j == len(JOINTS) - 1 else "°"


def runs(idx, width=1):
    out = []
    for f in idx:
        s, e = int(f), int(f) + width - 1
        if out and s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def evidence(ep: Episode, findings, th):
    """Turn non-pass findings into per-joint frame intervals and a plain-language sentence."""
    S = np.nan_to_num(ep.state)
    lo, hi = np.array(th["range_lo"]), np.array(th["range_hi"])
    out = []
    for f in findings:
        if f["status"] == "pass":
            continue
        code, st_ = f["code"], f["status"]
        if code == "JOINT-002":
            for j in range(S.shape[1]):
                for side, idx in (("below", np.where(S[:, j] < lo[j])[0]), ("above", np.where(S[:, j] > hi[j])[0])):
                    if len(idx):
                        ext = S[idx, j].min() if side == "below" else S[idx, j].max()
                        lim = lo[j] if side == "below" else hi[j]
                        out.append(dict(code=code, status=st_, joint=j, iv=runs(idx), text=(
                            f"<b>{JN[j]}</b> went {side} its calibrated range for <b>{len(idx)} frames</b>, "
                            f"reaching {ext:.1f}{unit(j)} against a limit of {lim:.1f}{unit(j)}.")))
        elif code == "JOINT-004" and ep.n >= th["stuck_window"]:
            w = th["stuck_window"]
            ptp = np.ptp(sliding_window_view(S, w, axis=0)[::5], axis=-1)
            for j in range(S.shape[1] - 1):
                starts = np.where((ptp[:, j] == 0) & (np.delete(ptp, j, axis=1).max(1) > th["stuck_other_motion"]))[0] * 5
                if len(starts):
                    out.append(dict(code=code, status=st_, joint=j, iv=runs(starts, w), text=(
                        f"<b>{JN[j]}</b> held exactly {S[starts[0], j]:.1f}° for {w}+ frames while the other joints kept moving.")))
        elif code == "JOINT-003":
            ratio = velocity(ep) / np.array(th["vel_limit"])
            for j in range(S.shape[1]):
                idx = np.where(ratio[:, j] > 1)[0]
                if len(idx):
                    out.append(dict(code=code, status=st_, joint=j, iv=runs(idx, 2), text=(
                        f"<b>{JN[j]}</b> moved at up to <b>{ratio[:, j].max():.1f}×</b> its velocity limit.")))
        elif code in ("TIME-001", "TIME-002"):
            # recompute every bad frame (Finding.frames stops at 40); shade only when the problem is local,
            # since jitter or drift touches nearly every frame and would just paint the whole plot
            if code == "TIME-001":
                bad = np.where((np.abs(ep.ts - ep.frame_index / ep.fps) > th["ts_tol"])[:-1] | (np.diff(ep.ts) <= 0))[0]
            else:
                bad = np.where(np.diff(ep.frame_index) != 1)[0]
            local = len(bad) <= 0.3 * ep.n
            out.append(dict(code=code, status=st_, joint=None, iv=runs(bad, 2) if local else [],
                            text=f["detail"].capitalize() + ("." if local else " (spread across the episode, so not shaded).")))
        elif code == "GRIP-001":
            out.append(dict(code=code, status=st_, joint=len(JOINTS) - 1, iv=[[0, ep.n - 1]], text=f["detail"].capitalize() + "."))
        else:
            out.append(dict(code=code, status=st_, joint=None, iv=runs(f.get("frames", [])), text=f["detail"].capitalize() + "."))
    return out


# ------------------------------------------------------------------ small HTML pieces
def pill(v):
    label, glyph = STATUS[v]
    return f'<span class="qg-pill {v}"><i>{glyph}</i>{label}</span>'


def head(title, sub=""):
    st.html(f'<div class="qg-h"><b>{title}</b><span>{sub}</span></div>')


def stat(label, value, sub="", alert=False):
    with st.container(border=True):
        st.html(f'<div class="qg-eyebrow">{label}</div><div class="qg-v {"alert" if alert else ""}"><b>{value}</b><small>{sub}</small></div>')


def checklist(checks):
    rows = []
    for c in checks:
        k = PROV_KEY.get(c["code"])
        origin, how = TH["provenance"][k] if k else ("structural", "no threshold: any violation fails")
        rows.append(f'<span class="s {c["status"]}"></span><span title="{c["detail"]} · threshold: {origin}, {how}">{CODES[c["code"]]}</span>'
                    f'<span class="d">{c["code"]}</span>')
    return f'<div class="qg-ckl">{"".join(rows)}</div>'


def base_layout(fig, height, **kw):
    kw.setdefault("margin", dict(l=0, r=8, t=8, b=0))
    fig.update_layout(height=height, font=FONT, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", **kw)
    return fig


# ------------------------------------------------------------------ charts
def status_history(sel):
    codes = list(GATE_CODES)
    ids = sorted(EPS)
    lv = {"pass": 0, "warn": 1, "fail": 2}
    z, text = [], []
    rows = ["Episode verdict"] + [f"{c}  {CODES[c]}" for c in codes]
    z.append([lv[EPS[i]["verdict"]] for i in ids])
    text.append([f"<b>Episode {i}</b> · {STATUS[EPS[i]['verdict']][0]}<br>{EPS[i]['n']} frames" for i in ids])
    for c in codes:
        r, t = [], []
        for i in ids:
            chk = next(x for x in EPS[i]["checks"] if x["code"] == c)
            r.append(lv[chk["status"]]); t.append(f"<b>Episode {i}</b> · {c}<br>{chk['detail']}")
        z.append(r); text.append(t)
    fig = go.Figure(go.Heatmap(z=z, x=ids, y=rows, text=text, hovertemplate="%{text}<extra></extra>", xgap=2, ygap=2, showscale=False,
                               zmin=0, zmax=2, colorscale=[[0, NEUTRAL], [.33, NEUTRAL], [.33, WARN], [.67, WARN], [.67, CRIT], [1, CRIT]]))
    fig.add_shape(type="rect", x0=sel - .5, x1=sel + .5, y0=-.6, y1=len(rows) - .4, line=dict(color=ACCENT, width=2))
    fig.add_vline(x=24.5, line=dict(color=MUTED_LINE, width=1, dash="dot"))
    fig.add_hline(y=.5, line=dict(color=NEUTRAL_LINE, width=1))
    for x, label in ((-.5, "calibration · ep 0–24"), (24.6, "held-out · ep 25–49")):
        fig.add_annotation(x=x, y=1, yref="paper", yanchor="bottom", xanchor="left", text=label, showarrow=False, font=dict(size=11), opacity=.7)
    base_layout(fig, 360, margin=dict(l=0, r=8, t=24, b=0))
    fig.update_yaxes(autorange="reversed", tickfont=dict(family="IBM Plex Mono, monospace", size=11))
    fig.update_xaxes(tickvals=list(range(0, len(ids), 5)), showgrid=False, zeroline=False)
    return fig


def headroom_bars(values, limit, sel, tick_fmt, name):
    ids = sorted(EPS)
    colors = [ACCENT if i == sel else CRIT if values[i] > limit else NEUTRAL_LINE for i in ids]
    fig = go.Figure(go.Bar(x=ids, y=[values[i] for i in ids], marker=dict(color=colors, cornerradius=2), marker_line_width=0,
                           customdata=[tick_fmt(values[i]) for i in ids],
                           hovertemplate=f"<b>Episode %{{x}}</b><br>{name}: %{{customdata}}<extra></extra>",
                           selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))))
    fig.add_hline(y=limit, line=dict(color=CRIT, width=1, dash="dash"))
    base_layout(fig, 170, bargap=.25)
    top = max(limit * 1.15, max(values.values()) * 1.05)
    fig.update_yaxes(range=[0, top], tickvals=[0, limit], ticktext=["0", tick_fmt(limit)], showgrid=False, zeroline=False,
                     tickfont=dict(family="IBM Plex Mono, monospace"))
    fig.update_xaxes(tickvals=list(range(0, len(ids), 10)), showgrid=False)
    return fig


def signal_fig(ep: Episode, flags, ref: Episode | None = None, row_h=78, full: Episode | None = None):
    """`full` (Live page): the episode being streamed; fixes both axes to it so the plot fills in left to right."""
    k = len(JOINTS)
    fig = make_subplots(rows=k, cols=1, shared_xaxes=True, vertical_spacing=0.012)
    flagged = {f["joint"] for f in flags if f["joint"] is not None}
    t = np.arange(ep.n) / FPS
    if full is not None:
        ref_span = np.nan_to_num(full.state)
        fig.update_xaxes(range=[0, full.n / FPS])
        fig.add_vline(x=max(ep.n - 1, 0) / FPS, line=dict(color=ACCENT, width=1))
    for j in range(k):
        if ref is not None:
            fig.add_trace(go.Scatter(x=np.arange(ref.n) / FPS, y=ref.state[:, j], line=dict(width=1, color=MUTED_LINE),
                                     name=f"{JN[j]} (original)", hovertemplate="%{y:.1f}"), row=j + 1, col=1)
        dim = flagged and j not in flagged and ref is None
        fig.add_trace(go.Scatter(x=t, y=ep.state[:, j], line=dict(width=1.6 if not dim else 1.2, color=MUTED_LINE if dim else ACCENT),
                                 name=JN[j], hovertemplate="%{y:.1f}" + unit(j)), row=j + 1, col=1)
        s = ep.state[:, j][np.isfinite(ep.state[:, j])]
        rng = f"{s.min():.1f} to {s.max():.1f}{unit(j)}" if len(s) else ""
        if ref is not None:
            s = np.concatenate([s, ref.state[:, j]])
        if full is not None:
            s = ref_span[:, j]
        if len(s):   # headroom above the data so the joint label never sits on the line
            span = (s.max() - s.min()) or 1.0
            fig.update_yaxes(range=[s.min() - .06 * span, s.max() + .42 * span], row=j + 1, col=1)
        fig.add_annotation(xref=f"x{j + 1 if j else ''} domain", yref=f"y{j + 1 if j else ''} domain", x=0, y=1, xanchor="left", yanchor="top",
                           text=f"<b>{JN[j]}</b>  <span style='font-size:10px'>{rng}</span>", showarrow=False, font=dict(size=11.5),
                           bgcolor="rgba(0,0,0,0)")
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=j + 1, col=1)
    for f in flags:
        rows = range(k) if f["joint"] is None else [f["joint"]]
        for s0, e0 in f["iv"]:
            for j in rows:
                fig.add_vrect(x0=s0 / FPS, x1=max(e0, s0 + 1) / FPS, fillcolor=FLAG_FILL[f["status"]], line_width=0, layer="below", row=j + 1, col=1)
    base_layout(fig, row_h * k + 30, hovermode="x unified", hoversubplots="axis", margin=dict(l=0, r=8, t=4, b=0))
    fig.update_xaxes(showgrid=True, gridcolor=NEUTRAL, zeroline=False, showspikes=True, spikemode="across", spikesnap="cursor",
                     spikethickness=1, spikedash="solid", spikecolor=MUTED_LINE)
    fig.update_xaxes(ticksuffix=" s", row=k, col=1)
    return fig


def bench_heatmap():
    table, sevs = BENCH["table"], ["subtle", "moderate", "obvious"]
    faults = list(table)
    bucket = lambda r: 4 if r >= 1 else 3 if r >= .75 else 2 if r >= .5 else 1 if r > 0 else 0
    z, text = [], []
    for f in faults:
        zr, tr = [], []
        for s in sevs:
            c = table[f][s]
            zr.append(bucket(c["flagged"] / c["n"]))
            flag = " ⚑" if c["attributed"] < c["flagged"] else ""
            tr.append(f"<b>{c['flagged']}/{c['n']}{flag}</b>   {c['param']} {c['unit']}")
        z.append(zr); text.append(tr)
    scale = []
    for i, col in enumerate(SEQ):
        scale += [[i / 5, col], [(i + 1) / 5, col]]
    fig = go.Figure(go.Heatmap(z=z, x=sevs, y=[f.replace("_", " ") for f in faults], text=text, texttemplate="%{text}", textfont=dict(size=12),
                               hovertemplate="%{y}, %{x}<extra></extra>", zmin=-.5, zmax=4.5, colorscale=scale, showscale=False, xgap=3, ygap=3))
    base_layout(fig, 560, margin=dict(l=0, r=8, t=28, b=0))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    return fig


# ------------------------------------------------------------------ playback (Live and Any dataset pages)
# One clock drives a stream: the server feeds frames to the streaming gate up to clock() * fps, and the browser
# seeks the video to the same clock. It counts episode-seconds, so speed changes re-anchor it.
TICK = 0.4    # seconds between dashboard refreshes; the gate still checks every frame in between


def clock(L):
    return L["base"] + ((time.time() - L["anchor"]) * L["speed"] if L["anchor"] is not None else 0.0)


def synced_video(url, start, dur, base, speed, playing, aspect="4/3"):
    """A <video> that follows the playback clock: `start` is the clip's offset inside the file (v3 packs episodes)."""
    st.html(f"""<div style="position:relative"><video id="qgv" muted playsinline preload="auto" src="{url}"
      style="width:100%;aspect-ratio:{aspect};border-radius:6px;background:#000;display:block;object-fit:contain"></video>
      <div id="qgv-err" style="display:none;position:absolute;inset:0;place-items:center;text-align:center;padding:24px;
           color:#e6eaf0;font-size:13px">Video unavailable in this browser.<br>The gate and signals below still run.</div></div>
    <script>(() => {{
      const v = document.getElementById("qgv"); if (!v) return;
      v.addEventListener("error", () => {{ document.getElementById("qgv-err").style.display = "grid"; }});
      if (window.__qgv) clearInterval(window.__qgv);
      const start = {start:.4f}, dur = {dur:.4f}, base = {base:.4f}, speed = {speed}, playing = {str(playing).lower()};
      const t0 = performance.now();
      const want = () => Math.min(playing ? base + (performance.now() - t0) / 1000 * speed : base, dur - 0.02);
      window.__qgv = setInterval(() => {{
        if (v.readyState < 1) return;           // wait for metadata before seeking
        v.playbackRate = speed;
        const t = want(), on = playing && t < dur - 0.03;
        if (Math.abs(v.currentTime - (start + t)) > 0.25) v.currentTime = start + t;
        if (on && v.paused) v.play().catch(() => {{}});
        if (!on && !v.paused) v.pause();
      }}, 200);
    }})();</script>""", unsafe_allow_javascript=True)


# ------------------------------------------------------------------ episode selection (shared by all clickable charts)
# The page and episode live in the URL (?page=validation&ep=31) so a link opens the same view.
PAGES = {"episodes": "Episodes", "live": "Live", "dataset": "Any dataset", "validation": "Gate validation"}
if "ep_sel" not in st.session_state:
    _fails = [i for i, e in EPS.items() if e["verdict"] == "fail"]
    _qp = st.query_params.get("ep", "")
    st.session_state["ep_sel"] = int(_qp) if _qp.isdigit() and int(_qp) in EPS else \
        next((i for i in _fails if i >= 25), _fails[0] if _fails else 0)   # first held-out failure
if "nav" not in st.session_state:
    st.session_state["nav"] = PAGES.get(st.query_params.get("page", ""), "Episodes")
    st.session_state["nav_last"] = st.session_state["nav"]


def _nav_changed():
    # a segmented control can be clicked off; keep the current page instead of showing nothing
    if st.session_state["nav"] is None:
        st.session_state["nav"] = st.session_state["nav_last"]
    st.session_state["nav_last"] = st.session_state["nav"]


def _pick(key):
    ev = st.session_state.get(key)
    pts = (ev or {}).get("selection", {}).get("points", [])
    if pts and pts[0].get("x") is not None:
        st.session_state["ep_sel"] = int(pts[0]["x"])


# ------------------------------------------------------------------ page
top_l, top_nav = st.columns([2, 1], vertical_alignment="center")
with top_l:
    st.html('<div class="qg-brand"><i></i>qgate</div>')
with top_nav:
    st.segmented_control("Page", list(PAGES.values()), key="nav", on_change=_nav_changed, label_visibility="collapsed")

PAGE = st.session_state["nav"] or "Episodes"
st.query_params["page"] = next(k for k, v in PAGES.items() if v == PAGE)
if PAGE == "Episodes":
    st.query_params["ep"] = str(st.session_state["ep_sel"])
    st.query_params.pop("repo", None)
elif PAGE != "Any dataset":          # that page writes its own ?repo=…&ep=…
    st.query_params.pop("ep", None)
    st.query_params.pop("repo", None)

if PAGE == "Episodes":
    sel = st.session_state["ep_sel"]
    failed = [i for i, e in EPS.items() if e["verdict"] == "fail"]
    c = st.columns(4)
    with c[0]: stat("Episodes checked", len(EPS), f"{len(SIG):,} frames")
    with c[1]: stat("Passed", COUNTS["pass"], f"{COUNTS['pass'] / len(EPS):.0%}")
    with c[2]: stat("Quarantined", COUNTS["fail"], ", ".join(f"ep {i}" for i in failed), alert=COUNTS["fail"] > 0)
    with c[3]: stat("Needs review", COUNTS["warn"], "warn verdicts")

    with st.container(border=True):
        head("Check status history", "every check × every episode · grey = pass · click a cell to inspect that episode")
        st.plotly_chart(status_history(sel), key="mx", on_select=lambda: _pick("mx"), selection_mode="points",
                        config={"displayModeBar": False}, width="stretch")

    peak = {i: max(float(x["detail"].split()[2][:-1]) for x in e["checks"] if x["code"] == "JOINT-003") for i, e in EPS.items()}
    idle = {i: float(next(x for x in e["checks"] if x["code"] == "EPIS-002")["detail"].split("%")[0]) / 100 for i, e in EPS.items()}
    lo, hi = np.array(TH["range_lo"]), np.array(TH["range_hi"])
    S_all = SIG[[f"s{j}" for j in range(6)]].to_numpy()
    out_mask = ((S_all < lo) | (S_all > hi)).any(axis=1)
    outr = pd.Series(out_mask).groupby(SIG.episode.to_numpy()).sum().astype(int).to_dict()
    c = st.columns(3)
    for col, key, title, sub, vals, lim, fmt in (
        (c[0], "b1", "Peak velocity", "× per-joint limit · warn above 1×", peak, 1.0, lambda v: f"{v:.2f}×"),
        (c[1], "b2", "Idle time", "share of frames with every joint below 1°/s", idle, TH["idle_max"], lambda v: f"{v:.0%}"),
        (c[2], "b3", "Frames outside joint range", "any joint beyond its calibrated limit", outr, 0.0, lambda v: f"{v:.0f}"),
    ):
        with col, st.container(border=True):
            head(title, sub)
            st.plotly_chart(headroom_bars(vals, lim, sel, fmt, title.lower()), key=key, on_select=lambda k=key: _pick(k),
                            selection_mode="points", config={"displayModeBar": False}, width="stretch")

    e = EPS[sel]
    ep = episode_obj(sel)
    flags = evidence(ep, e["checks"], TH)
    left, right = st.columns([3, 1])
    with left, st.container(border=True):
        h1, h2 = st.columns([3, 1], vertical_alignment="center")
        with h1:
            st.html(f'<div class="qg-h"><b>Episode {sel}</b>{pill(e["verdict"])}'
                    f'<span>{"calibration" if sel < 25 else "held-out"} · {e["n"]} frames · {e["n"] / FPS:.1f} s · hover to read values</span></div>')
        with h2:
            st.selectbox("Episode", sorted(EPS), key="ep_sel", label_visibility="collapsed",
                         format_func=lambda i: f"Episode {i}  {STATUS[EPS[i]['verdict']][1]}")
        st.plotly_chart(signal_fig(ep, flags), key="sig", config={"displayModeBar": False}, width="stretch")
    with right, st.container(border=True):
        head("Findings")
        if flags:
            st.html("".join(f'<div class="qg-f {f["status"]}"><span class="qg-code">{f["code"]}</span> {CODES[f["code"]]}<p>{f["text"]}</p></div>'
                            for f in flags))
            if e["verdict"] != "pass":
                st.html('<div class="qg-note">Real data, nothing injected. Limits come from 25 calibration episodes, '
                        "so a natural reach past them is a likely false alarm.</div>")
        else:
            st.html('<div class="qg-note">All 11 checks passed.</div>')
        st.html(checklist(e["checks"]))

elif PAGE == "Live":
    MIX = "Random mix"
    FAULT_OPTS = ["None", MIX] + list(FAULTS)

    def load(L, idx, play):
        fault, sev = L["fault"], L["sev"]
        rng = np.random.default_rng(idx * 1009 + L["round"])
        if fault == MIX:   # about half the episodes arrive with a random fault, like a fleet with a few bad rigs
            fault, sev = (list(FAULTS)[int(rng.integers(len(FAULTS)))], int(rng.integers(3))) if rng.random() < .5 else ("None", 0)
        base = episode_obj(idx)
        src = base if fault == "None" else inject(base, fault, sev, dict(TH), rng)
        L.update(ep=idx, src=src, gate=LiveGate(TH, FPS, src.video_frames, idx), cursor=0, base=0.0,
                 anchor=time.time() if play else None, token=L["token"] + 1, cost=[],
                 truth=None if fault == "None" else (fault, sev, FAULTS[fault][2]))

    if "live" not in st.session_state:
        st.session_state["live"] = L = dict(fault=MIX, sev=1, speed=1.0, auto=True, round=0, token=0, log=[], tally=[])
        load(L, 25, play=False)
    L = st.session_state["live"]

    def _play():
        if L["gate"].final is not None:
            L["round"] += 1; load(L, L["ep"], play=True)
        elif L["anchor"] is None:
            L["anchor"] = time.time()
        else:
            L["base"], L["anchor"] = clock(L), None

    def _restart():
        L["round"] += 1; load(L, L["ep"], play=L["anchor"] is not None)

    def _next():
        L["round"] += 1; load(L, (L["ep"] + 1) % len(EPS), play=True)

    def _ep():
        load(L, st.session_state["lv_ep"], play=False)

    def _fault():
        L["fault"], L["sev"] = st.session_state["lv_fault"], SEVERITIES.index(st.session_state["lv_sev"])
        load(L, L["ep"], play=False)

    def _speed():
        if st.session_state["lv_speed"] is None:          # clicked off: keep the current speed
            st.session_state["lv_speed"] = f"{L['speed']:g}×"; return
        L["base"] = clock(L)
        L["anchor"] = time.time() if L["anchor"] is not None else None
        L["speed"] = float(st.session_state["lv_speed"].rstrip("×"))

    st.session_state["lv_ep"] = L["ep"]          # auto-advance moves the episode; keep the picker in step
    playing = L["anchor"] is not None
    c = st.columns([1.1, 1.5, 1.1, .9, .7, .7, .7, 1.1], vertical_alignment="bottom")
    c[0].selectbox("Episode", sorted(EPS), key="lv_ep", on_change=_ep, format_func=lambda i: f"Episode {i}")
    c[1].selectbox("Inject fault", FAULT_OPTS, key="lv_fault", index=FAULT_OPTS.index(L["fault"]), on_change=_fault,
                   format_func=lambda f: f.replace("_", " "), help="Random mix: each episode has a 50% chance of a random fault and severity.")
    c[2].select_slider("Severity", SEVERITIES, key="lv_sev", value=SEVERITIES[L["sev"]], on_change=_fault,
                       disabled=L["fault"] in ("None", MIX))
    c[3].segmented_control("Speed", ["1×", "2×", "4×"], key="lv_speed", default=f"{L['speed']:g}×", on_change=_speed)
    c[4].button("Pause" if playing else "Play", on_click=_play, type="primary", width="stretch")
    c[5].button("Restart", on_click=_restart, width="stretch")
    c[6].button("Next", on_click=_next, width="stretch")
    st.session_state.setdefault("lv_auto", L["auto"])
    L["auto"] = c[7].toggle("Auto-advance", key="lv_auto", help="Stream the next episode when this one ends.")

    gate, src = L["gate"], L["src"]
    v_col, p_col = st.columns([1.25, 2])
    with v_col, st.container(border=True):
        cam = st.segmented_control("Camera", ["top", "wrist"], default="top", key="lv_cam", label_visibility="collapsed") or "top"
        # Prefer the episode's own H.264 clip served by this app (plays in every browser); fall back to seeking
        # into the Hub's long AV1 file if the clips were not built.
        clip = f"app/static/clips/{cam}/ep{L['ep']:03d}.mp4"
        if (ROOT / clip).exists() and st.get_option("server.enableStaticServing"):   # off until the server restarts
            vurl, vstart = clip, 0.0
        else:
            vid = META["videos"][str(L["ep"])][cam]
            vurl, vstart = f"{vid['url']}#t={vid['start']:.3f}", vid["start"]
        # Rendered on full reruns only (controls, episode change); the fragments below never touch it.
        synced_video(vurl, vstart, src.n / FPS, clock(L), L["speed"], playing)
        truth = L["truth"]
        inj = f"Injected: <b>{truth[0].replace('_', ' ')}</b>, {SEVERITIES[truth[1]]}. The fault is in the signals the gate reads; the video is the original recording." \
            if truth else "Nothing injected: real recorded signals."
        st.html(f'<div class="qg-note">{inj} The video follows the same clock as the gate, give or take network delay.</div>')

    @st.fragment(run_every=TICK if playing else None)
    def live_panel():
        """Advance the clock: push every frame due since the last tick through the gate, then draw the state."""
        L = st.session_state["live"]; gate, src = L["gate"], L["src"]
        target = min(src.n, int(clock(L) * FPS) + 1)
        t0 = time.perf_counter()
        for k in range(L["cursor"], target):
            for e in gate.push(src.ts[k], src.frame_index[k], src.state[k], src.action[k]):
                L["log"].append(dict(e, ep=L["ep"]))
        if target > L["cursor"]:
            L["cost"].append((time.perf_counter() - t0) / (target - L["cursor"]))
        L["cursor"] = target
        if L["cursor"] == src.n and gate.final is None:
            for e in gate.finish():
                L["log"].append(dict(e, ep=L["ep"]))
            got = {f.code for f in gate.final[1] if f.status != "pass"}
            live_first = [f for c, (f, s) in gate.first.items()]
            truth = L["truth"]
            if truth:
                outcome = "caught" if got & truth[2] else "wrong code" if got else "missed"
            else:
                outcome = "clean" if not got else "flagged (real data)"
            L["tally"].append(dict(ep=L["ep"], fault=truth[0] if truth else None, sev=truth[1] if truth else None, verdict=gate.verdict,
                                   first=min(live_first) / FPS if live_first else None, dur=src.n / FPS, outcome=outcome,
                                   codes=sorted(got)))
        if gate.final is not None and L["anchor"] is not None:
            if L["auto"] and clock(L) > src.n / FPS + 1.5 * L["speed"]:   # hold the result on screen ~1.5 s
                L["round"] += 1; load(L, (L["ep"] + 1) % len(EPS), play=True); st.rerun(scope="app")
            elif not L["auto"]:
                L["base"], L["anchor"] = src.n / FPS, None; st.rerun(scope="app")

        n = L["cursor"]; t = n / FPS
        live = L["anchor"] is not None and gate.final is None
        first = min(gate.first.items(), key=lambda kv: kv[1][0], default=None)
        cost = np.mean(L["cost"][-200:]) * 1e3 if L["cost"] else 0.0
        st.html(f'<div class="qg-h"><span class="qg-live {"" if live else "off"}"><i></i>{"LIVE" if live else "ended" if gate.final else "paused"}</span>'
                f'<b>Episode {L["ep"]}</b>{pill(gate.verdict)}<span>{"final verdict" if gate.final else "verdict so far"}</span></div>')
        c = st.columns(3)
        with c[0]: stat("Frame", f"{n}", f"of {src.n} · {t:.1f} / {src.n / FPS:.1f} s")
        with c[1]:
            if first:
                code, (fr, s) = first
                stat("First flag", f"{fr / FPS:.1f} s", f"{code} · {'at episode end' if fr >= src.n - 1 and gate.final else 'mid-episode'}", alert=True)
            else:
                stat("First flag", "none", "no finding yet")
        with c[2]: stat("Gate cost", f"{cost:.2f}", f"ms per frame · budget {1000 / FPS:.0f} ms")
        rows = []
        for code, s in gate.status.items():
            name = CODES[code]
            when = f"{gate.first[code][0] / FPS:.1f} s" if code in gate.first else "waits for end" if s == PENDING else "ok so far" if not gate.final else "pass"
            rows.append(f'<span class="s {s}"></span><span title="{gate.detail[code]}">{name}</span><span class="d">{code} · {when}</span>')
        st.html(f'<div class="qg-ckl" style="grid-template-columns:auto 1fr auto">{"".join(rows)}</div>'
                '<div class="qg-note">Solid = decided frame by frame · dashed = needs the whole episode (video count, lag, idle, short episode, gripper that never moved).</div>')

    @st.fragment(run_every=TICK if playing else None)
    def live_signals():
        L = st.session_state["live"]; gate, src = L["gate"], L["src"]
        part = gate.episode() if gate.n else None
        flags = evidence(part, gate.findings(), TH) if part is not None and part.n > 1 else []
        left, right = st.columns([3, 1])
        with left, st.container(border=True):
            head("Signals as they arrive", "joint state up to the current frame · shading = frames the gate has flagged so far")
            if part is not None and part.n > 1:
                st.plotly_chart(signal_fig(part, flags, row_h=58, full=src), key="lv_sig", config={"displayModeBar": False}, width="stretch")
        with right, st.container(border=True):
            head("Event log", "newest first")
            ev = L["log"][-14:][::-1]
            st.html('<div class="qg-log">' + "".join(
                f'<span class="t">ep {e["ep"]} · {e["frame"] / FPS:.1f}s</span>{pill(e["status"])}<span class="qg-code">{e["code"]}</span>'
                f'<span>{"escalated · " if e.get("escalated") else ""}{"at end · " if e.get("at_end") else ""}{e["detail"]}</span>'
                for e in ev) + "</div>" if ev else '<div class="qg-note">No findings yet.</div>')
        T = L["tally"]
        if T:
            with st.container(border=True):
                inj = [x for x in T if x["fault"]]
                caught = sum(x["outcome"] == "caught" for x in inj)
                early = [x["first"] / x["dur"] for x in T if x["first"] is not None and x["verdict"] != "pass" and x["first"] < x["dur"] - 1 / FPS]
                head("This session", f"{len(T)} episodes streamed · {sum(x['verdict'] == 'fail' for x in T)} quarantined · "
                     f"injected faults caught {caught}/{len(inj)}" + (f" · flags raised mid-episode at a median {np.median(early):.0%} of the way in" if early else ""))
                st.dataframe(pd.DataFrame([dict(Episode=x["ep"], Injected=(x["fault"] or "").replace("_", " ") + (f" ({SEVERITIES[x['sev']]})" if x["fault"] else ""),
                                                 Verdict=STATUS[x["verdict"]][0], Outcome=x["outcome"], Codes=", ".join(x["codes"]),
                                                 **{"First flag (s)": x["first"]}, **{"Length (s)": round(x["dur"], 1)}) for x in T[::-1]]),
                             hide_index=True, width="stretch", height=min(36 * len(T) + 38, 260),
                             column_config={"First flag (s)": st.column_config.NumberColumn(format="%.1f")})

    with p_col, st.container(border=True):
        live_panel()
    live_signals()

elif PAGE == "Any dataset":
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
    H = st.session_state.setdefault("hub", dict(repo=None, ep=None, speed=1.0, base=0.0, anchor=None))

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
        H["tbl"] = H.get("tbl", 0) + 1
    cams = insp.cameras
    sn, an = insp.state_names, insp.action_names
    st.html('<div class="qg-sum">' + "".join(f"<span>{k} <b>{v}</b></span>" for k, v in (
        ("LeRobot", insp.version), ("robot", insp.robot or "unknown"), ("fps", insp.fps), ("episodes", f"{len(eps):,}"),
        ("frames", f"{insp.info.get('total_frames', sum(insp.episodes.values())):,}"), ("state", len(sn)), ("action", len(an)),
        ("cameras", len(cams)))) + "</div>")

    @st.cache_data(ttl=600, max_entries=8, show_spinner=False)
    def hub_scan(repo):
        return scan_dataset(hub_inspect(repo), hub_fs())

    try:
        with st.spinner(f"Checking {len(eps):,} episodes of {repo}…"):
            scan = hub_scan(repo)
    except Exception as ex_:
        st.error(hub_error(ex_, f"scan {repo}")); st.stop()

    def _open(idx):
        st.session_state["ds_open"] = idx

    def _back():
        st.session_state["ds_open"] = None
        H["tbl"] += 1                                   # fresh table, so the old row selection does not reopen the episode

    # ---- the dataset verdict: what a user gets on entering a dataset
    flagged = {}                                         # code -> worst status over episodes, and which episodes
    for e in scan.episodes:
        for f in e.findings:
            if f.status != "pass":
                flagged[f.code] = "fail" if "fail" in (f.status, flagged.get(f.code)) else "warn"
    with st.container(border=True):
        st.html(f'<div class="qg-h"><b>{repo}</b></div>')
        cc = st.columns(4)
        with cc[0]: stat("Episodes", f"{scan.total:,}", f"{scan.scanned:,} checked")
        with cc[1]: stat("Passed", f"{scan.passed:,}", f"{scan.passed / max(scan.scanned, 1):.0%} of checked")
        with cc[2]: stat("Warned", f"{scan.warned:,}", "trim or review")
        with cc[3]: stat("Failed", f"{scan.failed:,}", "keep out of training", alert=scan.failed > 0)
        rows = []
        for code in UNIVERSAL_CODES:
            n_ep = len(scan.by_code.get(code, []))
            rows.append(f'<span class="s {flagged.get(code, "pass")}"></span><span>{CODES[code]}</span>'
                        f'<span class="d">{code} · {n_ep} episode{"s" if n_ep != 1 else ""}</span>')
        st.html(f'<div class="qg-ckl">{"".join(rows)}</div>')
        if scan.unreadable:
            why = next(iter(scan.unreadable.values()))
            st.warning(f"{len(scan.unreadable):,} episodes could not be checked ({why}).")

    with st.container(border=True):
        head("Episodes", "worst first · select a row to see what failed beside the video")
        show = st.segmented_control("Show", ["All", "Failed", "Warned"], default="All", key="ds_show", label_visibility="collapsed") or "All"
        order = {"fail": 0, "warn": 1, "pass": 2}
        table = pd.DataFrame([dict(Episode=e.idx, Frames=e.n, Verdict=STATUS[e.verdict][0],
                                   Flags=", ".join(CODES[f.code] for f in e.findings if f.status != "pass") or "–", _o=order[e.verdict])
                              for e in scan.episodes]).sort_values(["_o", "Episode"]).drop(columns="_o")
        if show != "All":
            table = table[table["Verdict"] == STATUS["fail" if show == "Failed" else "warn"][0]]
        sel = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                           key=f"ds_tbl_{H['tbl']}", height=min(35 * len(table) + 38, 330))
        if sel.selection.rows:
            _pick = int(table.iloc[sel.selection.rows[0]]["Episode"])
            if _pick != st.session_state.get("ds_open"):
                st.session_state["ds_open"] = _pick
                st.rerun()

    ep_idx = st.session_state.get("ds_open")
    if ep_idx is not None and ep_idx not in insp.episodes:
        st.warning(f"Episode {ep_idx} is not in this dataset."); ep_idx = None
    if ep_idx is None:
        st.query_params["repo"] = repo
        st.query_params.pop("ep", None)
        st.stop()
    st.query_params["repo"], st.query_params["ep"] = repo, str(ep_idx)
    st.button("← Back to the dataset verdict", on_click=_back)

    if (H["repo"], H["ep"]) != (repo, ep_idx):
        try:
            with st.spinner(f"Streaming episode {ep_idx} from the Hub…"):
                src = hub_episode(repo, ep_idx)
        except TooLarge as ex_:
            st.warning(str(ex_)); st.stop()
        except Exception as ex_:
            st.error(hub_error(ex_, f"read episode {ep_idx}")); st.stop()
        verdict, findings = universal_verdict(src)     # decided on load: no waiting for the episode to play out
        H.update(repo=repo, ep=ep_idx, src=src, verdict=verdict, findings={f.code: f for f in findings}, base=0.0, anchor=None)
    src, findings = H["src"], H["findings"]
    dur = src.n / src.fps

    # stretches of the episode the checks flagged (shaded on the plot, and jump targets for the video)
    ranges, whole = [], None
    for code, f in findings.items():
        if f.status == "pass":
            continue
        if code == "EPIS-004":
            ranges.append((code, f.status, 0, first_move(src) or 0))
        elif code == "EPIS-005":
            ranges.append((code, f.status, src.n - (first_move(src, from_end=True) or 0), src.n))
        elif f.status == "fail":
            whole = code
    if whole:
        ranges.insert(0, (whole, "fail", 0, src.n))

    def _hplay():
        if H["anchor"] is None:
            if clock(H) >= dur - 0.05:
                H["base"] = 0.0
            H["anchor"] = time.time()
        else:
            H["base"], H["anchor"] = clock(H), None

    def _hrestart():
        H.update(base=0.0, anchor=time.time() if H["anchor"] is not None else None)

    def _hjump(frame):
        H.update(base=frame / H["src"].fps, anchor=None)

    def _hspeed():
        if st.session_state["ds_speed"] is None:
            st.session_state["ds_speed"] = f"{H['speed']:g}×"; return
        H["base"] = clock(H)
        H["anchor"] = time.time() if H["anchor"] is not None else None
        H["speed"] = float(st.session_state["ds_speed"].rstrip("×"))

    playing = H["anchor"] is not None
    c = st.columns([.9, .7, .7, 3], vertical_alignment="bottom")
    c[0].segmented_control("Speed", ["1×", "2×", "4×"], key="ds_speed", default=f"{H['speed']:g}×", on_change=_hspeed)
    c[1].button("Pause" if playing else "Play", on_click=_hplay, type="primary", width="stretch")
    c[2].button("Restart", on_click=_hrestart, width="stretch")
    pairable = sn == an
    default_ch = list(range(min(8, len(sn))))
    with c[3].expander("Channels to plot"):
        chans = st.multiselect("Channels to plot", list(range(len(sn))), default=default_ch, key=f"ds_ch_{repo}",
                               format_func=lambda j: sn[j], max_selections=12, label_visibility="collapsed") or default_ch

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
            codec = (insp.info["features"][cam].get("info") or {}).get("video.codec", "unknown codec")
            synced_video(url, vstart, dur, clock(H), H["speed"], playing, aspect=f"{w}/{h}")
            st.html(f'<div class="qg-note">{cam.split(".")[-1]} camera · {codec} · streamed from the Hub, follows the playback clock.</div>')
        else:
            st.html('<div class="qg-note">This dataset has no video.</div>')

    with p_col, st.container(border=True):
        st.html(f'<div class="qg-h"><b>Episode {ep_idx}</b>{pill(H["verdict"])}'
                f'<span>{src.n} frames · {dur:.1f} s · decided on load</span></div>')
        rows = []
        for code, f in findings.items():
            why = WHY.get(code, "")
            rows.append(f'<span class="s {f.status}"></span><span title="{f.detail}">{CODES[code]}</span>'
                        f'<span class="d">{code} · {f.detail}</span>')
        st.html(f'<div class="qg-ckl">{"".join(rows)}</div>')
        for code, f in findings.items():
            if f.status != "pass" and code in WHY:
                st.html(f'<div class="qg-note"><b>{CODES[code]}.</b> {WHY[code]}</div>')
        if ranges:
            st.html('<div class="qg-note" style="margin:8px 0 2px">Flagged stretches · click to jump the video and cursor there</div>')
            rc = st.columns(min(len(ranges), 3))
            for col, (code, status, a, b) in zip(rc, ranges):
                col.button(f"{CODES[code]} · {a / src.fps:.1f}–{b / src.fps:.1f} s", key=f"ds_rng_{code}", on_click=_hjump,
                           args=(a,), width="stretch")
        else:
            st.html('<div class="qg-note">Nothing flagged: no stretch to jump to.</div>')

    def hub_fig(t_cur):
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
        fig.add_vline(x=min(t_cur, dur), line=dict(color=ACCENT, width=1))
        base_layout(fig, 58 * k + 30, hovermode="x unified", hoversubplots="axis", margin=dict(l=0, r=8, t=4, b=0))
        fig.update_xaxes(range=[0, dur], showgrid=True, gridcolor=NEUTRAL, zeroline=False)
        fig.update_xaxes(ticksuffix=" s", row=k, col=1)
        return fig

    @st.fragment(run_every=TICK if playing else None)
    def hub_signals():
        H = st.session_state["hub"]
        t_cur = clock(H)
        if H["anchor"] is not None and t_cur >= H["src"].n / H["src"].fps:     # played out: stop ticking
            H["base"], H["anchor"] = H["src"].n / H["src"].fps, None
            st.rerun(scope="app")
        with st.container(border=True):
            head("Signals", ("state solid · commanded action dotted" if pairable else
                             "state only: state and action channels differ, so the commanded action is not overlaid")
                 + f" · shading = frames the checks flagged · cursor at {min(t_cur, dur):.1f} s")
            st.plotly_chart(hub_fig(t_cur), key="ds_sig", config={"displayModeBar": False}, width="stretch")

    hub_signals()

else:
    table, sevs = BENCH["table"], ["subtle", "moderate", "obvious"]
    cells = [table[f][s] for f in table for s in sevs]
    fp = BENCH["clean_heldout"]
    gaps = []
    for f in table:
        for s in sevs:
            cc = table[f][s]
            if cc["flagged"] < cc["n"] or cc["attributed"] < cc["flagged"]:
                what = (f"caught {cc['flagged']}/{cc['n']}" if cc["flagged"] < cc["n"] else "")
                if cc["attributed"] < cc["flagged"]:
                    what += ("; " if what else "") + f"only {cc['attributed']} under the expected code"
                gaps.append(f"<li><b>{f.replace('_', ' ')}</b>, {s} ({cc['param']} {cc['unit']}): {what}</li>")
    c = st.columns(4)
    with c[0]: stat("Fault cells", len(cells), f"{len(table)} faults × 3 severities, {BENCH['per_cell']} each")
    with c[1]: stat("Fully caught", sum(x["flagged"] == x["n"] and x["attributed"] == x["n"] for x in cells), "detected, right code")
    with c[2]: stat("Gaps", len(gaps), "cells below full marks")
    with c[3]: stat("False positives", f"{fp['flagged']}/{fp['n']}", f"95% CI {fp['flagged_ci'][0]:.0%}–{fp['flagged_ci'][1]:.0%}")

    left, right = st.columns([3, 2])
    with left, st.container(border=True):
        head("Detection by fault and severity", f"injected into copies of clean held-out episodes · shade = detected of {BENCH['per_cell']} · ⚑ wrong code")
        st.plotly_chart(bench_heatmap(), key="bench", config={"displayModeBar": False}, width="stretch")
    with right:
        with st.container(border=True):
            head("Where the gate falls short")
            st.html(f'<ul class="qg-miss">{"".join(gaps)}</ul>')
        with st.container(border=True):
            head("False positives on clean data", "held-out episodes 25–49, nothing injected")
            fl = " and ".join("Episode {} ({})".format(x["episode"], ", ".join(x["codes"])) for x in fp["episodes"] if x["verdict"] != "pass")
            st.html(f'<div class="qg-v">{fp["flagged"]}<small>of {fp["n"]}</small></div><div class="qg-note">{fl or "None"}.</div>')

    with st.container(border=True):
        head("Fault lab", "inject a fault into a copy of a real episode and run the same gate live · the original is never modified")
        c1, c2, c3, c4 = st.columns(4)
        pool = [i for i, e in EPS.items() if e["verdict"] == "pass"]
        lab_ep = c1.selectbox("Episode (clean ones)", pool, index=pool.index(38) if 38 in pool else 0)
        fault = c2.selectbox("Fault", list(FAULTS), format_func=lambda f: f.replace("_", " "))
        sev = c3.select_slider("Severity", SEVERITIES, value="moderate")
        seed = c4.number_input("Seed", 0, 999, 0)
        _, params, expected, unit_label = FAULTS[fault]
        si = SEVERITIES.index(sev)
        base = episode_obj(lab_ep)
        bad = inject(base, fault, si, dict(TH), np.random.default_rng(int(seed)))
        verdict, fs = run_checks(bad, TH)
        fd = [f.to_dict() for f in fs]
        got = {f["code"] for f in fd if f["status"] != "pass"}
        if verdict == "pass":
            outcome = f"Missed. Expected {', '.join(sorted(expected))}. This is a real limit of the gate at this severity."
        elif got & expected:
            outcome = f"Caught under {', '.join(sorted(got & expected))}, the expected code."
        else:
            outcome = f"Flagged, but under {', '.join(sorted(got))} instead of {', '.join(sorted(expected))}."
        st.html(f'<div class="qg-outcome">{pill(verdict)} <b>{fault.replace("_", " ")}</b> at {params[si]} {unit_label}. {outcome}</div>')
        lab_flags = evidence(bad, fd, TH)
        lcol, rcol = st.columns([3, 1])
        with lcol:
            st.plotly_chart(signal_fig(bad, lab_flags, ref=base, row_h=64), key="lab", config={"displayModeBar": False}, width="stretch")
            st.html('<div class="qg-note">Grey = original episode · blue = injected copy · shading = frames the gate flagged.</div>')
        with rcol:
            st.html("".join(f'<div class="qg-f {f["status"]}"><span class="qg-code">{f["code"]}</span> {CODES[f["code"]]}<p>{f["text"]}</p></div>'
                            for f in lab_flags) or '<div class="qg-note">No findings.</div>')
            st.html(checklist(fd))
