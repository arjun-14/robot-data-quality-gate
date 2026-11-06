"""qgate dashboard ("Board" layout): fleet health, check status history, threshold headroom, episode inspector;
Gate validation page with the benchmark and a live fault lab.

Reads precomputed files in app/data (see scripts/build_app_data.py) and bench/results/results.json.
Run:  streamlit run app/app.py
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from numpy.lib.stride_tricks import sliding_window_view
from plotly.subplots import make_subplots

from bench.inject import FAULTS, SEVERITIES, inject
from qgate.checks import CODES, run_checks, velocity
from qgate.dataset import Episode

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
.st-key-nav [data-testid="stButtonGroup"] {{ justify-content:flex-end; }}
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
    codes = list(CODES)
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


def signal_fig(ep: Episode, flags, ref: Episode | None = None, row_h=78):
    k = len(JOINTS)
    fig = make_subplots(rows=k, cols=1, shared_xaxes=True, vertical_spacing=0.012)
    flagged = {f["joint"] for f in flags if f["joint"] is not None}
    t = np.arange(ep.n) / FPS
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


# ------------------------------------------------------------------ episode selection (shared by all clickable charts)
# The page and episode live in the URL (?page=validation&ep=31) so links and the theme toggle's reload keep your place.
PAGES = {"episodes": "Episodes", "validation": "Gate validation"}
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
top_l, top_nav, top_theme = st.columns([6, 3, .45], vertical_alignment="center")
with top_l:
    st.html('<div class="qg-brand"><i></i>qgate</div>')
with top_nav:
    st.segmented_control("Page", list(PAGES.values()), key="nav", on_change=_nav_changed, label_visibility="collapsed")
with top_theme:
    # Streamlit has no API to switch theme from app code. This writes the same browser setting as its own
    # Settings menu ("stActiveTheme-<path>-v2" = "Light" | "Dark" | "System", Streamlit >= 1.50) and reloads.
    # It relies on a Streamlit internal: if a Streamlit upgrade breaks it, use the menu's Settings > Theme instead.
    st.html("""<button id="qg-theme" type="button" title="Switch light / dark" aria-label="Switch light or dark theme"
      style="font:16px/1 sans-serif;width:38px;height:38px;border-radius:8px;cursor:pointer;background:transparent;
             color:inherit;border:1px solid rgba(128,140,155,.45)">◐</button>
    <script>
    (() => {
      const b = document.getElementById("qg-theme"); if (!b || b.dataset.wired) return; b.dataset.wired = "1";
      b.addEventListener("click", () => {
        const key = "stActiveTheme-" + window.location.pathname + "-v2";
        let cur = null; try { cur = JSON.parse(localStorage.getItem(key)); } catch (e) {}
        const dark = cur === "Dark" || ((cur === null || cur === "System") && matchMedia("(prefers-color-scheme: dark)").matches);
        try { localStorage.setItem(key, JSON.stringify(dark ? "Light" : "Dark")); } catch (e) {}
        location.reload();
      });
    })();
    </script>""", unsafe_allow_javascript=True)

PAGE = st.session_state["nav"] or "Episodes"
st.query_params["page"] = next(k for k, v in PAGES.items() if v == PAGE)
if PAGE == "Episodes":
    st.query_params["ep"] = str(st.session_state["ep_sel"])
else:
    st.query_params.pop("ep", None)

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
