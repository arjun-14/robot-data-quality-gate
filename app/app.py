"""qgate dashboard: fleet view, episode inspector, live fault lab, benchmark.

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
from plotly.subplots import make_subplots

from bench.inject import FAULTS, SEVERITIES, inject
from qgate.checks import CODES, run_checks
from qgate.dataset import Episode

st.set_page_config(page_title="qgate", layout="wide")

STATUS_COLOR = {"pass": "#1f8a4c", "warn": "#d99a1e", "fail": "#c53030"}
STATUS_MARK = {"pass": "✓", "warn": "!", "fail": "✗"}
CODE_COLOR = {"TIME-001": "#8b5cf6", "TIME-002": "#a855f7", "TIME-003": "#c084fc", "JOINT-001": "#e11d48", "JOINT-002": "#ea580c",
              "JOINT-003": "#f59e0b", "JOINT-004": "#0891b2", "GRIP-001": "#65a30d", "XSIG-001": "#2563eb", "EPIS-001": "#78716c", "EPIS-002": "#a16207"}
SHADE_WIDTH = {"JOINT-004": 60, "JOINT-003": 2, "TIME-001": 2}


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


def episode_obj(idx) -> Episode:
    d = SIG[SIG.episode == idx]
    return Episode(idx, float(META["fps"]), d[[f"s{j}" for j in range(6)]].to_numpy(float), d[[f"a{j}" for j in range(6)]].to_numpy(float),
                   d.timestamp.to_numpy(float), d.frame_index.to_numpy(np.int64), dict(META["video_frames"][str(idx)]))


def signal_plot(ep: Episode, findings, ref: Episode | None = None, height=640):
    k = len(JOINTS)
    fig = make_subplots(rows=k + 1, cols=1, shared_xaxes=True, vertical_spacing=0.015, row_heights=[1] * k + [0.6])
    x = np.arange(ep.n)
    for j, name in enumerate(JOINTS):
        if ref is not None:
            fig.add_trace(go.Scatter(x=np.arange(ref.n), y=ref.state[:, j], line=dict(width=1, color="#9aa4af"), showlegend=False, hoverinfo="skip"), row=j + 1, col=1)
        fig.add_trace(go.Scatter(x=x, y=ep.state[:, j], mode="lines", line=dict(width=1.3, color="#c53030" if ref is not None else "#2b59c3"), showlegend=False, name=name), row=j + 1, col=1)
        fig.update_yaxes(title_text=name, title_font_size=10, row=j + 1, col=1)
    dt = np.diff(ep.ts) * 1000
    fig.add_trace(go.Bar(x=x[1:], y=dt, marker_color="#7a8794", showlegend=False, name="dt"), row=k + 1, col=1)
    fig.update_yaxes(title_text="dt ms", title_font_size=10, range=[0, max(40, float(np.nanmax(dt)) * 1.1)], row=k + 1, col=1)
    for f in findings:
        if f["status"] == "pass":
            continue
        for fr in f["frames"][:40]:
            fig.add_vrect(x0=fr, x1=fr + SHADE_WIDTH.get(f["code"], 1), fillcolor=CODE_COLOR[f["code"]], opacity=0.22, line_width=0, layer="below")
    fig.update_xaxes(title_text="frame", row=k + 1, col=1)
    fig.update_layout(height=height, margin=dict(l=60, r=10, t=10, b=30), bargap=0)
    return fig


def findings_table(findings):
    return pd.DataFrame([dict(code=f["code"], check=f["name"], verdict=f"{STATUS_MARK[f['status']]} {f['status']}", detail=f["detail"]) for f in findings])


def verdict_badge(v):
    return f":{ {'pass': 'green', 'warn': 'orange', 'fail': 'red'}[v] }[**{v.upper()}**]"


st.title("qgate · robot data quality gate")
st.caption(f"{META['dataset']} · {len(EPS)} episodes · {META['fps']} fps · thresholds calibrated on episodes 0–24")
tab_fleet, tab_lab, tab_bench, tab_about = st.tabs(["Fleet", "Fault lab", "Benchmark", "Thresholds & codes"])

# ------------------------------------------------------------------ Fleet
with tab_fleet:
    counts = pd.Series([e["verdict"] for e in EPS.values()]).value_counts()
    c = st.columns(4)
    c[0].metric("episodes", len(EPS)); c[1].metric("pass", int(counts.get("pass", 0)))
    c[2].metric("warn (review)", int(counts.get("warn", 0))); c[3].metric("fail (quarantine)", int(counts.get("fail", 0)))

    left, right = st.columns([2, 1])
    sessions = sorted({i // 10 for i in EPS})
    z, text, cd = [], [], []
    for s in sessions:
        ids = [i for i in EPS if i // 10 == s]
        z.append([{"pass": 0, "warn": 1, "fail": 2}[EPS[i]["verdict"]] for i in ids])
        text.append([f"{i}<br>{STATUS_MARK[EPS[i]['verdict']]}" for i in ids]); cd.append(ids)
    heat = go.Figure(go.Heatmap(z=z, text=text, texttemplate="%{text}", customdata=cd, xgap=3, ygap=3, showscale=False, zmin=0, zmax=2,
                                colorscale=[[0, STATUS_COLOR["pass"]], [0.5, STATUS_COLOR["warn"]], [1, STATUS_COLOR["fail"]]],
                                hovertemplate="episode %{customdata}<extra></extra>"))
    heat.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed", tickvals=list(range(len(sessions))), ticktext=[f"session {s}" for s in sessions]), xaxis=dict(visible=False))
    with left:
        st.subheader("Episodes by session"); st.caption("session = block of 10 consecutive episodes · click a cell to inspect")
        ev = st.plotly_chart(heat, key="grid", on_select="rerun", selection_mode="points", width="stretch")
    with right:
        st.subheader("Fail rate by session")
        tr = pd.DataFrame({"session": [f"s{s}" for s in sessions],
                           "fail": [np.mean([EPS[i]["verdict"] == "fail" for i in EPS if i // 10 == s]) for s in sessions],
                           "warn": [np.mean([EPS[i]["verdict"] == "warn" for i in EPS if i // 10 == s]) for s in sessions]})
        bar = go.Figure([go.Bar(x=tr.session, y=tr.fail, name="fail", marker_color=STATUS_COLOR["fail"]), go.Bar(x=tr.session, y=tr.warn, name="warn", marker_color=STATUS_COLOR["warn"])])
        bar.update_layout(barmode="stack", height=270, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(tickformat=".0%", range=[0, 1]), legend=dict(orientation="h"))
        st.plotly_chart(bar, width="stretch")

    st.subheader("Findings by code")
    rows = []
    for code, name in CODES.items():
        w = [i for i, e in EPS.items() if any(f["code"] == code and f["status"] == "warn" for f in e["checks"])]
        f_ = [i for i, e in EPS.items() if any(f["code"] == code and f["status"] == "fail" for f in e["checks"])]
        rows.append(dict(code=code, check=name, warn=len(w), fail=len(f_), episodes=", ".join(map(str, sorted(w + f_))) or "-"))
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # click on the grid sets the inspected episode (once per new click)
    pts = ev.selection.points if ev and ev.selection else []
    if pts and pts[0].get("customdata") is not None and st.session_state.get("last_click") != pts[0]["customdata"]:
        st.session_state["last_click"] = pts[0]["customdata"]; st.session_state["ep_sel"] = int(pts[0]["customdata"])
    st.subheader("Episode inspector")
    sel = st.selectbox("episode", sorted(EPS), key="ep_sel", format_func=lambda i: f"episode {i} · {EPS[i]['verdict']}")
    e = EPS[sel]
    st.markdown(f"{verdict_badge(e['verdict'])} · {e['n']} frames")
    st.dataframe(findings_table(e["checks"]), hide_index=True, width="stretch")
    st.plotly_chart(signal_plot(episode_obj(sel), e["checks"]), width="stretch")
    if sel in (19, 31, 36):
        st.info("Natural finding in the real data (nothing injected). Range and stuck-joint thresholds come from only 25 calibration episodes, so some false alarms are expected; see the Benchmark tab.")

# ------------------------------------------------------------------ Fault lab
with tab_lab:
    st.markdown("Inject a fault into a **copy** of a real episode and run the same gate live. The original is never modified.")
    c1, c2, c3, c4 = st.columns(4)
    pool = [i for i, e in EPS.items() if e["verdict"] == "pass"]
    lab_ep = c1.selectbox("episode (clean ones)", pool, index=pool.index(38) if 38 in pool else 0)
    fault = c2.selectbox("fault", list(FAULTS), format_func=lambda f: f.replace("_", " "))
    sev = c3.select_slider("severity", SEVERITIES, value="moderate")
    seed = c4.number_input("seed", 0, 999, 0)
    fn, params, expected, unit = FAULTS[fault]
    si = SEVERITIES.index(sev)
    base = episode_obj(lab_ep); th = dict(TH)
    bad = inject(base, fault, si, th, np.random.default_rng(int(seed)))
    verdict, fs = run_checks(bad, th); fd = [f.to_dict() for f in fs]
    got = {f["code"] for f in fd if f["status"] != "pass"}
    st.markdown(f"**{fault.replace('_', ' ')}** at `{params[si]}` {unit} → {verdict_badge(verdict)}")
    if verdict == "pass":
        st.warning(f"Missed. Expected one of {sorted(expected)}. This is a real limit of the gate at this severity.")
    elif got & expected:
        st.success(f"Caught and attributed to {sorted(got & expected)} (expected {sorted(expected)}).")
    else:
        st.warning(f"Flagged, but under {sorted(got)} instead of {sorted(expected)}.")
    st.dataframe(findings_table(fd), hide_index=True, width="stretch")
    st.plotly_chart(signal_plot(bad, fd, ref=base), width="stretch")
    st.caption("Grey = original episode, red = injected copy. Shading marks frames flagged by each taxonomy code.")

# ------------------------------------------------------------------ Benchmark
with tab_bench:
    o = BENCH; fp = o["clean_heldout"]
    st.markdown(f"Calibrated on episodes {o['calibration'][0]}–{o['calibration'][1]-1}; faults injected into copies of held-out episodes that pass clean; "
                f"**{o['per_cell']} episodes per cell**, seed {o['seed']}. *Detected* = the injected copy was not a pass. *Attributed* = an expected taxonomy code was among the findings.")
    faults = list(o["table"]); sevs = ["subtle", "moderate", "obvious"]
    zdet = [[o["table"][f][s]["flagged"] / o["table"][f][s]["n"] for s in sevs] for f in faults]
    txt = [[f"{o['table'][f][s]['flagged']}/{o['table'][f][s]['n']} detected<br>{o['table'][f][s]['attributed']} attributed" for s in sevs] for f in faults]
    hm = go.Figure(go.Heatmap(z=zdet, x=sevs, y=[f.replace("_", " ") for f in faults], text=txt, texttemplate="%{text}", zmin=0, zmax=1, xgap=2, ygap=2,
                              colorscale=[[0, STATUS_COLOR["fail"]], [0.5, STATUS_COLOR["warn"]], [1, STATUS_COLOR["pass"]]], showscale=False))
    hm.update_layout(height=620, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"), xaxis=dict(side="top"))
    st.plotly_chart(hm, width="stretch")
    st.subheader("False positives on clean held-out episodes")
    st.write(f"Flagged (warn or fail): **{fp['flagged']}/{fp['n']}** (95% CI {fp['flagged_ci'][0]:.0%}–{fp['flagged_ci'][1]:.0%}) · "
             f"Failed (quarantined): **{fp['failed']}/{fp['n']}** (95% CI {fp['failed_ci'][0]:.0%}–{fp['failed_ci'][1]:.0%}). "
             f"Flagged: " + ", ".join(f"episode {c['episode']} ({', '.join(c['codes'])})" for c in fp["episodes"] if c["verdict"] != "pass"))
    st.caption("Only 25 calibration and 25 evaluation episodes: the confidence intervals are wide on purpose. Timing faults are caught at every severity because the dataset's timestamps sit exactly on the 1/fps grid; the 1e-4 s tolerance leaves no room for subtle timing faults.")

# ------------------------------------------------------------------ About
with tab_about:
    st.subheader("Where each threshold comes from")
    prov = pd.DataFrame([dict(threshold=k, origin=v[0], how=v[1]) for k, v in TH["provenance"].items()])
    st.dataframe(prov, hide_index=True, width="stretch")
    st.subheader("Taxonomy codes")
    st.dataframe(pd.DataFrame([dict(code=c, check=n) for c, n in CODES.items()]), hide_index=True, width="stretch")
