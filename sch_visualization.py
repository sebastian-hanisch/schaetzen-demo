"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sch_constants as C
import sch_estimators as E
from sch_evaluation import Analysis, de, signed

SHORT = {
    "Naiver Vergleich": "naiv", "Regression (linear)": "Regr. linear", "Regression (Wechselwirkung)": "Regr. Wechselw.", "Regression (quadratisch)": "Regr. quadr.",
    "Gewichten (IPW)": "IPW", "Gewichten (normiert)": "Gewichten", "Gewichten (gekappt)": "gekappt", "Schichten": "Schichten", "Matching (ATE)": "Matching ATE",
    "Matching (ATT)": "Matching ATT",
}
LINES = ("Regression (linear)", "Regression (quadratisch)", "Gewichten (normiert)", "Gewichten (gekappt)", "Schichten", "Matching (ATE)")
PALETTE = ("#4c78a8", "#8a8a8a", "#D68A2E", "#2F6B65", "#c0392b", "#7b5ea7")


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.22), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_diagram() -> go.Figure:
    """Das feste Diagramm: Schwankung X wirkt auf die Regelwahl T und auf den Servicegrad Y, T wirkt auf Y."""
    pos = {"X": (0.5, 1.0), "T": (0.0, 0.0), "Y": (1.0, 0.0)}
    names = {"X": "Schwankung<br>(Störfaktor X)", "T": "Regel B<br>(Behandlung T)", "Y": "Servicegrad<br>(Ergebnis Y)"}
    fig = go.Figure()
    for a, b, label, color, width in (("X", "T", "Zuteilung", C.COLOR_NEUTRAL, 2), ("X", "Y", "β X + Krümmung", C.COLOR_NEUTRAL, 2), ("T", "Y", "τ(X)", C.COLOR_GOOD, 4)):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        dx, dy = x1 - x0, y1 - y0
        ln = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / ln, dy / ln
        fig.add_annotation(x=x1 - ux * 0.17, y=y1 - uy * 0.17, ax=x0 + ux * 0.17, ay=y0 + uy * 0.17, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3,
                           arrowsize=1.3, arrowwidth=width, arrowcolor=color, text="")
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2 + 0.08, text=label, showarrow=False, font=dict(size=11, color="#555555"), bgcolor="rgba(255,255,255,0.75)")
    fig.add_trace(go.Scatter(x=[pos[k][0] for k in pos], y=[pos[k][1] for k in pos], mode="markers+text", text=[names[k] for k in pos], textposition="middle center",
                             textfont=dict(size=10), marker=dict(size=70, color="rgba(47,107,101,0.12)", line=dict(width=[2, 4, 4], color=["#2F6B65", "#1b1b1b", "#1b1b1b"])),
                             hoverinfo="skip", showlegend=False))
    fig.update_xaxes(visible=False, range=[-0.3, 1.3])
    fig.update_yaxes(visible=False, range=[-0.35, 1.3])
    return _fixed(fig, 280, showlegend=False)


def build_estimates_chart(a: Analysis) -> go.Figure:
    """Schätzung mit Bootstrap-Intervall je Schätzer; senkrechte Linie: ATE; Raute: die eigene Zielgröße (ATT, Stützung), wenn sie vom ATE abweicht."""
    names = [r.name for r in a.rows]
    fig = go.Figure()
    for r in a.rows:
        color = C.COLOR_GOOD if r.covers else C.COLOR_BAD
        fig.add_trace(go.Scatter(x=[r.estimate], y=[SHORT[r.name]], mode="markers", marker=dict(color=color, size=10), showlegend=False,
                                 error_x=dict(type="data", symmetric=False, array=[r.hi - r.estimate], arrayminus=[r.estimate - r.lo], color=color, thickness=2),
                                 hovertemplate=f"{r.name}: {de(r.estimate, 3)} [{de(r.lo, 2)}; {de(r.hi, 2)}]<extra></extra>"))
    own = [(SHORT[r.name], r.truth) for r in a.rows if abs(r.truth - a.ate) > 1e-9]
    if own:
        fig.add_trace(go.Scatter(x=[v for _, v in own], y=[k for k, _ in own], mode="markers", marker=dict(color=C.COLOR_TRUTH, size=11, symbol="diamond"),
                                 name="eigene Zielgröße"))
    fig.add_vline(x=a.ate, line=dict(color=C.COLOR_TRUTH, width=2), annotation_text=f"ATE {de(a.ate, 2)}", annotation_position="top")
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=[SHORT[n] for n in names])
    fig.update_xaxes(title="geschätzte Wirkung der Regel B")
    return _fixed(fig, 430, showlegend=bool(own))


def build_overlap_histogram(a: Analysis) -> go.Figure:
    bins = dict(start=0.0, end=1.0, size=1.0 / C.HIST_BINS)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=a.e[a.t == 0], xbins=bins, name="Kontrollen (Regel A)", marker_color=C.COLOR_CONTROL, opacity=0.7))
    fig.add_trace(go.Histogram(x=a.e[a.t == 1], xbins=bins, name="Behandelte (Regel B)", marker_color=C.COLOR_TREATED, opacity=0.7))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="geschätzter Neigungswert e(X)", range=[0, 1])
    fig.update_yaxes(title="Anzahl Standorte")
    return _fixed(fig, 300)


# ----------------------------------------------------------------------------- Experiment 1

def build_scenarios_table(res: dict) -> pd.DataFrame:
    """Verzerrung gegen die eigene Zielgröße je Szenario (Zeilen) und Schätzer (Spalten), als Text."""
    out = []
    for name, r in res.items():
        row = {"Szenario": name}
        for est in E.NAMES:
            row[SHORT[est]] = signed(r["estimators"][est]["bias"], 2)
        row["ESS"] = de(r["ess"], 0)
        row["außerhalb"] = f"{de(100 * r['outside'], 0)} %"
        out.append(row)
    return pd.DataFrame(out)


def style_scenarios_table(df: pd.DataFrame, res: dict):
    def cell(v):
        try:
            x = abs(float(v.replace("−", "-").replace("+", "").replace(",", ".")))
        except ValueError:
            return ""
        if x >= 0.25:
            return "background-color: rgba(192,57,43,0.25)"
        if x >= 0.1:
            return "background-color: rgba(214,138,46,0.25)"
        return ""

    cols = [SHORT[n] for n in E.NAMES]
    styler = df.style
    return (styler.map if hasattr(styler, "map") else styler.applymap)(cell, subset=cols)


def build_scenario_chart(r: dict) -> go.Figure:
    names = [n for n in E.NAMES if n != "Naiver Vergleich"]
    bias = [r["estimators"][n]["bias"] for n in names]
    sd = [r["estimators"][n]["sd"] for n in names]
    colors = [C.COLOR_BAD if abs(b) >= 0.25 else (C.COLOR_TRUTH if abs(b) >= 0.1 else C.COLOR_GOOD) for b in bias]
    fig = go.Figure(go.Bar(x=[SHORT[n] for n in names], y=bias, marker_color=colors, error_y=dict(type="data", array=sd, visible=True),
                           text=[signed(b, 2) for b in bias], textposition="outside"))
    fig.add_hline(y=0.0, line=dict(color="#888888", width=1))
    fig.update_yaxes(title="Verzerrung (Balken), Streuung (Linie)")
    fig.update_xaxes(tickangle=-25)
    return _fixed(fig, 400, showlegend=False)


# ----------------------------------------------------------------------------- Experiment 2

def build_overlap_chart(rows: list) -> go.Figure:
    gam = [r["gamma"] for r in rows]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.65, 0.35], subplot_titles=("Verzerrung gegen den ATE", "Effektive Stichprobe"))
    for color, name in zip(PALETTE, LINES):
        fig.add_trace(go.Scatter(x=gam, y=[r["estimators"][name]["bias_ate"] for r in rows], mode="lines+markers", name=SHORT[name], line=dict(color=color, width=2)), row=1, col=1)
    fig.add_trace(go.Bar(x=gam, y=[r["ess"] for r in rows], marker_color=C.COLOR_NEUTRAL, name="effektive Stichprobe", showlegend=False), row=2, col=1)
    fig.update_xaxes(title_text="Zuteilungsstärke γ", row=2, col=1)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.15))
    return fig


# ----------------------------------------------------------------------------- Experiment 3

def build_strata_chart(rows: list) -> go.Figure:
    ks = [str(r["k"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=[r["bias"] for r in rows], mode="lines+markers", name="Verzerrung", line=dict(color=C.COLOR_BAD, width=2.5)))
    fig.add_trace(go.Scatter(x=ks, y=[r["sd"] for r in rows], mode="lines+markers", name="Streuung", line=dict(color=C.COLOR_TREATED, width=2.5)))
    fig.update_xaxes(title="Anzahl Schichten K", type="category")
    fig.update_yaxes(title="Verzerrung / Streuung", rangemode="tozero")
    return _fixed(fig, 340)


def build_trim_chart(res: dict) -> go.Figure:
    fig = go.Figure()
    for (label, rows), color in zip(res.items(), (C.COLOR_GOOD, C.COLOR_BAD)):
        c = [r["trim"] for r in rows]
        fig.add_trace(go.Scatter(x=c, y=[r["mean"] for r in rows], mode="lines+markers", name=f"Schätzung, {label}", line=dict(color=color, width=2.5)))
        fig.add_trace(go.Scatter(x=c, y=[r["target"] for r in rows], mode="lines", name=f"Zielgröße auf der Stützung, {label}", line=dict(color=color, width=1.5, dash="dot")))
    fig.add_hline(y=C.TAU0, line=dict(color=C.COLOR_TRUTH, width=2), annotation_text=f"ATE {de(C.TAU0, 2)}", annotation_position="top left")
    fig.update_xaxes(title="Kappungsgrenze c (nur Einheiten mit c < e < 1 − c)")
    fig.update_yaxes(title="Wirkung")
    return _fixed(fig, 380)


# ----------------------------------------------------------------------------- Experiment 4

def build_coverage_chart(res: dict) -> go.Figure:
    from sch_evaluation import COVER_NAMES

    fig = go.Figure()
    for (label, color) in (("Grundfall", C.COLOR_GOOD), ("knappe Überlappung", C.COLOR_BAD)):
        fig.add_trace(go.Bar(x=[SHORT[n] for n in COVER_NAMES], y=[100 * res[label][n]["cover"] for n in COVER_NAMES], name=label, marker_color=color,
                             text=[de(100 * res[label][n]["cover"], 0) + " %" for n in COVER_NAMES], textposition="outside"))
    fig.add_hline(y=95, line=dict(color="#888888", dash="dot"), annotation_text="versprochen: 95 %", annotation_position="top left")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Deckung der wahren Wirkung (%)", range=[0, 112])
    return _fixed(fig, 380)


def build_estimates_table(a: Analysis) -> pd.DataFrame:
    out = []
    for r in a.rows:
        out.append({
            "Schätzer": r.name,
            "Schätzgröße": r.estimand,
            "Schätzung": signed(r.estimate, 3),
            "95-%-Intervall (Bootstrap)": f"{de(r.lo, 2)} bis {de(r.hi, 2)}",
            "Zielgröße": de(r.truth, 3),
            "Fehler": signed(r.error, 3),
            "Deckt": "ja" if r.covers else "nein",
            "Grenzfall (n → ∞)": signed(r.limit, 3) if r.limit is not None else "–",
        })
    return pd.DataFrame(out)


def style_estimates_table(df: pd.DataFrame):
    def row_style(row):
        return ["background-color: rgba(192,57,43,0.15)" if row["Deckt"] == "nein" else ""] * len(row)

    return df.style.apply(row_style, axis=1)
