"""Auswertung: Einstellungen, eine Stichprobe mit Bootstrap-Intervallen, die vier Experimente über feste Ströme und die Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import sch_constants as C
import sch_estimators as E
import sch_world as W
from kausal_generator import SplitMix64


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def signed(x: float, digits: int = 2) -> str:
    return ("+" if x >= 0 else "−") + de(abs(x), digits)


@dataclass(frozen=True)
class Settings:
    gamma: float = C.DEFAULT_GAMMA
    offset: float = 0.0
    delta: float = 0.0
    curve: float = 0.0
    tau1: float = 0.0
    k: int = C.DEFAULT_K
    trim: float = C.DEFAULT_TRIM
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    ps_quad: bool = False

    @property
    def world(self) -> W.World:
        return W.World(self.gamma, self.offset, self.delta, self.curve, self.tau1)


def target(w: W.World, name: str, trim: float) -> float:
    """Wahre Zielgröße des Schätzers: ATE, ATT (Matching ATT) oder ATE auf der Stützung (gekappt)."""
    if name == "Matching (ATT)":
        return W.att(w)
    if name == "Gewichten (gekappt)":
        return W.trimmed_ate(w, trim)
    return W.ate(w)


REGRESSION_KIND = {"Regression (linear)": "linear", "Regression (Wechselwirkung)": "interaction", "Regression (quadratisch)": "quadratic"}


@dataclass
class Row:
    name: str
    estimand: str
    estimate: float
    lo: float
    hi: float
    truth: float          # Wahrheit der Zielgröße dieses Schätzers
    error: float          # Schätzung minus Wahrheit der Zielgröße
    error_ate: float      # Schätzung minus ATE
    covers: bool
    limit: float | None   # Regression: Koeffizient bei n → ∞ (None sonst)


@dataclass
class Analysis:
    settings: Settings
    rows: list
    ate: float
    att: float
    share: float          # Anteil Behandelter in der Welt
    treated_n: int
    e: np.ndarray         # geschätzter Neigungswert je Einheit
    t: np.ndarray
    ess: float
    outside: float


def analyse(s: Settings) -> Analysis:
    w = s.world
    x, t, y = (a[0] for a in W.simulate(w, s.n, SplitMix64(s.seed)))
    est = E.estimate_all(x, t, y, s.k, s.trim, s.ps_quad)
    idx = E.bootstrap_indices(SplitMix64(s.seed + 1), s.n, C.BOOT_B)
    ci = E.bootstrap_all(x, t, y, idx, s.k, s.trim, s.ps_quad)
    ate = W.ate(w)
    rows = []
    for name in E.NAMES:
        tr = target(w, name, s.trim)
        lo, hi = ci[name]
        rows.append(Row(name, E.ESTIMAND[name], est[name], lo, hi, tr, est[name] - tr, est[name] - ate, bool(lo <= tr <= hi),
                        W.regression_limit(w, REGRESSION_KIND[name]) if name in REGRESSION_KIND else None))
    e = E.fit_propensity(x, t, s.ps_quad)
    return Analysis(s, rows, ate, W.att(w), W.treated_share(w), int(t.sum()), e, t, E.effective_sample_size(t, e), E.outside_support(e))


# ----------------------------------------------------------------------------- Experiment 1: Szenarien

def run_scenario(w: W.World, n: int, reps: int, seed: int, k: int = C.DEFAULT_K, trim: float = C.DEFAULT_TRIM, quad: bool = False, only=None) -> dict:
    """`reps` feste Stichproben: je Schätzer Mittel, Verzerrung gegen die eigene Zielgröße und gegen den ATE, SD, RMSE; dazu ESS und Anteil außerhalb der Stützung."""
    x, t, y = W.simulate(w, n, SplitMix64(seed), reps)
    cols = {name: [] for name in (E.NAMES if only is None else only)}
    ess, outside = [], []
    for r in range(reps):
        for name, v in E.estimate_all(x[r], t[r], y[r], k, trim, quad, only).items():
            cols[name].append(v)
        e = E.fit_propensity(x[r], t[r], quad)
        ess.append(E.effective_sample_size(t[r], e))
        outside.append(E.outside_support(e))
    ate = W.ate(w)
    est = {}
    for name, vals in cols.items():
        v = np.asarray(vals)
        v = v[np.isfinite(v)]
        tr = target(w, name, trim)
        est[name] = {"mean": float(v.mean()), "bias": float(v.mean() - tr), "bias_ate": float(v.mean() - ate), "sd": float(v.std()),
                     "rmse": float(np.sqrt(((v - tr) ** 2).mean())), "target": tr}
    return {"estimators": est, "ess": float(np.mean(ess)), "outside": float(np.mean(outside)), "ate": ate, "att": W.att(w), "n": n, "reps": reps}


def scenarios_experiment(n: int = C.EXP_N, reps: int = C.EXP_REPS) -> dict:
    return {name: run_scenario(W.World(**kw), n, reps, C.SEED_BASE + i) for i, (name, kw) in enumerate(C.SCENARIOS.items())}


# ----------------------------------------------------------------------------- Experiment 2: Überlappung

def overlap_experiment(gammas=C.EXP_OVERLAP_GAMMAS, reps: int = C.EXP_OVERLAP_REPS, n: int = C.EXP_N) -> list[dict]:
    rows = []
    for i, g in enumerate(gammas):
        res = run_scenario(W.World(gamma=g), n, reps, C.SEED_BASE + 100 + i)
        res["gamma"] = g
        rows.append(res)
    return rows


# ----------------------------------------------------------------------------- Experiment 3: Schichtenzahl und Kappung

def strata_experiment(ks=C.EXP_K_VALUES, reps: int = C.EXP_REPS, n: int = C.EXP_N) -> list[dict]:
    """Grundfall: je K Verzerrung und SD der Schichtung (dieselben Stichproben für alle K)."""
    w = W.World(gamma=C.DEFAULT_GAMMA)
    x, t, y = W.simulate(w, n, SplitMix64(C.SEED_BASE + 200), reps)
    es = [E.fit_propensity(x[r], t[r]) for r in range(reps)]
    out = []
    for k in ks:
        v = np.array([E.stratified(E.strata_labels(es[r], k), t[r], y[r]) for r in range(reps)])
        out.append({"k": k, "mean": float(v.mean()), "bias": float(v.mean() - W.ate(w)), "sd": float(v.std()), "rmse": float(np.sqrt(((v - W.ate(w)) ** 2).mean()))})
    return out


def trim_experiment(trims=C.EXP_TRIM_VALUES, reps: int = C.EXP_REPS, n: int = C.EXP_N) -> dict:
    """Gekapptes Gewichten im Grundfall und bei heterogener Wirkung mit Versatz (S3b): Mittel gegen ATE und gegen die Zielgröße auf der Stützung."""
    out = {}
    for label, w, seed in (("Grundfall", W.World(gamma=0.8), 300), ("heterogen, Versatz", W.World(gamma=0.8, tau1=0.8, offset=0.8), 301)):
        x, t, y = W.simulate(w, n, SplitMix64(C.SEED_BASE + seed), reps)
        es = [E.fit_propensity(x[r], t[r]) for r in range(reps)]
        rows = []
        for c in trims:
            v = np.array([E.hajek_trimmed(t[r], y[r], es[r], c) if c > 0 else E.hajek(t[r], y[r], es[r]) for r in range(reps)])
            v = v[np.isfinite(v)]
            tr = W.trimmed_ate(w, c) if c > 0 else W.ate(w)
            rows.append({"trim": c, "mean": float(v.mean()), "sd": float(v.std()), "target": tr, "bias_ate": float(v.mean() - W.ate(w)), "bias": float(v.mean() - tr)})
        out[label] = rows
    return out


# ----------------------------------------------------------------------------- Experiment 4: Abdeckung der Bootstrap-Intervalle

COVER_NAMES = ("Regression (linear)", "Gewichten (normiert)", "Matching (ATE)")


def coverage_experiment(reps: int = C.EXP_COVER_REPS, boot: int = C.EXP_COVER_B, n: int = C.EXP_N) -> dict:
    out = {}
    for label, gamma, seed in (("Grundfall", 0.8, 400), ("knappe Überlappung", 2.5, 401)):
        w = W.World(gamma=gamma)
        x, t, y = W.simulate(w, n, SplitMix64(C.SEED_BASE + seed), reps)
        rng = SplitMix64(C.SEED_BASE + seed + 50)
        hit = {name: 0 for name in COVER_NAMES}
        width = {name: [] for name in COVER_NAMES}
        for r in range(reps):
            idx = E.bootstrap_indices(rng, n, boot)
            ci = E.bootstrap_all(x[r], t[r], y[r], idx, C.DEFAULT_K, C.DEFAULT_TRIM, False, only=COVER_NAMES)
            for name in COVER_NAMES:
                lo, hi = ci[name]
                hit[name] += bool(lo <= W.ate(w) <= hi)
                width[name].append(hi - lo)
        out[label] = {name: {"cover": hit[name] / reps, "width": float(np.mean(width[name]))} for name in COVER_NAMES}
        out[label]["reps"], out[label]["boot"] = reps, boot
    return out


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def verdict_instance(a: Analysis) -> tuple[str, str]:
    naive = next(r for r in a.rows if r.name == "Naiver Vergleich")
    ate_rows = [r for r in a.rows if r.estimand == "ATE" and r.name != "Naiver Vergleich"]
    best = min(ate_rows, key=lambda r: abs(r.error))
    worst = max(ate_rows, key=lambda r: abs(r.error))
    text = (f"Der naive Vergleich liegt um {signed(naive.error)} neben dem ATE ({de(a.ate)}). Am nächsten kommt „{best.name}“ ({signed(best.error)}), am weitesten daneben "
            f"liegt „{worst.name}“ ({signed(worst.error)}).")
    if a.ess < 0.5 * a.settings.n or a.outside > 0.2:
        return "warning", text + f" Die Überlappung ist knapp: effektive Stichprobe {de(a.ess, 0)} von {a.settings.n}, {de(100 * a.outside, 0)} % der Einheiten mit Neigungswert unter 0,05 oder über 0,95."
    if abs(worst.error) > 0.2:
        return "warning", text
    return "success", text


def verdict_scenarios(res: dict) -> tuple[str, str]:
    ate_names = [n for n in E.NAMES if E.ESTIMAND[n] == "ATE" and n != "Naiver Vergleich"]
    best, worst = {}, {}
    for r in res.values():
        b = min(ate_names, key=lambda n: r["estimators"][n]["rmse"])
        w = max(ate_names, key=lambda n: abs(r["estimators"][n]["bias"]))
        best[b] = best.get(b, 0) + 1
        worst[w] = worst.get(w, 0) + 1
    top = max(best, key=best.get)
    return "info", (f"In {len(res)} Szenarien hat „{top}“ in {best[top]} den kleinsten RMSE unter den ATE-Schätzern; der größte Fehler im Mittel gehört je nach Szenario zu "
                    f"{len(worst)} verschiedenen Schätzern.")


def verdict_overlap(rows: list) -> tuple[str, str]:
    first, last = rows[0], rows[-1]
    return "info", (f"Bei γ = {de(first['gamma'], 1)} ist die effektive Stichprobe {de(first['ess'], 0)} von {first['n']}, bei γ = {de(last['gamma'], 1)} nur {de(last['ess'], 0)}; "
                    f"der Anteil mit Neigungswert unter 0,05 oder über 0,95 steigt von {de(100 * first['outside'], 0)} % auf {de(100 * last['outside'], 0)} %.")


def verdict_coverage(res: dict) -> tuple[str, str]:
    a, b = res["Grundfall"], res["knappe Überlappung"]
    return "info", (f"Im Grundfall decken die Bootstrap-Intervalle {de(100 * a['Regression (linear)']['cover'], 0)} / {de(100 * a['Gewichten (normiert)']['cover'], 0)} / "
                    f"{de(100 * a['Matching (ATE)']['cover'], 0)} % (Regression / Gewichten / Matching); bei knapper Überlappung "
                    f"{de(100 * b['Regression (linear)']['cover'], 0)} / {de(100 * b['Gewichten (normiert)']['cover'], 0)} / {de(100 * b['Matching (ATE)']['cover'], 0)} %.")
