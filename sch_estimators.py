"""Schätzer: naive Differenz, Regression, Gewichten, Schichten, Matching, logistische Regression für den Neigungswert und Bootstrap.
Nur numpy, kein Streamlit; jede Einheit eine reine Funktion auf Feldern (x, t, y) einer Stichprobe."""

from __future__ import annotations

import numpy as np

import sch_constants as C

# Name -> (Schätzgröße, Familie)
NAMES = (
    "Naiver Vergleich",
    "Regression (linear)",
    "Regression (Wechselwirkung)",
    "Regression (quadratisch)",
    "Gewichten (IPW)",
    "Gewichten (normiert)",
    "Gewichten (gekappt)",
    "Schichten",
    "Matching (ATE)",
    "Matching (ATT)",
)
ESTIMAND = {name: "ATE" for name in NAMES}
ESTIMAND["Matching (ATT)"] = "ATT"
ESTIMAND["Gewichten (gekappt)"] = "ATE auf der Stützung"


# ----------------------------------------------------------------------------- Neigungswert

def fit_propensity(x: np.ndarray, t: np.ndarray, quad: bool = False, iters: int = 30) -> np.ndarray:
    """Logistische Regression von T auf (1, X[, X²]) nach Newton; Rückgabe: angepasster Neigungswert, auf [PS_CLIP, 1 − PS_CLIP] begrenzt."""
    cols = [np.ones_like(x), x] + ([x * x] if quad else [])
    d = np.column_stack(cols)
    b = np.zeros(d.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(d @ b, -30.0, 30.0)))
        w = np.maximum(p * (1.0 - p), 1e-10)
        step = np.linalg.solve(d.T @ (d * w[:, None]) + 1e-8 * np.eye(d.shape[1]), d.T @ (t - p))
        b = b + step
        if np.abs(step).max() < 1e-9:
            break
    e = 1.0 / (1.0 + np.exp(-np.clip(d @ b, -30.0, 30.0)))
    return np.clip(e, C.PS_CLIP, 1.0 - C.PS_CLIP)


def effective_sample_size(t: np.ndarray, e: np.ndarray) -> float:
    """(Σw)² / Σw² der Gewichte 1/e (Behandelte) und 1/(1 − e) (Kontrollen)."""
    w = np.where(t == 1, 1.0 / e, 1.0 / (1.0 - e))
    return float(w.sum() ** 2 / (w ** 2).sum())


def outside_support(e: np.ndarray, lo: float = 0.05) -> float:
    """Anteil der Einheiten mit Neigungswert unter `lo` oder über 1 − `lo`."""
    return float(np.mean((e < lo) | (e > 1.0 - lo)))


# ----------------------------------------------------------------------------- Schätzer

def naive(t, y) -> float:
    return float(y[t == 1].mean() - y[t == 0].mean())


def regression(x, t, y, kind: str = "linear") -> float:
    """Koeffizient von T in der Regression von Y auf die Merkmale von `kind`; X wird um den Stichprobenmittelwert zentriert (Wechselwirkung, quadratisch)."""
    xc = x - x.mean()
    cols = [np.ones_like(x), t, xc]
    if kind == "interaction":
        cols.append(t * xc)
    elif kind == "quadratic":
        cols += [xc * xc, t * xc, t * xc * xc]
    return float(np.linalg.lstsq(np.column_stack(cols), y, rcond=None)[0][1])


def ipw(t, y, e) -> float:
    """Horvitz-Thompson: Mittel von T·Y/e − (1 − T)·Y/(1 − e)."""
    return float(np.mean(t * y / e - (1.0 - t) * y / (1.0 - e)))


def hajek(t, y, e) -> float:
    """Normiertes Gewichten: gewichtete Mittel je Gruppe, Gewichte 1/e und 1/(1 − e)."""
    w1, w0 = t / e, (1.0 - t) / (1.0 - e)
    return float((w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum())


def hajek_trimmed(t, y, e, c: float) -> float:
    """Normiertes Gewichten auf den Einheiten mit c < e < 1 − c."""
    m = (e > c) & (e < 1.0 - c)
    with np.errstate(invalid="ignore", divide="ignore"):         # eine Gruppe leer: 0 / 0 gibt nan
        return hajek(t[m], y[m], e[m])


def strata_labels(e: np.ndarray, k: int) -> np.ndarray:
    """Schichten gleicher Größe nach dem Neigungswert (Quantile); Rückgabe: Schichtnummer 0..k−1."""
    edges = np.quantile(e, np.linspace(0.0, 1.0, k + 1))
    return np.clip(np.searchsorted(edges, e, side="right") - 1, 0, k - 1)


def stratified(label, t, y) -> float:
    """Gewichtetes Mittel der Gruppendifferenzen je Schicht (Gewicht: Schichtgröße); Schichten ohne beide Gruppen entfallen."""
    num = den = 0.0
    for s in np.unique(label):
        m = label == s
        n1, n0 = int((t[m] == 1).sum()), int((t[m] == 0).sum())
        if n1 == 0 or n0 == 0:
            continue
        num += m.sum() * (y[m & (t == 1)].mean() - y[m & (t == 0)].mean())
        den += m.sum()
    return float(num / den) if den else float("nan")


def nearest_effects(score, t, y, from_treated: bool) -> np.ndarray:
    """Matching mit Zurücklegen, nächster Nachbar im Score: je Einheit der Quellgruppe die Differenz zum Ergebnis des Partners.
    Gleichstand: der Partner mit dem kleineren Score."""
    src = np.where(t == (1 if from_treated else 0))[0]
    dst = np.where(t == (0 if from_treated else 1))[0]
    order = np.argsort(score[dst], kind="stable")
    s, yd = score[dst][order], y[dst][order]
    pos = np.clip(np.searchsorted(s, score[src]), 1, len(s) - 1)
    left, right = pos - 1, pos
    pick = np.where(np.abs(score[src] - s[left]) <= np.abs(score[src] - s[right]), left, right)
    return (y[src] - yd[pick]) if from_treated else (yd[pick] - y[src])


def matching_att(score, t, y) -> float:
    return float(nearest_effects(score, t, y, True).mean())


def matching_ate(score, t, y) -> float:
    return float(np.concatenate([nearest_effects(score, t, y, True), nearest_effects(score, t, y, False)]).mean())


# ----------------------------------------------------------------------------- alle zusammen

def estimate_all(x, t, y, k: int = C.DEFAULT_K, trim: float = C.DEFAULT_TRIM, quad: bool = False, only=None) -> dict:
    """Alle Schätzer (oder die in `only` genannten) auf einer Stichprobe, mit demselben geschätzten Neigungswert."""
    want = set(NAMES if only is None else only)
    out = {}
    e = fit_propensity(x, t, quad) if want & {n for n in NAMES if n.startswith(("Gewichten", "Schichten", "Matching"))} else None
    if "Naiver Vergleich" in want:
        out["Naiver Vergleich"] = naive(t, y)
    for name, kind in (("Regression (linear)", "linear"), ("Regression (Wechselwirkung)", "interaction"), ("Regression (quadratisch)", "quadratic")):
        if name in want:
            out[name] = regression(x, t, y, kind)
    if "Gewichten (IPW)" in want:
        out["Gewichten (IPW)"] = ipw(t, y, e)
    if "Gewichten (normiert)" in want:
        out["Gewichten (normiert)"] = hajek(t, y, e)
    if "Gewichten (gekappt)" in want:
        out["Gewichten (gekappt)"] = hajek_trimmed(t, y, e, trim)
    if "Schichten" in want:
        out["Schichten"] = stratified(strata_labels(e, k), t, y)
    if "Matching (ATE)" in want:
        out["Matching (ATE)"] = matching_ate(e, t, y)
    if "Matching (ATT)" in want:
        out["Matching (ATT)"] = matching_att(e, t, y)
    return {n: out[n] for n in NAMES if n in out}


def bootstrap_indices(rng, n: int, b: int) -> np.ndarray:
    """(b, n) Indizes für Ziehungen mit Zurücklegen aus einem festen Strom (`rng` mit Methode uniform)."""
    return np.minimum((rng.uniform(b * n).reshape(b, n) * n).astype(int), n - 1)


def bootstrap_all(x, t, y, idx: np.ndarray, k: int, trim: float, quad: bool, only=None, level: float = 0.95) -> dict:
    """Perzentil-Intervalle je Schätzer: alle Schätzer auf jeder der Ziehungen `idx` (mit neu angepasstem Neigungswert)."""
    runs = [estimate_all(x[i], t[i], y[i], k, trim, quad, only) for i in idx]
    a = (1.0 - level) / 2.0
    out = {}
    for name in runs[0]:
        vals = np.array([r[name] for r in runs])
        vals = vals[np.isfinite(vals)]
        out[name] = tuple(float(v) for v in np.quantile(vals, [a, 1.0 - a])) if len(vals) else (float("nan"), float("nan"))
    return out
