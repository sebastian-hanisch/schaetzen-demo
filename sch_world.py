"""Vehikel: ein Standort wählt die neue Regel B (T = 1) abhängig von der Schwankung X; Ergebnis Servicegrad Y. Wahrheit in geschlossener Form
(Gauss-Hermite-Quadratur über X). Nur numpy und der Generator der Linie, kein Streamlit.

Modell: X ~ N(0, 1); P(T = 1 | X) = Φ(γ X + δ (X² − 1) + Versatz); Y = β X + Krümmung (X² − 1) + (τ0 + τ1 X) T + σ ε.
ATE = τ0 (X ist symmetrisch um 0); ATT = τ0 + τ1 E[X | T = 1]."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import sch_constants as C
from kausal_generator import SplitMix64, normal_cdf

_NODES, _WEIGHTS = np.polynomial.hermite_e.hermegauss(96)
_WEIGHTS = _WEIGHTS / np.sqrt(2.0 * np.pi)          # Gewichte der Standardnormalverteilung (Summe 1)


@dataclass(frozen=True)
class World:
    gamma: float = C.DEFAULT_GAMMA
    offset: float = 0.0
    delta: float = 0.0
    curve: float = 0.0
    tau1: float = 0.0


def propensity(x, w: World) -> np.ndarray:
    """Wahre Zuteilungswahrscheinlichkeit P(T = 1 | X = x)."""
    x = np.asarray(x, dtype=float)
    return normal_cdf(w.gamma * x + w.delta * (x * x - 1.0) + w.offset)


def outcome_mean(x, t, w: World) -> np.ndarray:
    """E[Y | X = x, T = t]."""
    x, t = np.asarray(x, dtype=float), np.asarray(t, dtype=float)
    return C.BETA * x + w.curve * (x * x - 1.0) + (C.TAU0 + w.tau1 * x) * t


def simulate(w: World, n: int, rng: SplitMix64, reps: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, t, y) mit Form (reps, n): erst alle Störfaktoren, dann alle Zuteilungs-Gleichverteilten, dann das Rauschen aus einem Strom."""
    size = reps * n
    x = rng.normal(size).reshape(reps, n)
    u = rng.uniform(size).reshape(reps, n)
    eps = rng.normal(size).reshape(reps, n)
    t = (u < propensity(x, w)).astype(float)
    y = outcome_mean(x, t, w) + C.SIGMA * eps
    return x, t, y


# ----------------------------------------------------------------------------- Wahrheit

def expect(f) -> float:
    """E[f(X)] über die Standardnormalverteilung (Gauss-Hermite; für glatte Funktionen)."""
    return float(np.sum(_WEIGHTS * f(_NODES)))


_GRID = np.linspace(-9.0, 9.0, 180001)
_GRID_W = np.exp(-0.5 * _GRID * _GRID) / np.sqrt(2.0 * np.pi) * (_GRID[1] - _GRID[0])


def expect_grid(f) -> float:
    """E[f(X)] auf einem feinen Gitter (für Funktionen mit Sprüngen, etwa Indikatoren)."""
    return float(np.sum(_GRID_W * f(_GRID)))


def ate(w: World) -> float:
    return C.TAU0 + w.tau1 * expect(lambda x: x)


def treated_share(w: World) -> float:
    return expect(lambda x: propensity(x, w))


def att(w: World) -> float:
    """Wirkung auf die Behandelten: τ0 + τ1 E[X | T = 1]."""
    return C.TAU0 + w.tau1 * expect(lambda x: x * propensity(x, w)) / treated_share(w)


def trimmed_ate(w: World, c: float) -> float:
    """Durchschnittliche Wirkung auf der Stützung c < e(X) < 1 − c (die Schätzgröße der gekappten Gewichtung)."""
    if c <= 0.0:
        return ate(w)                      # ohne Kappung zählt jede Einheit (der Neigungswert sättigt numerisch bei 1)

    def inside(x):
        p = propensity(x, w)
        return ((p > c) & (p < 1.0 - c)).astype(float)

    mass = expect_grid(inside)
    if mass <= 0:
        return float("nan")
    return C.TAU0 + w.tau1 * expect_grid(lambda x: x * inside(x)) / mass


def regression_limit(w: World, kind: str) -> float:
    """Koeffizient von T in der Regression von Y auf die Merkmale von `kind` bei n → ∞ (X mit Mittel 0).
    kind: 'linear' (1, T, X), 'interaction' (1, T, X, T·X), 'quadratic' (1, T, X, X², T·X, T·X²)."""
    def features(x, t):
        cols = [np.ones_like(x), t, x]
        if kind == "interaction":
            cols.append(t * x)
        elif kind == "quadratic":
            cols += [x * x, t * x, t * x * x]
        return np.stack(cols, axis=1)

    p = propensity(_NODES, w)
    zz = 0.0
    zy = 0.0
    for t, pt in ((0.0, 1.0 - p), (1.0, p)):
        z = features(_NODES, np.full_like(_NODES, t))
        weight = _WEIGHTS * pt
        zz = zz + (z * weight[:, None]).T @ z
        zy = zy + (z * weight[:, None]).T @ outcome_mean(_NODES, np.full_like(_NODES, t), w)
    return float(np.linalg.solve(zz, zy)[1])
