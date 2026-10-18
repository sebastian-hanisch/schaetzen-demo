"""Vehikel: Zuteilung, Ergebnis, Wahrheit per Quadratur und Grenzwert der Regression."""

import math

import numpy as np
import pytest

import sch_constants as C
import sch_world as W
from kausal_generator import SplitMix64


def phi(c):
    return math.exp(-0.5 * c * c) / math.sqrt(2.0 * math.pi)


def big(w, n=1_000_000, seed=7):
    x, t, y = W.simulate(w, n, SplitMix64(seed))
    return x[0], t[0], y[0]


def test_propensity_is_symmetric_without_offset_and_curvature():
    w = W.World(gamma=1.2)
    x = np.array([-2.0, -0.5, 0.0, 0.7, 3.0])
    p = W.propensity(x, w)
    assert p[2] == pytest.approx(0.5) and (p + W.propensity(-x, w)).tolist() == pytest.approx([1.0] * 5)
    assert W.propensity(np.array([0.0]), W.World(gamma=1.0, offset=0.4))[0] > 0.5
    assert W.propensity(np.array([0.0]), W.World(gamma=1.0, delta=1.0))[0] < 0.5      # δ (x² − 1) = −δ bei x = 0


def test_outcome_mean_by_hand():
    w = W.World(curve=0.5, tau1=0.4)
    assert W.outcome_mean(np.array([2.0]), np.array([1.0]), w)[0] == pytest.approx(0.6 * 2 + 0.5 * 3 + (0.5 + 0.8))
    assert W.outcome_mean(np.array([0.0]), np.array([0.0]), w)[0] == pytest.approx(-0.5)


def test_ate_and_att_closed_forms():
    w = W.World(gamma=0.8, tau1=0.8, offset=0.8)
    c = 0.8 / math.sqrt(1 + 0.8 ** 2)
    share = 0.5 * (1 + math.erf(c / math.sqrt(2)))
    att = C.TAU0 + 0.8 * (0.8 / math.sqrt(1 + 0.8 ** 2)) * phi(c) / share
    assert W.ate(w) == pytest.approx(C.TAU0) and W.treated_share(w) == pytest.approx(share, abs=1e-9)
    assert W.att(w) == pytest.approx(att, abs=1e-9)
    assert W.att(W.World(gamma=0.8, tau1=0.0, offset=0.8)) == pytest.approx(C.TAU0)
    assert W.att(W.World(gamma=0.8, tau1=0.8)) > C.TAU0                                   # Behandelte haben größeres X


def test_truth_against_a_large_simulation():
    w = W.World(gamma=1.0, tau1=0.8, offset=0.5)
    x, t, y = big(w)
    assert t.mean() == pytest.approx(W.treated_share(w), abs=0.003)
    assert (C.TAU0 + w.tau1 * x[t == 1].mean()) == pytest.approx(W.att(w), abs=0.005)
    p = W.propensity(x, w)
    inside = (p > 0.1) & (p < 0.9)
    assert (C.TAU0 + w.tau1 * x[inside].mean()) == pytest.approx(W.trimmed_ate(w, 0.1), abs=0.005)


def test_trimmed_ate_without_trimming_is_the_ate():
    for w in (W.World(gamma=0.8), W.World(gamma=0.8, tau1=0.8, offset=0.8), W.World(gamma=2.5, tau1=0.4)):
        assert W.trimmed_ate(w, 0.0) == pytest.approx(W.ate(w), abs=1e-4)
    w = W.World(gamma=0.8, tau1=0.8, offset=0.8)
    assert W.trimmed_ate(w, 0.1) < W.ate(w) - 0.2                                         # Kappung ändert die Schätzgröße
    assert W.trimmed_ate(W.World(gamma=0.8, tau1=0.8), 0.1) == pytest.approx(C.TAU0, abs=1e-6)   # symmetrisch: unverändert


def test_regression_limit_without_confounding_is_the_effect():
    for kind in ("linear", "interaction", "quadratic"):
        assert W.regression_limit(W.World(gamma=0.0, curve=0.8, tau1=0.5), kind) == pytest.approx(C.TAU0, abs=1e-9)


def test_regression_limit_against_a_large_sample():
    cases = [(W.World(gamma=0.8, curve=0.8, offset=0.8), "linear", 0.0823), (W.World(gamma=0.8, curve=0.8, offset=0.8), "interaction", 1.0105),
             (W.World(gamma=0.8, curve=0.8, offset=0.8), "quadratic", C.TAU0), (W.World(gamma=0.8, tau1=0.8, offset=0.8), "linear", 0.0849)]
    for w, kind, expected in cases:
        assert W.regression_limit(w, kind) == pytest.approx(expected, abs=2e-4)
        x, t, y = big(w, 1_000_000, 11)
        xc = x - x.mean()
        cols = [np.ones_like(x), t, xc]
        if kind == "interaction":
            cols.append(t * xc)
        elif kind == "quadratic":
            cols += [xc * xc, t * xc, t * xc * xc]
        coef = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)[0][1]
        assert coef == pytest.approx(W.regression_limit(w, kind), abs=0.02)


def test_simulate_shapes_and_reproducibility():
    w = W.World(gamma=1.0)
    a = W.simulate(w, 30, SplitMix64(5), reps=4)
    b = W.simulate(w, 30, SplitMix64(5), reps=4)
    assert a[0].shape == a[1].shape == a[2].shape == (4, 30)
    assert all((u == v).all() for u, v in zip(a, b))
    assert set(np.unique(a[1])) <= {0.0, 1.0}
    assert not (W.simulate(w, 30, SplitMix64(6), reps=4)[0] == a[0]).all()


def test_world_helpers():
    assert W.expect(lambda x: x * 0 + 1) == pytest.approx(1.0) and W.expect(lambda x: x * x) == pytest.approx(1.0) and W.expect(lambda x: x) == pytest.approx(0.0, abs=1e-12)
    assert W.expect_grid(lambda x: (x > 0).astype(float)) == pytest.approx(0.5, abs=1e-4)


def test_trimmed_ate_with_an_empty_support_is_nan():
    assert math.isnan(W.trimmed_ate(W.World(gamma=0.8), 0.5))


def test_trimmed_ate_without_trimming_is_exactly_the_ate():
    w = W.World(gamma=2.5, tau1=0.4)
    assert W.trimmed_ate(w, 0.0) == W.ate(w) and W.trimmed_ate(w, -0.1) == W.ate(w)
