"""Orakel: unabhängige Rechenwege (Brute Force, Normalgleichungen, Monte Carlo). Unter zehn Sekunden."""

import numpy as np
import pytest

import sch_estimators as E
import sch_world as W
from kausal_generator import SplitMix64


def sample(n=300, seed=3, w=W.World(gamma=0.9, curve=0.4, tau1=0.4, offset=0.3)):
    x, t, y = W.simulate(w, n, SplitMix64(seed))
    return x[0], t[0], y[0]


def test_regression_against_the_normal_equations():
    x, t, y = sample()
    xc = x - x.mean()
    for kind, cols in (("linear", [np.ones_like(x), t, xc]), ("interaction", [np.ones_like(x), t, xc, t * xc]),
                       ("quadratic", [np.ones_like(x), t, xc, xc * xc, t * xc, t * xc * xc])):
        d = np.column_stack(cols)
        beta = np.linalg.solve(d.T @ d, d.T @ y)
        assert E.regression(x, t, y, kind) == pytest.approx(beta[1], rel=1e-9)


def test_weighting_against_loops():
    x, t, y = sample()
    e = E.fit_propensity(x, t)
    total = sum((t[i] * y[i] / e[i]) - ((1 - t[i]) * y[i] / (1 - e[i])) for i in range(len(y))) / len(y)
    assert E.ipw(t, y, e) == pytest.approx(total, rel=1e-9)
    s1 = sum(y[i] / e[i] for i in range(len(y)) if t[i] == 1) / sum(1 / e[i] for i in range(len(y)) if t[i] == 1)
    s0 = sum(y[i] / (1 - e[i]) for i in range(len(y)) if t[i] == 0) / sum(1 / (1 - e[i]) for i in range(len(y)) if t[i] == 0)
    assert E.hajek(t, y, e) == pytest.approx(s1 - s0, rel=1e-9)


def test_matching_against_brute_force():
    x, t, y = sample(150, 5)
    e = E.fit_propensity(x, t)
    treated = [i for i in range(len(y)) if t[i] == 1]
    control = [i for i in range(len(y)) if t[i] == 0]

    def best(i, pool):
        d = [abs(e[i] - e[j]) for j in pool]
        return pool[int(np.argmin(d))]

    att = np.mean([y[i] - y[best(i, control)] for i in treated])
    atc = np.mean([y[best(j, treated)] - y[j] for j in control])
    assert E.matching_att(e, t, y) == pytest.approx(att, rel=1e-9)
    ate = (len(treated) * att + len(control) * atc) / len(y)
    assert E.matching_ate(e, t, y) == pytest.approx(ate, rel=1e-9)


def test_stratification_against_a_dictionary_loop():
    x, t, y = sample(400, 6)
    e = E.fit_propensity(x, t)
    k = 7
    labels = E.strata_labels(e, k)
    by = {}
    for lab, ti, yi in zip(labels, t, y):
        by.setdefault(int(lab), {1: [], 0: []})[int(ti)].append(yi)
    num = den = 0.0
    for g in by.values():
        if g[1] and g[0]:
            size = len(g[1]) + len(g[0])
            num += size * (np.mean(g[1]) - np.mean(g[0]))
            den += size
    assert E.stratified(labels, t, y) == pytest.approx(num / den, rel=1e-9)
    assert sorted(np.unique(labels)) == list(range(k)) and np.bincount(labels).min() >= len(y) // k - 1


def test_strata_are_balanced_in_size():
    e = np.random.default_rng(0).random(1000)
    counts = np.bincount(E.strata_labels(e, 10))
    assert counts.min() >= 99 and counts.max() <= 101


def test_weighting_with_the_true_propensity_is_unbiased():
    w = W.World(gamma=1.0, tau1=0.4, offset=0.4, curve=0.5)
    x, t, y = W.simulate(w, 400, SplitMix64(9), reps=400)
    est = [E.ipw(t[r], y[r], W.propensity(x[r], w)) for r in range(400)]
    assert np.mean(est) == pytest.approx(W.ate(w), abs=4 * np.std(est) / 20)


def test_matching_att_is_close_to_the_att_for_a_good_overlap():
    w = W.World(gamma=0.8, tau1=0.8, offset=0.8)
    x, t, y = W.simulate(w, 500, SplitMix64(10), reps=300)
    est = [E.matching_att(E.fit_propensity(x[r], t[r]), t[r], y[r]) for r in range(300)]
    assert np.mean(est) == pytest.approx(W.att(w), abs=0.05)
    assert abs(np.mean(est) - W.ate(w)) > 0.15                                         # und nicht der ATE


def test_the_naive_difference_is_biased_by_the_expected_amount():
    w = W.World(gamma=0.8)
    x, t, y = W.simulate(w, 500, SplitMix64(12), reps=200)
    est = np.array([E.naive(t[r], y[r]) for r in range(200)])
    # E[X | T = 1] − E[X | T = 0] mal β = 0,6 · 2 · γ/√(1+γ²) · φ(0)/(0,5 · …) ; hier numerisch aus der Quadratur
    ex1 = W.expect(lambda v: v * W.propensity(v, w)) / 0.5
    ex0 = W.expect(lambda v: v * (1 - W.propensity(v, w))) / 0.5
    assert est.mean() == pytest.approx(W.ate(w) + 0.6 * (ex1 - ex0), abs=0.03)
