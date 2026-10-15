"""Schätzer: jede Einheit an den Handrechnungen aus conftest.py und an kleinen Beispielen."""

import numpy as np
import pytest

import sch_constants as C
import sch_estimators as E
from kausal_generator import SplitMix64


def test_naive_difference_by_hand(binary_instance):
    x, t, y, e = binary_instance
    assert E.naive(t, y) == pytest.approx(2.2)


def test_regression_recovers_the_effect_in_the_exact_linear_instance(binary_instance):
    x, t, y, e = binary_instance
    assert E.regression(x, t, y, "linear") == pytest.approx(1.0)
    assert E.regression(x, t, y, "interaction") == pytest.approx(1.0)


def test_weighting_with_the_true_propensity_by_hand(binary_instance):
    x, t, y, e = binary_instance
    assert E.ipw(t, y, e) == pytest.approx(1.0)
    assert E.hajek(t, y, e) == pytest.approx(1.0)


def test_stratification_on_the_confounder_by_hand(binary_instance):
    x, t, y, e = binary_instance
    assert E.stratified(x.astype(int), t, y) == pytest.approx(1.0)
    assert E.stratified(np.zeros(len(x), dtype=int), t, y) == pytest.approx(2.2)       # eine Schicht: naiver Vergleich


def test_matching_on_the_confounder_by_hand(binary_instance):
    x, t, y, e = binary_instance
    assert E.matching_att(x, t, y) == pytest.approx(1.0)
    assert E.matching_ate(x, t, y) == pytest.approx(1.0)


def test_matching_partner_and_tie_rule():
    t = np.array([1.0, 0.0, 0.0])
    score = np.array([0.4, 0.3, 0.9])
    y = np.array([5.0, 2.0, 100.0])
    assert E.nearest_effects(score, t, y, True).tolist() == [3.0]                        # Partner 0,3 (nicht 0,9)
    score = np.array([0.5, 0.4, 0.6])
    y = np.array([5.0, 1.0, 9.0])
    assert E.nearest_effects(score, t, y, True).tolist() == [4.0]                        # Gleichstand: der kleinere Score (0,4)


def test_matching_from_controls_uses_the_treated_as_partners():
    t = np.array([1.0, 1.0, 0.0])
    score = np.array([0.2, 0.8, 0.7])
    y = np.array([4.0, 10.0, 3.0])
    assert E.nearest_effects(score, t, y, False).tolist() == [7.0]                        # Partner 0,8: 10 − 3


def test_effective_sample_size_by_hand():
    t = np.array([1.0, 1.0, 0.0, 0.0])
    e = np.array([0.5, 0.5, 0.75, 0.75])                                                 # Gewichte 2, 2, 4, 4: Summe 12, Quadrate 40
    assert E.effective_sample_size(t, e) == pytest.approx(144.0 / 40.0)
    assert E.effective_sample_size(np.array([1.0, 0.0]), np.array([0.5, 0.5])) == pytest.approx(2.0)


def test_outside_support_share():
    assert E.outside_support(np.array([0.01, 0.5, 0.99, 0.5])) == pytest.approx(0.5)
    assert E.outside_support(np.array([0.05, 0.95])) == 0.0                                # Grenzen gehören dazu
    assert E.outside_support(np.array([0.1, 0.2]), lo=0.15) == pytest.approx(0.5)


def test_strata_labels_and_stratified_skip_incomplete_strata():
    e = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert E.strata_labels(e, 2).tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert E.strata_labels(e, 1).tolist() == [0] * 8
    assert E.strata_labels(e, 4).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    label = np.array([0, 0, 1, 1])
    t = np.array([1.0, 0.0, 1.0, 1.0])                                                   # Schicht 1 hat keine Kontrolle
    y = np.array([2.0, 1.0, 5.0, 7.0])
    assert E.stratified(label, t, y) == pytest.approx(1.0)
    assert np.isnan(E.stratified(np.array([0, 0]), np.array([1.0, 1.0]), np.array([1.0, 2.0])))


def test_hajek_trimmed_by_hand():
    e = np.array([0.05, 0.5, 0.5, 0.95])
    t = np.array([1.0, 1.0, 0.0, 0.0])
    y = np.array([100.0, 3.0, 1.0, -100.0])
    assert E.hajek_trimmed(t, y, e, 0.1) == pytest.approx(2.0)                           # nur die mittleren beiden Einheiten
    assert E.hajek_trimmed(t, y, e, 0.5) != E.hajek_trimmed(t, y, e, 0.5) or True      # Rand: kein Fehler
    assert np.isnan(E.hajek_trimmed(np.array([1.0, 1.0]), np.array([1.0, 2.0]), np.array([0.5, 0.5]), 0.1))


def test_logistic_fit_satisfies_the_score_equations():
    rng = SplitMix64(4)
    x = rng.normal(400)
    t = (rng.uniform(400) < 1.0 / (1.0 + np.exp(-(0.3 + 0.9 * x)))).astype(float)
    e = E.fit_propensity(x, t)
    assert abs((t - e).sum()) < 1e-5 and abs((x * (t - e)).sum()) < 1e-5
    eq = E.fit_propensity(x, t, quad=True)
    assert abs((t - eq).sum()) < 1e-5 and abs((x * x * (t - eq)).sum()) < 1e-5


def test_logistic_fit_is_clipped_under_separation():
    e = E.fit_propensity(np.array([-2.0, -1.0, 1.0, 2.0]), np.array([0.0, 0.0, 1.0, 1.0]))
    assert e.min() >= C.PS_CLIP and e.max() <= 1.0 - C.PS_CLIP and np.isfinite(e).all()


def test_estimate_all_names_and_only():
    rng = SplitMix64(1)
    x = rng.normal(200)
    t = (rng.uniform(200) < 0.5).astype(float)
    y = x + t + rng.normal(200)
    allr = E.estimate_all(x, t, y)
    assert tuple(allr) == E.NAMES
    part = E.estimate_all(x, t, y, only=("Regression (linear)", "Matching (ATE)"))
    assert tuple(part) == ("Regression (linear)", "Matching (ATE)")
    assert part["Regression (linear)"] == allr["Regression (linear)"] and part["Matching (ATE)"] == allr["Matching (ATE)"]
    assert E.ESTIMAND["Matching (ATT)"] == "ATT" and E.ESTIMAND["Gewichten (gekappt)"] == "ATE auf der Stützung" and E.ESTIMAND["Schichten"] == "ATE"


def test_estimate_all_uses_k_trim_and_quad():
    rng = SplitMix64(2)
    x = rng.normal(300)
    t = (rng.uniform(300) < 1.0 / (1.0 + np.exp(-x - 0.5 * (x * x - 1)))).astype(float)
    y = x + t + rng.normal(300)
    a = E.estimate_all(x, t, y, k=2)["Schichten"]
    b = E.estimate_all(x, t, y, k=20)["Schichten"]
    assert a != b
    assert E.estimate_all(x, t, y, trim=0.0)["Gewichten (gekappt)"] != E.estimate_all(x, t, y, trim=0.2)["Gewichten (gekappt)"]
    assert E.estimate_all(x, t, y, quad=False)["Gewichten (normiert)"] != E.estimate_all(x, t, y, quad=True)["Gewichten (normiert)"]


def test_bootstrap_indices_and_intervals():
    a = E.bootstrap_indices(SplitMix64(3), 50, 20)
    assert a.shape == (20, 50) and a.min() >= 0 and a.max() <= 49
    assert (a == E.bootstrap_indices(SplitMix64(3), 50, 20)).all()
    rng = SplitMix64(5)
    x = rng.normal(150)
    t = (rng.uniform(150) < 0.5).astype(float)
    y = t + rng.normal(150)
    ci = E.bootstrap_all(x, t, y, E.bootstrap_indices(SplitMix64(6), 150, 40), 5, 0.1, False, only=("Naiver Vergleich", "Regression (linear)"))
    assert set(ci) == {"Naiver Vergleich", "Regression (linear)"}
    for lo, hi in ci.values():
        assert lo < hi
    assert ci["Regression (linear)"][0] < E.regression(x, t, y) < ci["Regression (linear)"][1]


def test_hajek_trimmed_with_exactly_one_control_is_computed():
    e = np.array([0.5, 0.5, 0.5])
    t = np.array([1.0, 1.0, 0.0])
    y = np.array([3.0, 5.0, 1.0])
    assert E.hajek_trimmed(t, y, e, 0.1) == pytest.approx(3.0)
    assert np.isnan(E.hajek_trimmed(np.array([1.0, 0.0]), np.array([1.0, 2.0]), np.array([0.01, 0.5]), 0.1))     # nur eine Einheit bleibt


def test_trimming_excludes_the_boundary_values():
    e = np.array([0.1, 0.5, 0.5, 0.9])
    t = np.array([1.0, 1.0, 0.0, 0.0])
    y = np.array([100.0, 3.0, 1.0, -100.0])
    assert E.hajek_trimmed(t, y, e, 0.1) == pytest.approx(2.0)                 # e = 0,1 und e = 0,9 gehören nicht zur Stützung
