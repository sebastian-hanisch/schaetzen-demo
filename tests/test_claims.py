"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über SplitMix64-Ströme (ganzzahlig, plattformstabil); Fließkomma-Reihenfolgen können je Bibliotheksversion die letzte Stelle ändern, deshalb
Toleranzen von einer halben Einheit der gedruckten Stelle.
"""

import pytest

import sch_constants as C
import sch_evaluation as ev
import sch_world as W


@pytest.fixture(scope="module")
def sc():
    return ev.scenarios_experiment()


@pytest.fixture(scope="module")
def overlap():
    return ev.overlap_experiment()


@pytest.fixture(scope="module")
def strata():
    return ev.strata_experiment()


@pytest.fixture(scope="module")
def trim():
    return ev.trim_experiment()


@pytest.fixture(scope="module")
def coverage():
    return ev.coverage_experiment()


def bias(res, scenario, name):
    return res[scenario]["estimators"][name]["bias"]


def sd(res, scenario, name):
    return res[scenario]["estimators"][name]["sd"]


S1, S2B, S3B, S4, S4C, S5 = ("S1 Grundfall", "S2b gekrümmtes Ergebnis, Versatz", "S3b heterogener Effekt, Versatz", "S4 knappe Überlappung",
                             "S4c knappe Überlappung, gekrümmt, Versatz", "S5 gekrümmte Zuteilung, Neigungsmodell linear")


def test_scenario_s1_base_case(sc):
    assert bias(sc, S1, "Naiver Vergleich") == pytest.approx(0.60, abs=0.005)
    assert bias(sc, S1, "Regression (linear)") == pytest.approx(0.0, abs=0.005) and sd(sc, S1, "Regression (linear)") == pytest.approx(0.105, abs=0.0006)
    assert bias(sc, S1, "Gewichten (normiert)") == pytest.approx(0.03, abs=0.005) and sd(sc, S1, "Gewichten (normiert)") == pytest.approx(0.124, abs=0.0006)
    assert bias(sc, S1, "Schichten") == pytest.approx(0.07, abs=0.005) and sd(sc, S1, "Schichten") == pytest.approx(0.110, abs=0.0006)
    assert bias(sc, S1, "Matching (ATE)") == pytest.approx(0.02, abs=0.005) and sd(sc, S1, "Matching (ATE)") == pytest.approx(0.138, abs=0.0006)
    assert sc[S1]["ess"] == pytest.approx(314, abs=1)


def test_scenario_s2b_curved_and_offset(sc):
    assert bias(sc, S2B, "Regression (linear)") == pytest.approx(-0.42, abs=0.005)
    assert bias(sc, S2B, "Regression (Wechselwirkung)") == pytest.approx(0.51, abs=0.005)
    assert bias(sc, S2B, "Regression (quadratisch)") == pytest.approx(0.0, abs=0.005)
    assert bias(sc, S2B, "Gewichten (normiert)") == pytest.approx(0.19, abs=0.005)
    assert bias(sc, S2B, "Matching (ATE)") == pytest.approx(0.13, abs=0.005) and bias(sc, S2B, "Schichten") == pytest.approx(0.08, abs=0.005)


def test_scenario_s3b_heterogeneous_effect(sc):
    r = sc[S3B]
    assert r["ate"] == pytest.approx(0.50) and r["att"] == pytest.approx(0.72, abs=0.005)
    assert bias(sc, S3B, "Regression (linear)") == pytest.approx(-0.42, abs=0.005)
    assert W.regression_limit(W.World(**C.SCENARIOS[S3B]), "linear") == pytest.approx(0.085, abs=0.0006)
    assert bias(sc, S3B, "Regression (Wechselwirkung)") == pytest.approx(0.0, abs=0.005)
    assert bias(sc, S3B, "Gewichten (normiert)") == pytest.approx(0.05, abs=0.005)
    assert bias(sc, S3B, "Matching (ATT)") == pytest.approx(0.04, abs=0.005)
    assert r["estimators"]["Gewichten (gekappt)"]["bias_ate"] == pytest.approx(-0.38, abs=0.005)
    assert bias(sc, S3B, "Gewichten (gekappt)") == pytest.approx(-0.02, abs=0.005)
    assert r["estimators"]["Gewichten (gekappt)"]["target"] == pytest.approx(0.15, abs=0.005)


def test_scenario_s4_poor_overlap(sc):
    assert sc[S4]["ess"] == pytest.approx(132, abs=1) and 100 * sc[S4]["outside"] == pytest.approx(51, abs=0.5)
    assert bias(sc, S4, "Gewichten (normiert)") == pytest.approx(0.33, abs=0.005) and bias(sc, S4, "Matching (ATE)") == pytest.approx(0.16, abs=0.005)
    assert bias(sc, S4, "Regression (linear)") == pytest.approx(0.01, abs=0.005) and bias(sc, S4, "Gewichten (gekappt)") == pytest.approx(-0.01, abs=0.005)


def test_scenario_s4c(sc):
    assert bias(sc, S4C, "Regression (linear)") == pytest.approx(-0.70, abs=0.005)
    assert bias(sc, S4C, "Regression (Wechselwirkung)") == pytest.approx(0.60, abs=0.005)
    assert bias(sc, S4C, "Regression (quadratisch)") == pytest.approx(0.01, abs=0.005)
    assert bias(sc, S4C, "Gewichten (normiert)") == pytest.approx(0.40, abs=0.005) and bias(sc, S4C, "Matching (ATE)") == pytest.approx(0.32, abs=0.005)
    assert bias(sc, S4C, "Matching (ATT)") == pytest.approx(0.73, abs=0.005)


def test_scenario_s5_wrong_propensity_model(sc):
    assert bias(sc, S5, "Gewichten (normiert)") == pytest.approx(-0.16, abs=0.005) and bias(sc, S5, "Gewichten (IPW)") == pytest.approx(-0.14, abs=0.005)
    assert bias(sc, S5, "Gewichten (gekappt)") == pytest.approx(-0.09, abs=0.005)
    assert bias(sc, S5, "Regression (linear)") == pytest.approx(0.0, abs=0.005) and bias(sc, S5, "Schichten") == pytest.approx(0.01, abs=0.005)
    assert bias(sc, S5, "Matching (ATE)") == pytest.approx(0.04, abs=0.005)


def test_experiment_2_overlap(overlap):
    first, last = overlap[0], overlap[-1]
    assert (first["gamma"], last["gamma"]) == (0.2, 3.0)
    assert first["ess"] == pytest.approx(485, abs=1) and last["ess"] == pytest.approx(131, abs=1)
    assert 100 * first["outside"] == pytest.approx(0, abs=0.05) and 100 * last["outside"] == pytest.approx(58, abs=0.5)
    assert first["estimators"]["Gewichten (normiert)"]["bias_ate"] == pytest.approx(0.01, abs=0.005)
    assert last["estimators"]["Gewichten (normiert)"]["bias_ate"] == pytest.approx(0.41, abs=0.005)
    assert first["estimators"]["Matching (ATE)"]["bias_ate"] == pytest.approx(0.0, abs=0.005) and last["estimators"]["Matching (ATE)"]["bias_ate"] == pytest.approx(0.16, abs=0.005)
    reg = [r["estimators"]["Regression (linear)"]["bias_ate"] for r in overlap]
    cut = [r["estimators"]["Gewichten (gekappt)"]["bias_ate"] for r in overlap]
    assert min(reg) >= -0.0105 and max(reg) <= 0.0105 and min(cut) >= -0.0205 and max(cut) <= 0.0105


def test_experiment_3_strata(strata):
    by = {r["k"]: r for r in strata}
    expected = {1: 0.60, 2: 0.23, 3: 0.14, 5: 0.07, 10: 0.03, 20: 0.01, 40: 0.01}
    for k, v in expected.items():
        assert by[k]["bias"] == pytest.approx(v, abs=0.005), k
    assert by[1]["sd"] == pytest.approx(0.10, abs=0.005) and by[20]["sd"] == pytest.approx(0.13, abs=0.005) and by[40]["sd"] == pytest.approx(0.12, abs=0.005)


def test_experiment_3_trimming_changes_the_estimand(trim):
    base = {round(r["trim"], 2): r for r in trim["Grundfall"]}
    assert base[0.0]["bias"] == pytest.approx(0.03, abs=0.005) and base[0.1]["bias"] == pytest.approx(0.0, abs=0.005) and base[0.1]["target"] == pytest.approx(0.5)
    het = {round(r["trim"], 2): r for r in trim["heterogen, Versatz"]}
    assert het[0.0]["mean"] == pytest.approx(0.55, abs=0.005) and het[0.1]["mean"] == pytest.approx(0.13, abs=0.005) and het[0.2]["mean"] == pytest.approx(-0.10, abs=0.005)
    assert het[0.0]["target"] == pytest.approx(0.50, abs=0.005) and het[0.1]["target"] == pytest.approx(0.15, abs=0.005) and het[0.2]["target"] == pytest.approx(-0.06, abs=0.005)
    assert max(abs(r["bias"]) for r in trim["heterogen, Versatz"]) <= 0.065


def test_experiment_4_coverage(coverage):
    a, b = coverage["Grundfall"], coverage["knappe Überlappung"]
    assert (a["reps"], a["boot"]) == (100, 60)
    assert 100 * a["Regression (linear)"]["cover"] == pytest.approx(91, abs=0.5) and 100 * a["Gewichten (normiert)"]["cover"] == pytest.approx(94, abs=0.5)
    assert 100 * a["Matching (ATE)"]["cover"] == pytest.approx(93, abs=0.5)
    assert 100 * b["Regression (linear)"]["cover"] == pytest.approx(90, abs=0.5) and 100 * b["Gewichten (normiert)"]["cover"] == pytest.approx(37, abs=0.5)
    assert 100 * b["Matching (ATE)"]["cover"] == pytest.approx(86, abs=0.5)
