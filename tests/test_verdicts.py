"""Verdikte an handgesetzten Zeilen: jede Schwelle und jede Zahl im Text."""

import numpy as np
import pytest

import sch_estimators as E
import sch_evaluation as ev


def row(name, estimate, truth=0.5, ate=0.5, estimand="ATE"):
    return ev.Row(name, estimand, estimate, estimate - 0.1, estimate + 0.1, truth, estimate - truth, estimate - ate, True, None)


def analysis(errors: dict, ess=400.0, outside=0.03, n=500):
    rows = [row(name, 0.5 + errors.get(name, 0.0), estimand=E.ESTIMAND[name]) for name in E.NAMES]
    return ev.Analysis(ev.Settings(n=n), rows, 0.5, 0.5, 0.5, n // 2, np.full(n, 0.5), np.zeros(n), ess, outside)


def test_verdict_instance_good_case():
    a = analysis({"Naiver Vergleich": 0.6, "Regression (linear)": 0.01, "Gewichten (normiert)": 0.04, "Matching (ATE)": -0.05})
    kind, text = ev.verdict_instance(a)
    assert kind == "success"
    assert "Der naive Vergleich liegt um +0,60 neben dem ATE (0,50)" in text
    assert "Am nächsten kommt „Regression (Wechselwirkung)“ (+0,00), am weitesten daneben liegt „Matching (ATE)“ (−0,05)." in text


def test_verdict_instance_names_best_and_worst():
    errors = {"Naiver Vergleich": 0.6, "Regression (linear)": -0.30, "Regression (Wechselwirkung)": 0.02, "Regression (quadratisch)": 0.01, "Gewichten (IPW)": 0.1,
              "Gewichten (normiert)": 0.12, "Schichten": 0.07, "Matching (ATE)": 0.05}
    kind, text = ev.verdict_instance(analysis(errors))
    assert kind == "warning"
    assert "Am nächsten kommt „Regression (quadratisch)“ (+0,01)" in text and "am weitesten daneben liegt „Regression (linear)“ (−0,30)" in text


def test_verdict_instance_overlap_warning_thresholds():
    base = {"Naiver Vergleich": 0.6, "Regression (linear)": 0.01}
    kind, text = ev.verdict_instance(analysis(base, ess=249.0, outside=0.03))                  # ESS unter der Hälfte von n
    assert kind == "warning" and "knapp: effektive Stichprobe 249 von 500, 3 %" in text
    kind, text = ev.verdict_instance(analysis(base, ess=250.0, outside=0.03))                  # genau die Hälfte: noch kein Hinweis
    assert "knapp" not in text
    kind, text = ev.verdict_instance(analysis(base, ess=400.0, outside=0.21))
    assert kind == "warning" and "21 % der Einheiten" in text
    kind, text = ev.verdict_instance(analysis(base, ess=400.0, outside=0.20))
    assert "knapp" not in text


def test_verdict_instance_worst_threshold_is_strict():
    errors = {"Naiver Vergleich": 0.6, "Regression (linear)": 0.2}
    assert ev.verdict_instance(analysis(errors))[0] == "success"
    errors["Regression (linear)"] = 0.2001
    assert ev.verdict_instance(analysis(errors))[0] == "warning"


def test_verdict_overlap_numbers():
    rows = [{"gamma": 0.2, "ess": 485.0, "n": 500, "outside": 0.0}, {"gamma": 3.0, "ess": 131.0, "n": 500, "outside": 0.585}]
    kind, text = ev.verdict_overlap(rows)
    assert kind == "info" and text == ("Bei γ = 0,2 ist die effektive Stichprobe 485 von 500, bei γ = 3,0 nur 131; der Anteil mit Neigungswert unter 0,05 oder über 0,95 steigt "
                                       "von 0 % auf 58 %.")


def test_verdict_coverage_numbers():
    def block(a, b, c):
        return {"Regression (linear)": {"cover": a}, "Gewichten (normiert)": {"cover": b}, "Matching (ATE)": {"cover": c}}

    res = {"Grundfall": block(0.91, 0.94, 0.93), "knappe Überlappung": block(0.90, 0.37, 0.86)}
    kind, text = ev.verdict_coverage(res)
    assert kind == "info" and "91 / 94 / 93 % (Regression / Gewichten / Matching)" in text and "90 / 37 / 86 %" in text


def test_verdict_scenarios_counts():
    def est(rmse, bias):
        return {"rmse": rmse, "bias": bias}

    names = [n for n in E.NAMES if E.ESTIMAND[n] == "ATE" and n != "Naiver Vergleich"]
    s1 = {"estimators": {n: est(0.5, 0.0) for n in E.NAMES}}
    s1["estimators"]["Regression (linear)"] = est(0.1, 0.0)
    s1["estimators"]["Schichten"] = est(0.5, 0.2)
    s2 = {"estimators": {n: est(0.5, 0.0) for n in E.NAMES}}
    s2["estimators"]["Regression (linear)"] = est(0.1, 0.0)
    s2["estimators"]["Gewichten (normiert)"] = est(0.5, -0.3)
    kind, text = ev.verdict_scenarios({"a": s1, "b": s2})
    assert kind == "info" and "In 2 Szenarien hat „Regression (linear)“ in 2 den kleinsten RMSE" in text and "zu 2 verschiedenen Schätzern" in text
    assert names
