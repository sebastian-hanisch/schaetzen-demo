"""Auswertung: Einstellungen, Analyse einer Stichprobe, Experimente (feste Ströme), Formatierung."""

import numpy as np
import pytest

import sch_constants as C
import sch_estimators as E
import sch_evaluation as ev
import sch_world as W


def test_number_format():
    assert ev.de(1.5, 1) == "1,5" and ev.de(-0.25) == "−0,25" and ev.signed(0.3) == "+0,30" and ev.signed(-0.3) == "−0,30" and ev.signed(0.0) == "+0,00"


def test_settings_world():
    s = ev.Settings(gamma=1.5, offset=0.3, delta=0.2, curve=0.4, tau1=0.6)
    assert s.world == W.World(1.5, 0.3, 0.2, 0.4, 0.6)
    assert ev.Settings().world == W.World(C.DEFAULT_GAMMA, 0.0, 0.0, 0.0, 0.0)


def test_targets_by_estimator():
    w = W.World(gamma=0.8, tau1=0.8, offset=0.8)
    assert ev.target(w, "Regression (linear)", 0.1) == pytest.approx(C.TAU0)
    assert ev.target(w, "Matching (ATT)", 0.1) == pytest.approx(W.att(w))
    assert ev.target(w, "Gewichten (gekappt)", 0.1) == pytest.approx(W.trimmed_ate(w, 0.1))
    assert ev.target(w, "Schichten", 0.1) == pytest.approx(C.TAU0)


def test_analysis_rows_and_consistency():
    a = ev.analyse(ev.Settings())
    assert [r.name for r in a.rows] == list(E.NAMES)
    for r in a.rows:
        assert r.lo <= r.hi and r.error == pytest.approx(r.estimate - r.truth) and r.error_ate == pytest.approx(r.estimate - a.ate)
        assert r.covers == (r.lo <= r.truth <= r.hi)
        assert (r.limit is not None) == (r.name in ev.REGRESSION_KIND)
    assert a.ate == pytest.approx(0.5) and a.att == pytest.approx(0.5)
    assert a.treated_n == int(a.t.sum()) and len(a.e) == a.settings.n and a.ess > 0 and 0 <= a.outside <= 1
    assert a.share == pytest.approx(0.5)
    assert next(r for r in a.rows if r.name == "Regression (linear)").limit == pytest.approx(0.5, abs=1e-9)


def test_analysis_is_deterministic_and_the_seed_matters():
    a, b = ev.analyse(ev.Settings(n=200)), ev.analyse(ev.Settings(n=200))
    assert [r.estimate for r in a.rows] == [r.estimate for r in b.rows] and [r.lo for r in a.rows] == [r.lo for r in b.rows]
    assert ev.analyse(ev.Settings(n=200, seed=36)).rows[1].estimate != a.rows[1].estimate


def test_analysis_at_the_slider_edges_does_not_raise():
    for kw in (dict(gamma=C.GAMMA_MIN), dict(gamma=C.GAMMA_MAX, n=C.N_MIN), dict(offset=C.OFFSET_MAX, delta=C.DELTA_MAX, curve=C.CURVE_MAX, tau1=C.TAU1_MAX),
               dict(k=C.K_MAX, trim=C.TRIM_MAX), dict(k=C.K_MIN, trim=C.TRIM_MIN, ps_quad=True)):
        a = ev.analyse(ev.Settings(n=C.N_MIN if kw.get("n") else 300, **{k: v for k, v in kw.items() if k != "n"}))
        assert all(np.isfinite(r.estimate) or r.name == "Gewichten (gekappt)" for r in a.rows)


def test_analysis_matches_estimate_all_on_the_same_sample():
    from kausal_generator import SplitMix64

    s = ev.Settings(gamma=1.2, curve=0.4, n=250, k=7, trim=0.05, ps_quad=True)
    x, t, y = (v[0] for v in W.simulate(s.world, s.n, SplitMix64(s.seed)))
    direct = E.estimate_all(x, t, y, 7, 0.05, True)
    a = ev.analyse(s)
    assert {r.name: r.estimate for r in a.rows} == pytest.approx(direct)


def test_run_scenario_structure_and_reproducibility():
    r = ev.run_scenario(W.World(gamma=0.8), 200, 30, 5)
    assert set(r["estimators"]) == set(E.NAMES) and r["n"] == 200 and r["reps"] == 30 and r["ate"] == pytest.approx(0.5)
    again = ev.run_scenario(W.World(gamma=0.8), 200, 30, 5)
    assert r["estimators"] == again["estimators"] and r["ess"] == again["ess"]
    for v in r["estimators"].values():
        assert v["rmse"] >= abs(v["bias"]) - 1e-12 and v["sd"] >= 0
    only = ev.run_scenario(W.World(gamma=0.8), 200, 10, 5, only=("Regression (linear)",))
    assert set(only["estimators"]) == {"Regression (linear)"}


def test_scenarios_experiment_small():
    res = ev.scenarios_experiment(n=150, reps=12)
    assert list(res) == list(C.SCENARIOS)
    assert res["S4 knappe Überlappung"]["outside"] > res["S1 Grundfall"]["outside"] and res["S4 knappe Überlappung"]["ess"] < res["S1 Grundfall"]["ess"]


def test_overlap_experiment_small():
    rows = ev.overlap_experiment((0.2, 2.5), reps=20, n=200)
    assert [r["gamma"] for r in rows] == [0.2, 2.5] and rows[0]["ess"] > rows[1]["ess"] and rows[0]["outside"] < rows[1]["outside"]


def test_strata_and_trim_experiments_small():
    s = ev.strata_experiment((1, 5, 20), reps=40, n=300)
    assert [r["k"] for r in s] == [1, 5, 20] and s[0]["bias"] > s[1]["bias"] > s[2]["bias"] - 0.03
    t = ev.trim_experiment((0.0, 0.1), reps=30, n=300)
    assert set(t) == {"Grundfall", "heterogen, Versatz"}
    het = t["heterogen, Versatz"]
    assert het[1]["target"] < het[0]["target"] - 0.2 and abs(het[1]["bias"]) < abs(het[1]["bias_ate"])


def test_coverage_experiment_small():
    res = ev.coverage_experiment(reps=8, boot=15, n=200)
    for label in ("Grundfall", "knappe Überlappung"):
        assert set(res[label]) == set(ev.COVER_NAMES) | {"reps", "boot"}
        for name in ev.COVER_NAMES:
            assert 0.0 <= res[label][name]["cover"] <= 1.0 and res[label][name]["width"] > 0
