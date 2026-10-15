"""Presets, Reglergrenzen, Permalink-Raster und Konstanten."""

import pytest

import sch_constants as C
import sch_evaluation as ev
import sch_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for name, p in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            v = p[key]
            if key == "ps_quad":
                assert v in (True, False)
                continue
            assert spec.lo <= v <= spec.hi, (name, key)
            if spec.step and spec.caster is float:
                assert P.snap(v, spec) == pytest.approx(v), (name, key)


def test_first_preset_is_the_default_setting():
    p = C.PRESETS["Grundfall (Voreinstellung)"]
    assert ev.Settings() == ev.Settings(p["gamma"], p["offset"], p["delta"], p["curve"], p["tau1"], p["k"], p["trim"], p["n"], p["seed"], p["ps_quad"])
    assert all(P.SETTING_SPECS[sk].default == (False if sk == "psquad_toggle" else C.PRESETS["Grundfall (Voreinstellung)"][k]) or sk == "psquad_toggle"
               for k, sk in P.PRESET_KEYS.items())


def test_presets_show_different_pictures():
    out = {}
    for name, p in C.PRESETS.items():
        a = ev.analyse(ev.Settings(p["gamma"], p["offset"], p["delta"], p["curve"], p["tau1"], p["k"], p["trim"], 400, p["seed"], p["ps_quad"]))
        out[name] = {r.name: r.error for r in a.rows}
        out[name]["limit"] = next(r.limit for r in a.rows if r.name == "Regression (linear)") - a.ate        # Fehler der linearen Regression bei n → ∞
    assert abs(out["Krümmung und Versatz"]["limit"]) > 0.25 and abs(out["Grundfall (Voreinstellung)"]["limit"]) < 0.01
    assert abs(out["Heterogene Wirkung"]["limit"]) > 0.25
    assert abs(out["Knappe Überlappung"]["Gewichten (normiert)"]) > abs(out["Grundfall (Voreinstellung)"]["Gewichten (normiert)"])
    assert abs(out["Zuteilung gekrümmt, Modell richtig"]["Gewichten (normiert)"]) <= abs(out["Zuteilung gekrümmt"]["Gewichten (normiert)"]) + 0.1


def test_snap_clamps_and_rounds_to_the_grid():
    spec = P.SETTING_SPECS["gamma_slider"]
    assert P.snap(5.0, spec) == C.GAMMA_MAX and P.snap(-1.0, spec) == C.GAMMA_MIN and P.snap(0.83, spec) == pytest.approx(0.8)
    assert P.snap(0.85000001, spec) == pytest.approx(0.9)
    off = P.SETTING_SPECS["offset_slider"]
    assert P.snap(-1.46, off) == pytest.approx(-1.5) and P.snap(0.04, off) == pytest.approx(0.0)
    assert P.snap(999999.0, P.SETTING_SPECS["seed_input"]) == 999999 and P.snap(7, P.SETTING_SPECS["k_slider"]) == 7


def test_bounds_and_defaults():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("k_slider") == (C.K_MIN, C.K_MAX)
    assert C.DEFAULT_N in range(C.N_MIN, C.N_MAX + 1, C.N_STEP) and C.K_MIN <= C.DEFAULT_K <= C.K_MAX and C.TRIM_MIN <= C.DEFAULT_TRIM <= C.TRIM_MAX
    assert {s.url_param for s in P.SETTING_SPECS.values()} == {"g", "o", "d", "c", "t1", "k", "tr", "n", "seed", "pq"}


def test_experiment_constants_are_consistent():
    assert C.EXP_REPS >= 200 and C.EXP_COVER_B >= 40 and C.BOOT_B >= 100 and C.PS_CLIP < 0.01
    assert tuple(sorted(C.EXP_OVERLAP_GAMMAS)) == C.EXP_OVERLAP_GAMMAS and tuple(sorted(C.EXP_K_VALUES)) == C.EXP_K_VALUES and C.EXP_K_VALUES[0] == 1
    assert C.EXP_TRIM_VALUES[0] == 0.0 and len(C.SCENARIOS) == 8 and C.DEFAULT_GAMMA in C.EXP_OVERLAP_GAMMAS
