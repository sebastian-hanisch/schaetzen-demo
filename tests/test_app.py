"""End-to-end Smoke-Test via Streamlits AppTest: lädt app.py, klickt jeden Preset-Button, fährt die Regler an die Grenzen, startet das Abdeckungsexperiment und prüft,
dass kein Python-Fehler auftritt."""

import os

from streamlit.testing.v1 import AppTest

import sch_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 400


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _metrics(at):
    return {m.label: m.value for m in at.metric}


def test_app_loads_without_exception_and_shows_the_truth():
    at = _fresh_app()
    m = _metrics(at)
    assert m["Wahre Wirkung (ATE)"] == "0,50" and m["Naiver Vergleich"].startswith("+1,0")
    assert at.slider(key="gamma_slider").value == C.DEFAULT_GAMMA and at.toggle(key="psquad_toggle").value is False
    assert any("Der naive Vergleich liegt um" in s.value for s in list(at.success) + list(at.warning))


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_min_and_max_does_not_raise():
    for key in ("gamma_slider", "offset_slider", "delta_slider", "curve_slider", "tau1_slider", "k_slider", "trim_slider", "n_slider"):
        for edge in ("min", "max"):
            at = _fresh_app()
            s = at.slider(key=key)
            s.set_value(s.min if edge == "min" else s.max).run(timeout=TIMEOUT)
            assert not at.exception, (key, edge, [str(e) for e in at.exception])


def test_every_preset_is_applied_to_the_widgets():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.slider(key="gamma_slider").value == preset["gamma"] and at.slider(key="offset_slider").value == preset["offset"]
        assert at.slider(key="curve_slider").value == preset["curve"] and at.slider(key="tau1_slider").value == preset["tau1"]
        assert at.toggle(key="psquad_toggle").value == preset["ps_quad"]


def test_poor_overlap_preset_shows_the_warning():
    at = _fresh_app()
    at.button[[b.label for b in at.button].index("Knappe Überlappung")].click().run(timeout=TIMEOUT)
    assert any("Überlappung ist knapp" in w.value for w in at.warning)
    assert float(_metrics(at)["Effektive Stichprobe"].replace(",", ".")) < 250


def test_the_quadratic_switch_runs():
    at = _fresh_app()
    at.toggle(key="psquad_toggle").set_value(True).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_coverage_experiment_runs_on_click():
    at = _fresh_app()
    assert "coverage_ready" not in at.session_state
    at.button(key="coverage_run").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Im Grundfall decken die Bootstrap-Intervalle" in i.value for i in at.info)


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_permalink_values_are_taken_and_clamped():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["g"] = "9"
    at.query_params["o"] = "-0.46"
    at.query_params["k"] = "999"
    at.query_params["n"] = "50"
    at.query_params["pq"] = "1"
    at.query_params["seed"] = "abc"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="gamma_slider").value == C.GAMMA_MAX and at.slider(key="offset_slider").value == -0.5
    assert at.slider(key="k_slider").value == C.K_MAX and at.slider(key="n_slider").value == C.N_MIN
    assert at.toggle(key="psquad_toggle").value is True and at.number_input(key="seed_input").value == C.DEFAULT_SEED


def test_the_address_bar_mirrors_the_settings():
    at = _fresh_app()
    at.slider(key="curve_slider").set_value(0.6).run(timeout=TIMEOUT)
    assert at.query_params["c"] in ("0.6", ["0.6"])
