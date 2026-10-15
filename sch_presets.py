"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio). Alle Regler sind in jeder Einstellung wirksam und werden immer
gezeichnet; es gibt weder KEPT-Regler noch berechnete Grenzen."""

import random
from dataclasses import dataclass

import streamlit as st

import sch_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    step: float | None = None


SETTING_SPECS = {
    "gamma_slider": SettingSpec("g", float, C.DEFAULT_GAMMA, C.GAMMA_MIN, C.GAMMA_MAX, C.GAMMA_STEP),
    "offset_slider": SettingSpec("o", float, 0.0, C.OFFSET_MIN, C.OFFSET_MAX, C.OFFSET_STEP),
    "delta_slider": SettingSpec("d", float, 0.0, C.DELTA_MIN, C.DELTA_MAX, C.DELTA_STEP),
    "curve_slider": SettingSpec("c", float, 0.0, C.CURVE_MIN, C.CURVE_MAX, C.CURVE_STEP),
    "tau1_slider": SettingSpec("t1", float, 0.0, C.TAU1_MIN, C.TAU1_MAX, C.TAU1_STEP),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX, 1),
    "trim_slider": SettingSpec("tr", float, C.DEFAULT_TRIM, C.TRIM_MIN, C.TRIM_MAX, C.TRIM_STEP),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX, C.N_STEP),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX, 1),
    "psquad_toggle": SettingSpec("pq", int, 0, 0, 1, 1),
}
PRESET_KEYS = {"gamma": "gamma_slider", "offset": "offset_slider", "delta": "delta_slider", "curve": "curve_slider", "tau1": "tau1_slider", "k": "k_slider",
               "trim": "trim_slider", "n": "n_slider", "seed": "seed_input", "ps_quad": "psquad_toggle"}


def snap(value, spec: SettingSpec):
    """Auf das Raster des Reglers legen und in die Grenzen klemmen."""
    value = max(spec.lo, min(spec.hi, value))
    if spec.step and spec.caster is float:
        value = round(round((value - spec.lo) / spec.step) * spec.step + spec.lo, 6)
    return value


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = bool(spec.default) if state_key == "psquad_toggle" else spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    init_session_state_defaults()
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = snap(spec.caster(float(qp[spec.url_param])) if spec.caster is int else spec.caster(qp[spec.url_param]), spec)
        except (ValueError, TypeError):
            continue
        st.session_state[state_key] = bool(value) if state_key == "psquad_toggle" else value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values: dict):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(int(value) if state_key == "psquad_toggle" else value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
