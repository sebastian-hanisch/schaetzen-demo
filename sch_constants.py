"""Konstanten: Regler-Grenzen, feste Szenarien und Experimente, Farben und Presets."""

# Rechenaufwand am Maximum: n = 5 000, Bootstrap B = 200 mit logistischer Regression je Ziehung, wenige Sekunden (zwischengespeichert).
N_MIN, N_MAX, N_STEP, DEFAULT_N = 100, 5000, 100, 500
SEED_MAX, DEFAULT_SEED = 999999, 35
GAMMA_MIN, GAMMA_MAX, GAMMA_STEP, DEFAULT_GAMMA = 0.0, 3.0, 0.1, 0.8      # Zuteilungsstärke
OFFSET_MIN, OFFSET_MAX, OFFSET_STEP = -1.5, 1.5, 0.1                      # Versatz der Zuteilung (Anteil Behandelter)
DELTA_MIN, DELTA_MAX, DELTA_STEP = 0.0, 1.5, 0.1                          # gekrümmter Zuteilungsterm
CURVE_MIN, CURVE_MAX, CURVE_STEP = 0.0, 1.2, 0.1                          # Krümmung des Ergebnisses
TAU1_MIN, TAU1_MAX, TAU1_STEP = 0.0, 1.2, 0.1                             # Effektheterogenität
K_MIN, K_MAX, DEFAULT_K = 1, 40, 5                                        # Schichten
TRIM_MIN, TRIM_MAX, TRIM_STEP, DEFAULT_TRIM = 0.0, 0.25, 0.01, 0.10      # Kappung des Neigungswerts auf [c, 1 - c]
BOOT_B = 200                                                             # Bootstrap-Ziehungen je Einstellung
Z_CRIT = 1.959964

TAU0, BETA, SIGMA = 0.5, 0.6, 1.0           # wahre mittlere Wirkung, Steigung des Ergebnisses im Störfaktor, Rauschen
PS_CLIP = 1e-4                               # Neigungswert wird auf [PS_CLIP, 1 - PS_CLIP] begrenzt (Division)
HIST_BINS = 20

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 960000
EXP_N, EXP_REPS = 500, 500                   # Experiment 1: Szenarien
EXP_OVERLAP_REPS = 300                       # Experiment 2: Zuteilungsstärke
EXP_OVERLAP_GAMMAS = (0.2, 0.5, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0)
EXP_K_VALUES = (1, 2, 3, 5, 10, 20, 40)     # Experiment 3: Schichtenzahl
EXP_TRIM_VALUES = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20)
EXP_COVER_REPS, EXP_COVER_B = 100, 60        # Experiment 4: Bootstrap-Abdeckung (je Wiederholung B Ziehungen)

# Szenarien des Experiments 1: Name -> Parameter der Welt (Rest Standard).
SCENARIOS = {
    "S1 Grundfall": dict(gamma=0.8),
    "S2 gekrümmtes Ergebnis, symmetrische Zuteilung": dict(gamma=0.8, curve=0.8),
    "S2b gekrümmtes Ergebnis, Versatz": dict(gamma=0.8, curve=0.8, offset=0.8),
    "S3 heterogener Effekt, symmetrische Zuteilung": dict(gamma=0.8, tau1=0.8),
    "S3b heterogener Effekt, Versatz": dict(gamma=0.8, tau1=0.8, offset=0.8),
    "S4 knappe Überlappung": dict(gamma=2.5),
    "S4c knappe Überlappung, gekrümmt, Versatz": dict(gamma=2.5, curve=0.8, offset=0.8),
    "S5 gekrümmte Zuteilung, Neigungsmodell linear": dict(gamma=0.8, delta=1.0),
}

COLOR_TRUTH = "#D68A2E"
COLOR_GOOD = "#2F6B65"
COLOR_BAD = "#c0392b"
COLOR_TREATED = "#4c78a8"
COLOR_CONTROL = "#8a8a8a"
COLOR_NEUTRAL = "#8a8a8a"


def preset(**kw) -> dict:
    base = dict(gamma=DEFAULT_GAMMA, offset=0.0, delta=0.0, curve=0.0, tau1=0.0, k=DEFAULT_K, trim=DEFAULT_TRIM, n=DEFAULT_N, seed=DEFAULT_SEED, ps_quad=False)
    base.update(kw)
    return base


PRESETS = {
    "Grundfall (Voreinstellung)": preset(),
    "Krümmung und Versatz": preset(curve=0.8, offset=0.8),
    "Heterogene Wirkung": preset(tau1=0.8, offset=0.8),
    "Knappe Überlappung": preset(gamma=2.5),
    "Zuteilung gekrümmt": preset(delta=1.0),
    "Zuteilung gekrümmt, Modell richtig": preset(delta=1.0, ps_quad=True),
}
PRESET_HELP = {
    "Grundfall (Voreinstellung)": "Moderate Zuteilung, lineares Ergebnis, konstante Wirkung: Alle Schätzer treffen, der naive Vergleich liegt um 0,6 daneben.",
    "Krümmung und Versatz": "Das Ergebnis ist im Störfaktor gekrümmt und die Zuteilung asymmetrisch: Die lineare Regression und die mit Wechselwirkung verfehlen die Wirkung stark.",
    "Heterogene Wirkung": "Die Wirkung hängt vom Störfaktor ab: Die lineare Regression schätzt eine gewichtete Wirkung (mit p(1 − p)), nicht den Durchschnitt.",
    "Knappe Überlappung": "Starke Zuteilung: Die Hälfte der Einheiten hat einen Neigungswert unter 0,05 oder über 0,95. Gewichten und Matching leiden, die Regression nicht.",
    "Zuteilung gekrümmt": "Die Zuteilung hängt quadratisch vom Störfaktor ab, das Neigungsmodell ist linear: Das Gewichten ist verzerrt, Regression und Schichten nicht.",
    "Zuteilung gekrümmt, Modell richtig": "Wie zuvor, aber das Neigungsmodell enthält den quadratischen Term: Das Gewichten ist wieder fast richtig, auf Kosten der Überlappung.",
}
