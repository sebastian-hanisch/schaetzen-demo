"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def binary_instance():
    """Mini-Instanz 1: binärer Störfaktor X, 20 Einheiten, Y = 1·T + 2·X ohne Rauschen.
    X = 1: 10 Einheiten, 8 behandelt (Y = 3), 2 Kontrollen (Y = 2). X = 0: 10 Einheiten, 2 behandelt (Y = 1), 8 Kontrollen (Y = 0).
    Handrechnung: Behandelte Mittel (8·3 + 2·1) / 10 = 2,6, Kontrollen Mittel (2·2 + 8·0) / 10 = 0,4: naive Differenz 2,2.
    Je Schicht X: Differenz 1. Neigungswert 0,8 bei X = 1 und 0,2 bei X = 0 (= die beobachteten Anteile): Gewichten 1; Matching auf X: Partner mit gleichem X, Differenz 1."""
    x = np.array([1.0] * 10 + [0.0] * 10)
    t = np.array([1.0] * 8 + [0.0] * 2 + [1.0] * 2 + [0.0] * 8)
    y = 1.0 * t + 2.0 * x
    e = np.where(x == 1.0, 0.8, 0.2)
    return x, t, y, e
