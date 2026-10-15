# Schätzen: vier Wege zur Zahl – Streamlit-Demo

Sechstes Stück der Linie **Kausale Diagramme und Adjustierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Nachfolger von [adjustierungsmenge-demo](https://github.com/sebastian-hanisch/adjustierungsmenge-demo): Dort stand, welche Mengen gültig sind und welche die genaueste ist. Hier steht, wie man eine feste Menge in eine Schätzung übersetzt, und welche zweite Annahme jeder Weg verlangt.

## Kernfrage

Dieselbe Adjustierung (ein stetiger Störfaktor, binäre Behandlung), vier Wege zur Zahl: **Regression**, **Gewichten** mit dem Neigungswert, **Schichten** und **Matching**. Welcher trifft die Wirkung, wenn das Ergebnis gekrümmt ist, die Zuteilung asymmetrisch, die Wirkung heterogen oder die Überlappung knapp? Und decken die Intervalle?

## Modell

- **Vehikel:** ein Standort wählt die neue Regel B (T = 1) abhängig von der Schwankung X; Ergebnis ist der Servicegrad Y. X ~ N(0, 1); P(T = 1 | X) = Φ(γ X + δ (X² − 1) + Versatz); Y = 0,6 X + Krümmung · (X² − 1) + (0,5 + τ₁ X) T + Rauschen. Wahrheit **ATE = 0,50**; der ATT hängt von τ₁ und der Zuteilung ab.
- **Wahrheit** in geschlossener Form (Gauss-Hermite-Quadratur), kein Simulationsrauschen. Der Grenzwert der Regression bei n → ∞ wird ebenfalls aus den Momenten gerechnet.
- **Daten:** `SplitMix64`-Ströme (Generator `kausal_generator.py`, wortgleich in allen Stücken der Linie).

## Methodik

- **Schätzer** (`sch_estimators.py`, nur numpy): naiver Vergleich; Regression (linear, mit Wechselwirkung, quadratisch); Gewichten (Horvitz-Thompson, normiert, gekappt auf c < e < 1 − c); Schichten nach dem Neigungswert (K einstellbar); Matching (nächster Nachbar im Neigungswert mit Zurücklegen, ATE und ATT). Neigungswert: logistische Regression nach Newton, auf X oder (1, X, X²).
- **Zielgröße:** ATE; Matching (ATT) gegen den ATT, gekapptes Gewichten gegen den ATE auf der Stützung (die Wirkung auf Einheiten mit c < e < 1 − c).
- **Intervalle:** Perzentil-Bootstrap mit neu angepasstem Neigungswert je Ziehung.
- **Experimente** laufen über feste Zufallsströme.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. n = 500, 500 feste Wiederholungen, Verzerrung = Mittel minus Zielgröße; ATE 0,50.

| Frage | Befund | Test |
|---|---|---|
| Grundfall (γ = 0,8, lineares Ergebnis) | Der naive Vergleich liegt um **+0,60** daneben. Regression 0,00 (SD 0,105), Gewichten +0,03 (SD 0,124), Schichten (K = 5) +0,07 (SD 0,110), Matching +0,02 (SD 0,138); effektive Stichprobe 314 von 500. | `test_scenario_s1_base_case` |
| Krümmung und asymmetrische Zuteilung (S2b) | Lineare Regression **−0,42**, mit Wechselwirkung **+0,51**, quadratisch 0,00; Gewichten +0,19, Matching +0,13, Schichten +0,08. | `test_scenario_s2b_curved_and_offset` |
| Heterogene Wirkung mit Versatz (S3b): ATE 0,50, ATT 0,72 | Die lineare Regression liegt bei **−0,42**: Sie schätzt eine gewichtete Wirkung (Grenzwert 0,085). Mit Wechselwirkung 0,00, Gewichten +0,05, Matching-ATT +0,04 gegen den ATT. Gekapptes Gewichten −0,38 gegen den ATE, aber −0,02 gegen seine Zielgröße (0,15). | `test_scenario_s3b_heterogeneous_effect` |
| Knappe Überlappung (S4, γ = 2,5) | Effektive Stichprobe **132** von 500, **51 %** der Einheiten mit Neigungswert unter 0,05 oder über 0,95. Normiertes Gewichten **+0,33**, Matching +0,16; Regression +0,01 und gekapptes Gewichten −0,01. | `test_scenario_s4_poor_overlap` |
| Knapp, gekrümmt, Versatz (S4c) | Lineare Regression **−0,70**, mit Wechselwirkung +0,60, quadratisch +0,01; Gewichten +0,40, Matching +0,32, Matching-ATT +0,73. | `test_scenario_s4c` |
| Gekrümmte Zuteilung, lineares Neigungsmodell (S5) | Normiertes Gewichten **−0,16**, HT −0,14, gekappt −0,09; Regression 0,00, Schichten +0,01, Matching +0,04. | `test_scenario_s5_wrong_propensity_model` |
| Wie verschwindet die Überlappung? (γ = 0,2 bis 3,0) | Effektive Stichprobe **485 → 131**, Anteil außerhalb **0 → 58 %**; Gewichten +0,01 → **+0,41**, Matching 0,00 → +0,16; lineare Regression bleibt zwischen −0,01 und +0,01, gekapptes Gewichten zwischen −0,02 und +0,01. | `test_experiment_2_overlap` |
| Wie viele Schichten? (Grundfall) | Verzerrung K = 1: +0,60; 2: +0,23; 3: +0,14; 5: +0,07; 10: +0,03; 20: +0,01; 40: +0,01; Streuung 0,10 (K = 1), 0,13 (K = 20), 0,12 (K = 40). | `test_experiment_3_strata` |
| Was kostet die Kappung? | Im Grundfall bleibt die Zielgröße 0,50 und die Verzerrung geht von +0,03 (ohne) auf 0,00 (c = 0,10). Bei heterogener Wirkung mit Versatz wandert die Schätzung mit der Zielgröße: 0,55 (c = 0) → 0,13 (c = 0,10) → −0,10 (c = 0,20); Zielgröße 0,50 → 0,15 → −0,06. | `test_experiment_3_trimming_changes_the_estimand` |
| Decken die Bootstrap-Intervalle? (100 Wiederholungen, B = 60) | Grundfall: Regression **91 %**, Gewichten 94 %, Matching 93 %. Knappe Überlappung: Regression 90 %, Gewichten **37 %**, Matching **86 %**. | `test_experiment_4_coverage` |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Positivität: jede Einheit könnte beides bekommen** | Bei knapper Überlappung brechen Gewichten und Matching ein; die Regression extrapoliert über die Lücke und ist dort nur so gut wie ihr Modell. | Wie falsch darf das Diagramm sein? |
| **Das Ergebnismodell stimmt (Regression)** | Mit Krümmung und asymmetrischer Zuteilung verfehlt die lineare Regression die Wirkung um mehr als die Hälfte. | Doppelt robust und Double ML |
| **Das Zuteilungsmodell stimmt (Gewichten)** | Mit gekrümmter Zuteilung und linearem Neigungsmodell ist das Gewichten verzerrt. | Doppelt robust und Double ML |
| **Alle Störfaktoren sind beobachtet** | Ein verborgener Störfaktor verzerrt jeden Schätzer; die Daten allein zeigen es nicht. | Frontdoor, Instrumente, Sensitivität und Falsifikation |
| **Ein stetiger Störfaktor** | Mit vielen Störfaktoren wird Matching in hohen Dimensionen dünn; die Demo hat einen. | nicht gebaut |
| **Einheiten beeinflussen sich nicht** | Interferenz zwischen Standorten wird nicht modelliert. | nicht gebaut |

Die Zahlen gelten für dieses erfundene Vehikel (ein Störfaktor, probit-Zuteilung, Normalrauschen, n = 500); „Wahrheit“ ist das simulierte Modell, nicht ein Befund über echte Standorte. Die Quellen im 📐-Expander sind aus dem Gedächtnis, nicht geprüft; die Aussage zum Bootstrap des Matching (Abadie, Imbens 2008) taucht hier nur als gemessene Unterdeckung auf.

## Bezug zu den Nachbarn im Portfolio

- [off-policy-demo](https://github.com/sebastian-hanisch/off-policy-demo): IPS, SNIPS, Direkte Methode und Doppelt robust mit **bekannter** Protokoll-Wahrscheinlichkeit und diskreten Aktionen. Hier ist der Neigungswert **geschätzt**, der Störfaktor stetig, und es gibt Schichten und Matching; Doppelt robust bleibt dem nächsten Stück.
- [adjustierungsmenge-demo](https://github.com/sebastian-hanisch/adjustierungsmenge-demo): wählt die Menge nach der Varianz der Regression; hier ist die Menge fest und der Schätzer die Frage.
- [cuped-demo](https://github.com/sebastian-hanisch/cuped-demo): Regression auf eine Vorab-Kovariate bei Zufallszuteilung (Varianzreduktion); hier Konfundierung bei beobachteter Zuteilung.

## Tests

Mini-Instanzen von Hand (binärer Störfaktor: naive Differenz 2,2, Schichten, Regression, Gewichten und Matching 1; effektive Stichprobe 3,2; Matching-Partner und Gleichstand), Regression gegen `numpy.linalg.lstsq`, Gewichten und Matching gegen Brute Force, Wahrheit (ATT, Kappung) gegen Monte Carlo mit einer Million Einheiten, Grenzwert der Regression gegen eine große Stichprobe, Scorebedingungen der logistischen Regression, Auswertung, jede README-Zahl in `test_claims.py`, AppTest (Presets, Regler, Permalink, PDF), Footer und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `kausal_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Normalverteilung, Strukturmodell) |
| `sch_world.py` | Vehikel: Zuteilung, Ergebnis, ATE, ATT, Kappung, Grenzwert der Regression |
| `sch_estimators.py` | Neigungswert, Regression, Gewichten, Schichten, Matching, Bootstrap |
| `sch_evaluation.py` | Einstellungen, Analyse, vier Experimente, Verdikte |
| `sch_visualization.py`, `sch_pdf_export.py` | Figuren, Tabellen, PDF |
| `sch_constants.py`, `sch_presets.py` | Regler, Szenarien, Presets, Permalink |
| `tests/` | Mini-Instanzen, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Doppelt robuster Schätzer und Cross-Fitting:** eigenes Stück der Linie; hier nur die Einzelschätzer.
- **Kernel-, Caliper-, Genetic- und Optimal-Matching:** eigener Aufwand ohne neue Aussage; hier nächster Nachbar mit Zurücklegen.
- **Mehrere Störfaktoren, verborgene Störfaktoren, Interferenz:** nicht gebaut; die Demo hat einen beobachteten Störfaktor.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
