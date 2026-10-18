"""Schätzen: Regression, Gewichten, Schichten, Matching - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Sechstes Stück der Linie Kausale Diagramme und Adjustierung. Dieselbe Adjustierung (ein stetiger Störfaktor), vier Wege zur Zahl: Regression, Gewichten mit dem
Neigungswert, Schichten, Matching. Das Vehikel steht in sch_world.py (Wahrheit numerisch exakt per Quadratur), die Schätzer in sch_estimators.py.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import sch_constants as C
from sch_evaluation import (
    COVER_NAMES,
    Settings,
    analyse,
    coverage_experiment,
    de,
    overlap_experiment,
    scenarios_experiment,
    signed,
    strata_experiment,
    trim_experiment,
    verdict_coverage,
    verdict_instance,
    verdict_overlap,
    verdict_scenarios,
)
from sch_pdf_export import build_pdf
from sch_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from sch_visualization import (
    SHORT,
    build_coverage_chart,
    build_diagram,
    build_estimates_chart,
    build_estimates_table,
    build_overlap_chart,
    build_overlap_histogram,
    build_scenario_chart,
    build_scenarios_table,
    build_strata_chart,
    build_trim_chart,
    style_estimates_table,
    style_scenarios_table,
)

st.set_page_config(page_title="Schätzen – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"


@st.cache_data(show_spinner="Bootstrap rechnet …")
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _scenarios():
    return scenarios_experiment()


@st.cache_data(show_spinner=SPINNER)
def _overlap():
    return overlap_experiment()


@st.cache_data(show_spinner=SPINNER)
def _strata():
    return strata_experiment(), trim_experiment()


@st.cache_data(show_spinner=SPINNER)
def _coverage():
    return coverage_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🧮 Schätzen: vier Wege zur Zahl")
st.markdown(
    """
**Dieselbe Adjustierung, vier Wege zur Zahl, und jeder hat seine Annahme.** Dies ist das sechste Stück der Linie **Kausale Diagramme und Adjustierung**. Die Stücke davor haben
gesagt, welche Variablen man festhält; hier steht, wie man das in eine Schätzung übersetzt: **Regression** (ein Modell für das Ergebnis), **Gewichten** und **Schichten**
(ein Modell für die Zuteilung, der **Neigungswert**) und **Matching** (Vergleich mit ähnlichen Einheiten). Der Störfaktor ist stetig, die Behandlung binär, die Wahrheit steht
numerisch exakt (Quadratur) fest.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So liest man die Schätzer", expanded=True):
    st.markdown(
        """
1. **Regression.** Ergebnis auf Behandlung und Störfaktor: richtig, wenn die Form des Ergebnisses stimmt. Ohne Wechselwirkung schätzt sie bei heterogener Wirkung eine
   **gewichtete** Wirkung (die Gewichte hängen von der Zuteilung ab, im gesättigten Fall p(1 − p)), nicht den Durchschnitt.
2. **Gewichten.** Jede Einheit zählt mit dem Kehrwert ihrer Zuteilungswahrscheinlichkeit: richtig, wenn das Zuteilungsmodell stimmt, und instabil bei knapper Überlappung.
3. **Schichten.** Vergleich innerhalb von Gruppen gleichen Neigungswerts: je mehr Schichten, desto weniger Restverzerrung, desto weniger Einheiten je Schicht; die Annahme steckt im Neigungsmodell.
4. **Matching.** Jede Einheit bekommt die ähnlichste der anderen Gruppe als Partner: ohne Ergebnismodell (hier aber im geschätzten Neigungswert, also mit einem Zuteilungsmodell), und nur so gut wie die Ähnlichkeit, die es gibt.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Zuteilung der Regel B**")
    gamma = st.slider("Zuteilungsstärke γ", *bounds("gamma_slider"), step=C.GAMMA_STEP, key="gamma_slider",
                      help="Wie stark die Schwankung die Wahl der Regel B bestimmt. Hohe Werte heißen knappe Überlappung: Fast jeder Standort ist durch seine Schwankung festgelegt.")
    offset = st.slider("Versatz", *bounds("offset_slider"), step=C.OFFSET_STEP, key="offset_slider",
                       help="Verschiebt die Zuteilung: Positiv heißt mehr Standorte mit Regel B. Macht die Zuteilung asymmetrisch zum Mittel des Störfaktors.")
    delta = st.slider("Zuteilung gekrümmt (δ)", *bounds("delta_slider"), step=C.DELTA_STEP, key="delta_slider",
                      help="Quadratischer Term in der Zuteilung. Das Neigungsmodell ist linear, solange „quadratisch“ nicht eingeschaltet ist.")
    st.markdown("**Ergebnis**")
    curve = st.slider("Krümmung des Ergebnisses", *bounds("curve_slider"), step=C.CURVE_STEP, key="curve_slider",
                      help="Quadratischer Term in der Abhängigkeit des Servicegrads von der Schwankung.")
    tau1 = st.slider("Effektheterogenität τ₁", *bounds("tau1_slider"), step=C.TAU1_STEP, key="tau1_slider",
                     help="Die Wirkung der Regel B ist τ₀ + τ₁ · X: Sie hängt vom Störfaktor ab. Der ATE bleibt 0,5, der ATT weicht ab.")
    st.markdown("**Verfahren**")
    k = st.slider("Schichten K", *bounds("k_slider"), step=1, key="k_slider", help="Anzahl der Schichten nach dem Neigungswert (gleich große Gruppen).")
    trim = st.slider("Kappungsgrenze c", *bounds("trim_slider"), step=C.TRIM_STEP, key="trim_slider",
                     help="Beim gekappten Gewichten zählen nur Einheiten mit c < e < 1 − c. Das ändert die Schätzgröße: die Wirkung auf diese Einheiten.")
    ps_quad = st.toggle("Neigungsmodell mit quadratischem Term", key="psquad_toggle", help="Die logistische Regression für den Neigungswert bekommt X² dazu.")
    st.markdown("**Daten**")
    n = st.slider("Beobachtungen n", *bounds("n_slider"), step=C.N_STEP, key="n_slider", help="Stichprobengröße der simulierten Daten.")
    seed = st.number_input("Zufalls-Seed der Stichprobe", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Stichprobe würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"gamma_slider": round(float(gamma), 1), "offset_slider": round(float(offset), 1), "delta_slider": round(float(delta), 1), "curve_slider": round(float(curve), 1),
                   "tau1_slider": round(float(tau1), 1), "k_slider": int(k), "trim_slider": round(float(trim), 2), "n_slider": int(n), "seed_input": int(seed),
                   "psquad_toggle": bool(ps_quad)})

settings = Settings(round(float(gamma), 1), round(float(offset), 1), round(float(delta), 1), round(float(curve), 1), round(float(tau1), 1), int(k), round(float(trim), 2),
                    int(n), int(seed), bool(ps_quad))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
(c1, c2), (c3, c4) = st.columns(2), st.columns(2)
naive = a.rows[0]
c1.metric("Wahre Wirkung (ATE)", de(a.ate, 2), delta=f"ATT {de(a.att, 2)}", delta_color="off", help="Durchschnittliche Wirkung der Regel B auf alle Standorte (ATE) und auf die mit Regel B (ATT).")
c2.metric("Naiver Vergleich", signed(naive.estimate, 3), delta=f"Fehler {signed(naive.error, 2)}", delta_color="off",
          help="Mittel der Behandelten minus Mittel der Kontrollen, ohne Adjustierung.")
c3.metric("Effektive Stichprobe", de(a.ess, 0), delta=f"von n = {settings.n}", delta_color="off",
          help="(Σw)² / Σw² der Gewichte 1/e und 1/(1 − e): Je kleiner, desto weniger tragen die Daten für das Gewichten.")
c4.metric("Neigungswert außerhalb 0,05 bis 0,95", f"{de(100 * a.outside, 0)} %", delta=f"{a.treated_n} von {settings.n} behandelt", delta_color="off",
          help="Einheiten, die fast sicher behandelt oder fast sicher nicht behandelt werden: Für sie gibt es kaum Vergleichseinheiten.")

left, right = st.columns([1, 1.25])
with left:
    st.plotly_chart(build_diagram(), width="stretch", key="diagram_chart")
    st.plotly_chart(build_overlap_histogram(a), width="stretch", key="overlap_chart")
with right:
    st.plotly_chart(build_estimates_chart(a), width="stretch", key="estimates_chart")
    st.caption("Punkt: Schätzung mit 95-%-Perzentil-Bootstrap-Intervall (grün: deckt die Zielgröße, rot: deckt nicht). Orange Linie: ATE. Orange Raute: die eigene Zielgröße des "
               "Schätzers, wenn sie vom ATE abweicht (Matching ATT, gekappt).")
verdict(verdict_instance(a))
st.dataframe(style_estimates_table(build_estimates_table(a)), hide_index=True, width="stretch")
st.caption("„Grenzfall“ ist der Koeffizient der Regression bei unendlich vielen Beobachtungen, exakt aus der Wahrheit gerechnet; Gewichten, Schichten und Matching haben keine "
           "einfache Grenzform.")

if st.button("📄 PDF der aktuellen Auswertung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="schaetzen.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Alle Schätzer in acht festen Szenarien")
res1 = _scenarios()
st.caption(f"Je Szenario {C.EXP_REPS} feste Wiederholungen mit n = {C.EXP_N}, K = {C.DEFAULT_K}, Kappung {de(C.DEFAULT_TRIM, 2)}. Gezeigt ist die Verzerrung gegen die eigene "
           "Zielgröße des Schätzers (ATE; beim Matching ATT und beim gekappten Gewichten die Wirkung auf der Stützung). ESS: effektive Stichprobe; außerhalb: Anteil mit "
           "Neigungswert unter 0,05 oder über 0,95.")
st.dataframe(style_scenarios_table(build_scenarios_table(res1), res1), hide_index=True, width="stretch")
verdict(verdict_scenarios(res1))
pick = st.selectbox("Szenario im Diagramm", list(res1), key="scenario_pick")
st.plotly_chart(build_scenario_chart(res1[pick]), width="stretch", key="scenario_chart")
st.caption("Balken: Verzerrung gegen die eigene Zielgröße; Linie: Streuung der Schätzung über die Wiederholungen. Grün unter 0,1, orange ab 0,1, rot ab 0,25.")

st.subheader("🔬 Experiment 2: Wenn die Überlappung knapp wird")
res2 = _overlap()
st.caption(f"Zuteilungsstärke γ von {de(res2[0]['gamma'], 1)} bis {de(res2[-1]['gamma'], 1)}, sonst Grundfall, n = {C.EXP_N}, {C.EXP_OVERLAP_REPS} feste Wiederholungen je Wert.")
st.plotly_chart(build_overlap_chart(res2), width="stretch", key="overlap_exp_chart")
verdict(verdict_overlap(res2))

st.subheader("🔬 Experiment 3: Wie viele Schichten, wie viel Kappung?")
strata_rows, trim_res = _strata()
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Schichten K im Grundfall**")
    st.plotly_chart(build_strata_chart(strata_rows), width="stretch", key="strata_chart")
    st.caption("Mehr Schichten senken die Restverzerrung, die Streuung steigt leicht. Mit K = 1 ist es der naive Vergleich.")
with col_b:
    st.markdown("**Kappung c bei gleichem Zuteilungsmodell**")
    st.plotly_chart(build_trim_chart(trim_res), width="stretch", key="trim_chart")
    st.caption("Im Grundfall bleibt die Zielgröße gleich; bei heterogener Wirkung mit Versatz wandert sie mit der Kappung. Gestrichelt: die Wirkung auf der Stützung.")

st.subheader("🔬 Experiment 4: Decken die Intervalle?")
st.caption(f"Perzentil-Bootstrap, {C.EXP_COVER_REPS} feste Wiederholungen, B = {C.EXP_COVER_B} Ziehungen mit neu angepasstem Neigungswert, n = {C.EXP_N}. Das Experiment läuft erst auf "
           "Knopfdruck.")
if st.button("▶ Abdeckung berechnen", key="coverage_run"):
    st.session_state["coverage_ready"] = True
if st.session_state.get("coverage_ready"):
    res4 = _coverage()
    st.plotly_chart(build_coverage_chart(res4), width="stretch", key="coverage_chart")
    verdict(verdict_coverage(res4))
    st.dataframe([{"Szenario": label, "Schätzer": SHORT[name], "Deckung": f"{de(100 * res4[label][name]['cover'], 0)} %", "mittlere Breite": de(res4[label][name]["width"], 2)}
                  for label in ("Grundfall", "knappe Überlappung") for name in COVER_NAMES], hide_index=True, width="stretch")
    st.caption("Für das Bootstrap-Intervall des Matching ist ein Ausfall bekannt (Abadie und Imbens 2008, aus dem Gedächtnis); hier zeigt sich eine Unterdeckung beim Matching erst bei knapper Überlappung, und beim Gewichten bricht das Intervall mit den Gewichten ein.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Positivität: jede Einheit könnte beides bekommen** | Bei knapper Überlappung brechen Gewichten und Matching ein; die Regression extrapoliert über die Lücke und ist dort nur so gut wie ihr Modell (Experiment 1 und 2). | Wie falsch darf das Diagramm sein? |
| **Das Ergebnismodell stimmt (Regression)** | Mit Krümmung und asymmetrischer Zuteilung verfehlt die lineare Regression die Wirkung um mehr als die Hälfte (Experiment 1). | Doppelt robust und Double ML |
| **Das Zuteilungsmodell stimmt (Gewichten)** | Mit gekrümmter Zuteilung und linearem Neigungsmodell ist das Gewichten verzerrt (Szenario S5). | Doppelt robust und Double ML |
| **Alle Störfaktoren sind beobachtet** | Ein verborgener Störfaktor verzerrt die hier gezeigten Schätzer, sofern er Zuteilung und Ergebnis bewegt; die Daten allein zeigen es nicht. | Frontdoor, Instrumente, Sensitivität und Falsifikation |
| **Ein stetiger Störfaktor** | Mit vielen Störfaktoren wird Matching auf den Kovariaten dünn (Fluch der Dimension); ein Neigungswert-Matching verlagert das Problem in das Neigungsmodell. Die Demo hat einen Störfaktor. | nicht gebaut |
| **Einheiten beeinflussen sich nicht** | Interferenz zwischen Standorten wird nicht modelliert. | nicht gebaut |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** $X \sim N(0,1)$; $P(T=1\mid X) = \Phi(\gamma X + \delta(X^2-1) + a)$; $Y = \beta X + c\,(X^2-1) + (\tau_0 + \tau_1 X)\,T + \sigma\varepsilon$. Die **Zielgrößen**:
$\mathrm{ATE} = \mathbb E[\tau(X)] = \tau_0$ und $\mathrm{ATT} = \tau_0 + \tau_1\,\mathbb E[X \mid T=1]$, beide numerisch exakt per Quadratur gerechnet (der ATE ist $\tau_0$ selbst).

**Neigungswert.** $e(x) = P(T=1\mid X=x)$; die Demo schätzt ihn mit einer logistischen Regression auf $(1, X)$ oder $(1, X, X^2)$.

**Regression.** Koeffizient von $T$ in $Y \sim 1 + T + X$ (Wechselwirkung: $+ T\,X$, quadratisch: $+ X^2, T X^2$, $X$ zentriert). Ohne Wechselwirkung gewichtet er $\tau(x)$ mit
zuteilungsabhängigen Gewichten; bei gesättigtem Kovariatenmodell sind es $\mathrm{Var}(T \mid X=x)$ (Angrist 1998).

**Gewichten.** $\hat\tau_{\mathrm{IPW}} = \frac1n \sum_i \big(\tfrac{T_i Y_i}{e_i} - \tfrac{(1-T_i)Y_i}{1-e_i}\big)$; normiert: gewichtete Mittel je Gruppe mit $w_i = T_i/e_i$ bzw.
$(1-T_i)/(1-e_i)$. Gekappt: nur $c < e_i < 1-c$; die Schätzgröße ist dann $\mathbb E[\tau(X) \mid c < e(X) < 1-c]$.

**Schichten.** $\hat\tau_{\mathrm{S}} = \sum_k \frac{n_k}{n}\,(\bar Y_{k,1} - \bar Y_{k,0})$ über $K$ Schichten gleicher Größe im Neigungswert.

**Matching.** Für jede Einheit der Quellgruppe die Einheit der anderen Gruppe mit dem nächsten Neigungswert (mit Zurücklegen); ATE: Mittel über beide Gruppen, ATT: Mittel über
die Behandelten.

**Effektive Stichprobe.** $\mathrm{ESS} = (\sum_i w_i)^2 / \sum_i w_i^2$.

**Literatur.** P. Rosenbaum, D. Rubin, *The central role of the propensity score*, Biometrika 70, 1983; A. Abadie, G. Imbens, *Large sample properties of matching
estimators for average treatment effects*, Econometrica 74, 2006; *On the failure of the bootstrap for matching estimators*, Econometrica 76, 2008; J. Angrist, *Estimating the
labor market impact of voluntary military service using social security data*, Econometrica 66, 1998; R. Crump u. a., *Dealing with limited overlap*, Biometrika 96, 2009
(alle aus dem Gedächtnis, nicht geprüft).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Kausale Diagramme: von Konfundierung bis Wahl](https://sebastianhanisch.net/konzepte-kausale-diagramme.html)."
)
