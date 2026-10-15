"""PDF-Export der aktuellen Auswertung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from sch_evaluation import Analysis, de, signed, verdict_instance

_REPLACE = {"→": "->", "←": "<-", "–": "-", "—": "-", "−": "-", "€": "EUR", "≥": ">=", "≤": "<=", "„": '"', "“": '"', "×": "x", "γ": "gamma", "τ": "tau", "δ": "delta",
            "**": "", "²": "^2", "≈": "~"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Schätzen: vier Wege zur Zahl"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Zuteilungsstärke γ = {de(s.gamma, 1)}, Versatz {de(s.offset, 1)}, Krümmung der Zuteilung {de(s.delta, 1)}, Krümmung des Ergebnisses {de(s.curve, 1)}, "
                            f"Heterogenität τ1 = {de(s.tau1, 1)}; n = {s.n}, Seed {s.seed}. Alles simuliert. ATE {de(a.ate, 2)}, ATT {de(a.att, 2)}, "
                            f"effektive Stichprobe {de(a.ess, 0)} von {s.n}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    for r in a.rows:
        pdf.multi_cell(0, 4.6, _s(f"{r.name} ({r.estimand}): {de(r.estimate, 3)} [{de(r.lo, 2)}; {de(r.hi, 2)}], Zielgröße {de(r.truth, 3)}, Fehler {signed(r.error, 3)}, "
                                  f"{'deckt' if r.covers else 'deckt nicht'}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 4.5, _s(verdict_instance(a)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: schaetzen-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
