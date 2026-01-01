# FINWELL_LAB
**A Hitchhiker’s Guide to Financial Wellbeing**  
**Decoding the Fin-Well Protocol.**

> **Assumptions first. Repro or it didn’t happen. Robustness beats optimism.**

This repository is the **lab**: code, notebooks, tests, and data handling that back the Ferropolis docs site (under construction).  
No sales. No hype. No “trust me”. **You can run it. You can break it. You can verify it.**

---

## What this is
- **Reproducible financial modeling** for consumers
- **Transparent assumptions** (explicit parameter sheets)
- **Auditable results** (sensitivity, stress, failure modes)
- **Community review** (issues/PRs + discussions)

## What this is not
- Investment / tax / legal advice  
- Product pitches, affiliate lists, “top ETF 2026” content  
- Magic spreadsheets without provenance

---

## The protocol (in one breath)
**Question → Data → Assumptions → Model → Sensitivity → Stress → Repro → Plain-language summary.**

---

## Minimum Repro Standard (MRP)

Jede Analyse muss Folgendes liefern — sonst ist sie **Draft**.

### 1) Parameter (Assumptions Sheet)
- vollständiger Parameterblock (Defaults + Einheiten + kurze Erklärung)
- klare Trennung: **Input** vs. **abgeleitete Parameter**
- Annahmen explizit: Rendite (real/nominal), Inflation, Kosten, Steuern (falls modelliert), Entnahmeregel

### 2) Daten-Provenienz
- Quelle(n) + Zeitraum + Lizenz
- exakte Referenz auf **Snapshot/Version** (Hash/Tag/Datum)
- dokumentierte Daten-Checks (Missingness, Outlier, Revisionen)

### 3) Reproduzierbarkeit
- Seed / Random State (falls stochastisch)
- Environment festgepinnt (`requirements.txt`, `poetry.lock`, `package-lock.json`, etc.)
- **One-command run** (oder klarer “Run all”-Pfad)

### 4) Outputs (Pflicht)
- mindestens **1 Tabelle** (CSV/Parquet/Markdown)
- mindestens **1 Plot** (PNG/SVG)
- **Summary (3–7 Sätze)** in Alltagssprache:
  - Ergebnis als Bandbreite (nicht Punktwert)
  - **robust vs. fragil**
  - dominierende Annahmen nennen
  - wichtigste Sensitivitäten/Schwellenwerte

### 5) Artifacts & Links
- reproduzierbarer Run erzeugt Outputs in definierten Pfaden (`outputs/…`)
- Link auf Code/Notebook + **Commit-Hash**
- Changelog-Eintrag, wenn Ergebnis/Annahmen sich ändern

### Definition “Draft”
Eine Analyse gilt als **Draft**, wenn irgendein Punkt oben fehlt oder unklar ist.
