# Dutch Labour Market ABM — AI Automation, Wages & Employment

Agent-based model (ABM) of the Dutch labour market that studies how AI-driven
task automation affects employment, wages and wage inequality. Built with
[Mesa](https://mesa.readthedocs.io/) and a [Streamlit](https://streamlit.io/)
dashboard.

This repository contains the model code, input data, saved results of the
thesis experiments, and the verification/validation notebooks. It is the code
companion to the MSc thesis (Engineering and Policy Analysis, TU Delft).

---

## Quick start

Requires Python 3.10+ (developed on 3.12). From the package root:

**Windows:** double-click `run_live_dashboard.bat` (launcher installs missing
dependencies when needed).

**macOS, Linux, or command line:**

```bash
python -m pip install -r requirements.txt
python model/run_live_dashboard.py
```

The dashboard opens at **http://localhost:8501**. Alternatively:
`streamlit run model/live_dashboard.py`.

Dependencies are pinned in [requirements.txt](requirements.txt)
(`mesa`, `streamlit`, `numpy`, `pandas`, `matplotlib`, `openpyxl`, etc.).

---

## Package layout

- `model/` — Python source (core ABM, dashboard, runners)
- `data/` — input spreadsheets + saved experiment workbooks
- `notebooks/` — verification and validation notebooks
- `scripts/` — helper scripts

See `model/` and `data/` for detailed descriptions and the experiment table.

---

## Not included

The thesis manuscript (LaTeX source/PDF) and internal working documents are
kept separately and are not part of this code deliverable.

If anything still looks odd on GitHub, tell me which section and I will adjust
the rendering or formatting.
