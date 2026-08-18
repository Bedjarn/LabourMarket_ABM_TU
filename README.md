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

- `model/` — Python source (core ABM, dashboard, runners, analysis scripts)
- `data/` — input spreadsheets + saved experiment workbooks (`data/Experiments/`)
- `notebooks/` — verification and validation notebooks, incl. saved figures
- `scripts/` — helper scripts used to generate thesis figures
- `thesis.pdf` — the thesis manuscript this code accompanies

The model is a single script, `model/labour_market_model.py`, organised by the
seven conceptual components (C1–C7) and the macro-feedback layer (M1) of the
thesis. Every code block carries the identifier of its component as an inline
comment, so all code belonging to one component can be found by searching for
its tag. Appendix A of `thesis.pdf` documents the implementation; appendices C
and F cover the calibration and the full parameter table.

---

## Experiments

The eight experiments reported in chapter 6 are saved as workbooks in
`data/Experiments/`. They are produced by the multi-run Experimenter tab of the
dashboard; `model/multirun_experiments.py` holds that logic (it is a module, not
a stand-alone script).

| # | Workbook suffix | What it varies |
|---|---|---|
| E1 | `baseline_meanfield_vs_ulc` | Forward-looking NPV baseline vs. the myopic ULC rule |
| E2 | `meanfield_off_vs_ulc` | Contribution of the forward-looking decision logic alone |
| E3 | `employment_protection_on_off` | Contribution of the Dutch employment-protection institutions |
| E4 | `ai_cost_trajectory` | The AI cost curve (falling, flat, rising) |
| E5 | `nr_multiplier_comparative_advantage` | AI's comparative advantage on non-routine vs. routine tasks |
| E6 | `minimum_wage` | The statutory wage floor |
| E7 | `severance` | The transitievergoeding rate |
| E8 | `transition_probability_pconvert` | Conversion probability at the chain-regulation limit |

Verification and validation are in `notebooks/`, reproducing appendices D and E
of the thesis; the saved figures are in
`notebooks/validation/validation_figures/`.

---

## Reproducing the results

```bash
python model/run_live_dashboard.py          # dashboard; the Experimenter tab reruns E1–E8
python model/run_model.py                   # all four adoption modes, one seed, one parameter set
python model/analysis/OFAT_sensitivity.py   # one-factor-at-a-time sensitivity sweep
```

Runs are seeded, so the reported results reproduce exactly on the pinned
dependency versions in `requirements.txt`.

---

## Also available

The source code is mirrored at
<https://github.com/Bedjarn/LabourMarket_ABM_TU>. The LaTeX source of the
manuscript and internal working documents are kept separately and are not part
of this deliverable.
