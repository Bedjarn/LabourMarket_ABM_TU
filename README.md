# Dutch Labour Market ABM — AI Automation, Wages & Employment

Agent-based model (ABM) of the Dutch labour market that studies how AI-driven task
automation affects employment, wages and wage inequality. Built with
[Mesa](https://mesa.readthedocs.io/) and a [Streamlit](https://streamlit.io/)
dashboard.

This package contains the model code, the input data, the saved results of the
eight thesis experiments, and the verification/validation notebooks. It is the
code companion to the MSc thesis (Engineering and Policy Analysis, TU Delft).

---

## Quick start

Python 3.10+ (developed on 3.12). From the package root:

**Windows:** double-click `run_live_dashboard.bat`. The launcher works independently
of the folder location, finds common Python/Conda installations, and offers to install
the pinned dependencies when they are missing.

**macOS, Linux, or command line:**

```bash
python -m pip install -r requirements.txt
python model/run_live_dashboard.py
```

The dashboard opens at **http://localhost:8501**. Equivalent direct launch:
`streamlit run model/live_dashboard.py`.

Dependencies are pinned in [requirements.txt](requirements.txt)
(`mesa`, `streamlit`, `numpy`, `pandas`, `matplotlib`, `openpyxl`; `scipy` for the notebooks).

---

## Loading the saved thesis experiments

The exact results reported in the thesis (Chapter 6) ship as Excel workbooks in
[data/Experiments/](data/Experiments/). There are two ways to use them.

**A. Open directly in Excel.** Each workbook is self-contained — the `summary_mean_sd`
sheet holds the headline outcomes (Gini, skill-wage premium, unemployment rate as
mean ± sd) and `mean_timeseries` / `per_run_timeseries` hold the full trajectories.

**B. Load into the dashboard** (for the interactive plots and the paired test):

1. Launch the dashboard and open the **Experimenter** tab.
2. Add one experiment to the queue (any configuration — this reveals the analysis panel).
3. In **Extended analysis → "Load previous extended-analysis Excel"**, upload one of
   the [data/Experiments/](data/Experiments/) workbooks and click **Load old extended analysis**.

The eight workbooks (50 seeds, 42–91; 960 steps = 80 years; mean-field NPV baseline):

| Workbook | Exp. | What it varies |
|----------|------|----------------|
| [E1 baseline vs ULC](data/Experiments/extended_analysis_E1_baseline_meanfield_vs_ulc.xlsx) | E1 | Mean-field NPV baseline vs. Upreti & Sridhar ULC logic |
| [E2 MF-off vs ULC](data/Experiments/extended_analysis_E2_meanfield_off_vs_ulc.xlsx) | E2 | Decomposition: forward-looking NPV (protection off) vs. ULC |
| [E3 protection on/off](data/Experiments/extended_analysis_E3_employment_protection_on_off.xlsx) | E3 | Employment protection on vs. off |
| [E4 AI cost trajectory](data/Experiments/extended_analysis_E4_ai_cost_trajectory.xlsx) | E4 | AI rental-cost path: declining / flat / rising |
| [E5 comparative advantage](data/Experiments/extended_analysis_E5_nr_multiplier_comparative_advantage.xlsx) | E5 | Non-routine AI multiplier: 0.50 / 0.85 / 1.15 / 1.50 |
| [E6 minimum wage](data/Experiments/extended_analysis_E6_minimum_wage.xlsx) | E6 | Minimum wage `w_min`: 0 / 10 / 14.71 / 20 |
| [E7 severance](data/Experiments/extended_analysis_E7_severance.xlsx) | E7 | Severance rate: 0.0 / 0.33 / 1.0 month per tenure year |
| [E8 transition probability](data/Experiments/extended_analysis_E8_transition_probability_pconvert.xlsx) | E8 | Flex→permanent conversion `p_convert`: 0 / 0.25 / 0.5 / 0.75 / 1.0 |

---

## Package contents

| Path | What it is |
|------|------------|
| [model/](model/) | All Python source code (see below) |
| [data/](data/) | Input data + saved experiment results |
| [notebooks/](notebooks/) | Verification, validation and data-extraction notebooks |
| [requirements.txt](requirements.txt) | Pinned dependencies |
| [scripts/](scripts/) | Auxiliary helper scripts — *not part of the thesis pipeline* |

### [model/](model/)

| File | Role |
|------|------|
| [labour_market_model.py](model/labour_market_model.py) | Core ABM: `Worker`, `Producer`, `Task`, `LabourMarketModel` |
| [base_parameters.py](model/base_parameters.py) | Default parameter set (`BASE_PARAMS`) — copy before mutating |
| [live_dashboard.py](model/live_dashboard.py) | Streamlit dashboard frontend |
| [run_live_dashboard.py](model/run_live_dashboard.py) | Dashboard launcher |
| [dashboard_utils.py](model/dashboard_utils.py) | Simulation runners, figure builders, run I/O |
| [run_model.py](model/run_model.py) | Headless batch run of the four adoption modes |
| [multirun_experiments.py](model/multirun_experiments.py) | Multi-seed engine behind the Extended-analysis panel |
| [trajectory_stats.py](model/trajectory_stats.py) | Paired permutation test over full trajectories |
| [analysis/OFAT_sensitivity.py](model/analysis/OFAT_sensitivity.py) | One-factor-at-a-time sensitivity sweep |
| [analysis/export_ofat_sensitivity_to_xlsx.py](model/analysis/export_ofat_sensitivity_to_xlsx.py) | Excel export of the OFAT sweep |
| [analysis/calibrate_validate.py](model/analysis/calibrate_validate.py) | Calibration/validation helper |

Generated run output is written to `model/outputs/` at run time (created on demand; not shipped).

### [data/](data/)

- [AI_Exposure.xlsx](data/AI_Exposure.xlsx), [4digits_with_tasks.xlsx](data/4digits_with_tasks.xlsx),
  [tasks_by_4digit.xlsx](data/tasks_by_4digit.xlsx), [tasks_extracted.xlsx](data/tasks_extracted.xlsx)
  — occupation/task exposure inputs (Gmyrek et al.).
- [Experiments/](data/Experiments/) — the eight saved experiment workbooks (table above).

### [notebooks/](notebooks/)

- [verification/verification_tests.ipynb](notebooks/verification/verification_tests.ipynb) — mechanical unit/conservation/edge-case tests.
- [validation/validation_tests.ipynb](notebooks/validation/validation_tests.ipynb) — pattern-oriented validation (six stylised facts); write-up in [validation_tests.docx](notebooks/validation/validation_tests.docx); generated figures in [validation/validation_figures/](notebooks/validation/validation_figures/).
- [Gmyrek_extraction.ipynb](notebooks/Gmyrek_extraction.ipynb) — builds the AI-exposure inputs in `data/`.

---

## Other ways to run

**Headless batch** — runs all four adoption modes, exports figures + Excel:

```bash
cd model
python run_model.py
```

**OFAT sensitivity sweep** — varies each parameter around the baseline:

```bash
cd model
python analysis/OFAT_sensitivity.py
```

**Dashboard tabs:** `Dashboard` · `Saved runs` · `Experimenter` · `Sensitivity` ·
`OFAT Explorer` · `Setup previews` · `Metric guide` · `Adoption mode info`.
The `Setup previews` tab (productivity, cost comparison, automation frontier, NPV
waterfall/heatmap) is useful for sanity-checking parameters before running.

---

## Model in brief

Firms decide **task by task** whether to automate. Output is a Leontief combination
of tasks; human productivity rises with task complexity (comparative advantage on
non-routine tasks), AI productivity falls with complexity. Displacement lowers the
wage bill, which lowers aggregate demand and price (a Kaleckian two-class feedback),
which feeds back into firms' profitability and further automation.

The central design choice is the adoption rule, selected at run time via `adoption_mode`:

| Mode | Decision rule |
|------|--------------|
| `ulc` | Automate whenever AI unit cost < cheapest human unit cost (myopic, instantaneous) |
| `npv_naive` | NPV with constant wage expectations over the horizon `T` |
| `npv_adaptive` | NPV with extrapolated recent wage trend |
| `npv_mean_field` | NPV anticipating displacement-driven wage softening (thesis baseline) |

Full parameter definitions and defaults are in
[base_parameters.py](model/base_parameters.py); their calibration and the model
equations are documented in the thesis (conceptual model, implementation, and the
parameter appendices).

**Statistical comparison.** Configurations are compared over their whole trajectory
with a paired permutation test (common random numbers across seeds), reporting a
family-wise-corrected p-value, a simultaneous confidence band, and the effect size
`d_z`. Implementation: [trajectory_stats.py](model/trajectory_stats.py).

---

## Not included in this package

The thesis manuscript (LaTeX source / PDF) and internal working documents are kept
separately and are not part of this code deliverable.
#   L a b o u r M a r k e t _ A B M 
 
 #   L a b o u r M a r k e t _ A B M _ T U  
 #   L a b o u r M a r k e t _ A B M _ T U  
 