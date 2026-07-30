# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An agent-based model (ABM) of the Dutch labour market simulating AI-driven automation effects on employment and wages. Built with Mesa. The central innovation is four distinct AI adoption decision modes that produce different macroeconomic dynamics.

## Running the Model

**Interactive dashboard (primary interface):**
```bash
cd model
python run_live_dashboard.py
# Access at http://localhost:8501
```

**Headless batch run (all 4 adoption modes, exports figures + Excel):**
```bash
cd model
python run_model.py
# Output: model/outputs/run_model_outputs/[timestamp]/
```

**Direct Python API:**
```python
from labour_market_model import LabourMarketModel
from base_parameters import BASE_PARAMS

params = BASE_PARAMS.copy()
params['adoption_mode'] = 'npv_mean_field'  # 'ulc' | 'npv_naive' | 'npv_adaptive' | 'npv_mean_field'
model = LabourMarketModel(**params)   # note: step count is not a model parameter
for _ in range(250):
    model.step()
df = model.datacollector.get_model_vars_dataframe()
```

**OFAT sensitivity analysis:**
```bash
cd model
python analysis/OFAT_sensitivity.py
```

## Architecture

### Core Files (`model/`)

| File | Purpose |
|---|---|
| `labour_market_model.py` | Main ABM (~1800 lines): `Worker`, `Producer`, `Task` classes + `LabourMarketModel` |
| `base_parameters.py` | Default parameter dict (~40 parameters) — always copy before mutating |
| `dashboard_utils.py` | Simulation runners, figure builders, run I/O helpers (~3400 lines) |
| `live_dashboard.py` | Streamlit dashboard with 6 pages (~6000 lines) |
| `run_live_dashboard.py` | Dashboard entry point |
| `run_model.py` | Batch headless runner |
| `multirun_experiments.py` | Multi-seed scenario experiment runner |
| `trajectory_stats.py` | Paired permutation-test / trajectory statistics helpers |
| `analysis/OFAT_sensitivity.py` | One-factor-at-a-time parallel sensitivity sweeps (`ProcessPoolExecutor`) |
| `analysis/export_ofat_sensitivity_to_xlsx.py` | Excel export of the OFAT sweep |
| `analysis/calibrate_validate.py` | Validation runs comparing baseline to alternatives |

### Agent Hierarchy

```
LabourMarketModel (mesa.Model)
├── Worker agents  (passive)
│   └── skill_level, employed, employer, wage, tenure, contract_type ('flex'|'vast')
└── Producer agents  (active decision-makers)
    └── tasks: List[Task]  (20 tasks per firm)
        └── task_type ('routine'|'non_routine'), complexity_index (1–20),
            automated, n_ai, employees
```

Agents are iterated manually in `LabourMarketModel.step()` — Mesa's scheduler is not used.

### Step Sequence (in order)

1. Decay AI rental cost: `k_ai = k_ai_floor + (k_ai_0 − k_ai_floor) × exp(−k_ai_decay × t)`
2. Compute lagged wage bill (Keynesian demand shifter)
3. Producers `produce()` — Leontief min over all 20 tasks
4. Set market price: `p = A − B×Q` where `A = A_base + γ×Y`
5. Chain-limit (ketenregeling): flex workers at tenure cap → convert or release
6. Exogenous separations (probability `delta` per employed worker)
7. Wage update: `w_new = λ × w_target + (1−λ) × w_old`, `w_target = a + b×L`
8. Smooth wage expectations (NPV-adaptive / NPV-mean-field modes only)
9. Two-pass job matching: preferred skill first, then cross-skill with `theta` tolerance
10. Tenure increments
11. Propagate wages to worker objects
12. Re-run production + pricing for end-of-step consistency
13. Single-pass stat computation (`_compute_all_stats()` → `_step_stats` cache)
14. Mesa `DataCollector` records ~50 metrics

### Four Adoption Modes

| Mode | Automation trigger |
|---|---|
| `ulc` | Automate if AI unit cost < cheapest human unit cost (instantaneous) |
| `npv_naive` | NPV positive with constant wage expectations over horizon `T` |
| `npv_adaptive` | NPV with extrapolated recent wage trend |
| `npv_mean_field` | NPV anticipating displacement-driven wage softening |

### Key Parameters (`base_parameters.py`)

| Group | Key params |
|---|---|
| Scale | `n_producers=20`, `n_high_skilled=200`, `n_low_skilled=250` |
| AI cost | `k_ai=40.0`, `k_ai_decay=0.020`, `k_ai_floor=20.0` |
| Wages | `a_h=15.0`, `b_h=0.10`, `a_l=6.0`, `b_l=0.06`, `lam=0.08`, `w_min=14.71` |
| NPV | `I_base=80.0`, `r_discount=0.007`, `T_horizon=36`, `eta_hurdle=0.15` |
| Demand | `A_base=100.0`, `gamma=0.05`, `gamma_pi=0.008`, `B=2.0` |
| Employment protection | `employment_protection=True`, `severance_rate=1/3`, `chain_limit=36`, `p_convert=0.5`, `init_share_vast=0.67` |
| Alpha (routine share) | `alpha_source='uniform'`, `alpha_min=0.25`, `alpha_max=0.75` |

Wages and monetary quantities are in EUR/hour-equivalent units (high-skilled target ≈ €35, low-skilled ≈ €21 at full employment); `r_discount` is a per-step (monthly) rate.

### Production Technology

- **Leontief**: output = min over all tasks of (task_output / requirement) — output is bottlenecked by the scarcest task
- **Human productivity**: exponential in task complexity (higher complexity → higher human advantage)
- **AI productivity**: declining in task complexity (routine tasks automated first)
- **Comparative advantage**: high-skill workers preferred on non-routine tasks; low-skill on routine

### Data

Input data lives in `data/`:
- `AI_Exposure.xlsx`: AI exposure scores by 4-digit occupation (Gmyrek et al.)
- `4digits_with_tasks.xlsx`, `tasks_by_4digit.xlsx`: Task breakdowns by occupation
- `Experiments/`: saved result workbooks for the eight reported experiments

Generated output is written to `model/outputs/` at run time (git-ignored).

## Dependencies

```
mesa, streamlit, matplotlib, numpy, pandas, openpyxl
```

Install: `pip install -r requirements.txt`
