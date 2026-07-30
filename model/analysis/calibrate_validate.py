"""
Validation run: show dynamic trajectories of Y, p, employment, adoption
for the chosen baseline + a few alternatives. Confirms the Keynesian loop
actually moves quantities over time, and compares all three NPV modes.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labour_market_model import LabourMarketModel

warnings.filterwarnings("ignore")

N_STEPS = 240

BASE = dict(
    n_producers=20, n_high_skilled=200, n_low_skilled=250,
    delta=0.005, p_evaluate=0.15, seed=42, lam=0.25,
    a_h=2.5, b_h=0.04, a_l=1.5, b_l=0.025, w_min=0.0,
    a_prod=0.1, xi_prod=0.5,
    phi_base=2.0, phi_decay=0.1, nr_multiplier=0.75,
    mismatch_theta=1.5,
    ai_irreversible=True,
    I_base=20.0, complexity_scaling=0.15,
    r_discount=0.05, T_horizon=20, eta_hurdle=0.15,
    omega_adapt=0.3,
    employment_protection=True,
    steps_per_year=12, severance_rate=1/3,
    chain_limit=36, p_convert=0.5,
    init_share_vast=0.6, init_tenure_max_years=10.0,
    alpha_source="uniform", alpha_min=0.2, alpha_max=0.8,
    # Goods market
    A_base=15.0, gamma=0.05, B=0.5,
)

CONFIGS = {
    "A (conservative)":  dict(k_ai=15.0, k_ai_decay=0.02, k_ai_floor=0.5),
    "B (base)":          dict(k_ai=12.0, k_ai_decay=0.02, k_ai_floor=0.5),
    "C (aggressive AI)": dict(k_ai=8.0,  k_ai_decay=0.03, k_ai_floor=0.5),
}


def run(cfg_overrides: dict, mode: str):
    p = dict(BASE); p.update(cfg_overrides); p["adoption_mode"] = mode
    m = LabourMarketModel(**p)
    for _ in range(N_STEPS):
        m.step()
    df = m.datacollector.get_model_vars_dataframe()
    return df, m


def summarise(df, m, label: str):
    tail_early = df.iloc[20:40]
    tail_late  = df.iloc[-20:]

    def win(col, win_df):
        return float(win_df[col].mean()) if col in df else float("nan")

    mc = np.mean([x for x in (prod.calculate_marginal_cost() for prod in m.producers) if np.isfinite(x)])

    print(f"  {label:>14}:  "
          f"p {win('price', tail_early):5.1f}->{win('price', tail_late):5.1f}  "
          f"Y {win('wage_bill', tail_early):5.0f}->{win('wage_bill', tail_late):5.0f}  "
          f"emp_l {win('employment_rate_low', tail_early):4.0%}->{win('employment_rate_low', tail_late):4.0%}  "
          f"emp_h {win('employment_rate_high', tail_early):4.0%}->{win('employment_rate_high', tail_late):4.0%}  "
          f"adopt {win('ai_adoption_rate', tail_early):5.1%}->{win('ai_adoption_rate', tail_late):5.1%}  "
          f"profit {win('Avg profit per producer', tail_late):5.1f}  "
          f"MC {mc:4.2f}")


for cfg_name, cfg in CONFIGS.items():
    print(f"\n>>> Config {cfg_name}: {cfg}")
    for mode in ("npv_naive", "npv_adaptive", "npv_mean_field"):
        df, m = run(cfg, mode)
        summarise(df, m, mode)
