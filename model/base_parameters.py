from __future__ import annotations

BASE_PARAMS = dict(
    # ---- Scale & timing (M1 wiring) ----
    n_producers=20,
    n_high_skilled=200,
    n_low_skilled=250,
    seed=42,
    steps_per_year=12,

    # ---- C3 Wage-setting (wage curve, EUR/hour-equivalent scale) ----
    a_h=15.0,
    b_h=0.10,
    a_l=6.0,
    b_l=0.06,
    w_min=14.71,
    lam=0.08,

    # ---- C1 Production: productivity (Upreti & Sridhar baseline) ----
    a_prod=0.1,
    xi_prod=0.5,
    phi_base=3.0,
    phi_decay=0.04,
    nr_multiplier=0.85,

    # ---- C5 Learning Curve: AI cost dynamics (exogenous exponential decay to a floor; time-driven, NOT a Wright/experience curve) ----
    k_ai=40.0,
    k_ai_floor=20.0,
    k_ai_decay=0.020,

    # ---- C4 AI Adoption / investment (with C2 separation & matching params) ----
    p_evaluate=0.10,
    delta=0.013,
    mismatch_theta=1.2,
    ai_irreversible=True,
    I_base=80.0,
    complexity_scaling=0.15,
    r_discount=0.007,
    T_horizon=36,
    eta_hurdle=0.15,
    omega_adapt=0.15,

    # ---- C6 Demand (Kaleckian two-class consumption function) ----
    # A_t = A_base + gamma * W_{t-1} + gamma_pi * Pi_{t-1}
    # gamma    = wage-income demand weight (NOT a literal MPC: scales wage bill -> demand intercept)
    # gamma_pi = profit-income demand weight (gamma_pi < gamma; keep < 1/Q ~ 0.035 for stability)
    # Setting gamma_pi = 0 reproduces the original pure wage-led specification.
    A_base=100.0,   # small autonomous demand floor (buffer), not dominant
    B=2.0,
    gamma=0.05,
    gamma_pi=0.008,  # < gamma and < 1/Q (~0.035): keeps the nominal profit feedback stable

    # ---- C7 Employment Protection (Dutch institutions) ----
    employment_protection=True,
    severance_rate=1 / 3,
    chain_limit=36,
    p_convert=0.5,
    init_share_vast=0.67,
    init_tenure_max_years=10.0,

    # ---- C1 Production: alpha (routine-task share per firm) ----
    alpha_source="uniform",
    alpha_min=0.25,
    alpha_max=0.75,
)
