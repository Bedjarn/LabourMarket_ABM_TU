from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys


# Streamlit and its test runner do not always put the script directory on
# sys.path. Make sibling imports independent of the current working directory.
MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from multirun_experiments import render_multirun_analysis

from dashboard_utils import (
    BASE_PARAMS,
    BOOL_PARAMS,
    DEFAULT_DASHBOARD_MODES,
    KEY_METRICS,
    LABOUR_MARKET_PANELS,
    MACRO_PANELS,
    METRIC_LABELS,
    MODE_LABELS,
    MODE_COLORS,
    MODE_MARKERS,
    MODES,
    NUMERIC_PARAMS,
    OFAT_PARAMS,
    OTHER_PLOT_GROUPS,
    PLOT_METRICS,
    _config_fingerprint,
    build_comparison_figure,
    build_cost_comparison_figure,
    build_cost_comparison_preview,
    build_experiment_comparison_figure,
    build_experiment_final_table,
    build_labour_market_figure,
    build_macro_figure,
    build_other_figure,
    build_employment_protection_figure,
    build_investment_log_dataframe,
    build_investment_log_excel_bytes,
    build_other_figures,
    build_npv_heatmap_figure,
    build_npv_waterfall_figure,
    build_productivity_preview,
    build_productivity_preview_figure,
    build_sensitivity_final_figure,
    build_sensitivity_multirun_timeseries,
    build_sensitivity_summary_table,
    build_sensitivity_timeseries_figure,
    build_summary_dataframe,
    build_wage_preview_figure,
    build_automation_frontier_figure,
    build_automation_complexity_figure,
    build_alpha_status_figure,
    build_alpha_workforce_figure,
    build_task_complexity_status_figure,
    build_task_complexity_workforce_figure,
    make_single_run_bundle,
    CUMULATIVE_CHANNEL_METRICS,
    compute_example_npv_preview,
    delete_ofat_batch,
    delete_experiment_batch,
    delete_run_from_history,
    delete_sensitivity_sweep,
    estimate_initial_assignment,
    list_ofat_batches,
    list_experiment_batches,
    list_sensitivity_sweeps,
    load_combined_history_frame,
    load_ofat_batch,
    load_experiment_batch,
    load_run_history,
    load_saved_run_bundle,
    load_sensitivity_sweep,
    normalize_mode,
    normalize_modes,
    rename_run_in_history,
    run_experiment_batch,
    run_sensitivity_sweep,
    run_sensitivity_sweep_multi,
    run_simulation,
    save_experiment_batch,
    save_run_to_history,
    save_sensitivity_sweep,
    save_sensitivity_setup,
    list_sensitivity_setups,
)


DEFAULT_SELECTED_METRICS = [
    "gini_income",
    "skill_wage_premium",
    "ai_adoption_rate",
    "employment_rate_high",
    "employment_rate_low",
    "wage_high",
    "wage_low",
    "A_demand",
    "total_output",
]


APP_DIR = Path(__file__).resolve().parent / "outputs"
SETUPS_DIR = APP_DIR / "dashboard_setups"

PARAMETER_META = {
    "n_producers": {
        "label": "Amount of producers in the model",
        "formula": "Each producer has tasks, so more producers means a larger simulated economy.",
        "range_hint": "Practical range: 1 and up. No hard theoretical max, but very high values make runs slower.",
        "up": "Bigger economy and usually more hires, more tasks and smoother aggregate patterns.",
        "down": "Smaller economy, less aggregation and more visible noise from small numbers.",
        "min": 1,
        "step": 1,
    },
    "n_high_skilled": {
        "label": "Amount of high-skilled workers",
        "formula": "This is the size of the high-skilled labour pool available for hiring.",
        "range_hint": "Must be at least 0. Higher values mainly affect speed and labour supply.",
        "up": "Less scarcity in non-routine work and usually less pressure on high-skilled wages.",
        "down": "More bottlenecks in complex tasks and stronger wage pressure for high-skilled labour.",
        "min": 0,
        "step": 1,
    },
    "n_low_skilled": {
        "label": "Amount of low-skilled workers",
        "formula": "This is the size of the low-skilled labour pool available for hiring.",
        "range_hint": "Must be at least 0. Higher values mainly affect speed and labour supply.",
        "up": "More slack in routine labour and usually lower pressure on low-skilled wages.",
        "down": "More scarcity in routine tasks and stronger wage pressure for low-skilled labour.",
        "min": 0,
        "step": 1,
    },
    "seed": {
        "label": "Random seed for reproducibility",
        "formula": "Same seed means the same random draw sequence, so runs become reproducible.",
        "range_hint": "Any integer works. There is no meaningful economic maximum.",
        "up": "Not better or worse, just a different random path through the simulation.",
        "down": "Also just a different random path; use the same seed for clean comparisons.",
        "step": 1,
    },
    "delta": {
        "label": "Job separation probability per step",
        "formula": "Each employed worker faces probability delta of separation in a step.",
        "range_hint": "Natural range: 0 to 1 because it is a probability.",
        "up": "More churn, more vacancies, faster reallocation and often faster automation decisions.",
        "down": "More stable matches and slower labour-market dynamics.",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "p_evaluate": {
        "label": "Chance that a vacancy is re-evaluated for automation",
        "formula": "When a task becomes empty, the firm evaluates automation with probability p_evaluate.",
        "range_hint": "Natural range: 0 to 1 because it is a probability.",
        "up": "Firms reconsider automation more often, so adoption usually reacts faster.",
        "down": "Automation decisions happen less often and the system changes more slowly.",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "A_base": {
        "label": "Baseline demand level",
        "formula": "Demand shifter follows A = A_base + gamma * wage_bill + gamma_pi * profit_income.",
        "range_hint": "Usually positive. Very low values can create weak demand and low prices.",
        "up": "Higher demand, often supporting more output, higher prices and stronger profits.",
        "down": "Weaker demand and more downward pressure on output and employment.",
        "step": 0.5,
    },
    "gamma": {
        "label": "Wage-income demand weight (gamma)",
        "formula": "Weight on the wage bill in the demand shifter A = A_base + gamma * wage_bill + gamma_pi * profit_income. NOT a literal marginal propensity to consume: it scales an aggregate income LEVEL (euros) into a shift of the demand intercept (a price), so its units are price per euro and its absolute size is a calibration choice. What carries economic meaning is its size relative to gamma_pi.",
        "range_hint": "Positive and calibrated above gamma_pi, which encodes a wage-led regime (a euro of wage income lifts demand more than a euro of profit income). Current calibration ~0.05. Read it together with gamma_pi, not as a standalone propensity.",
        "up": "Wage income transmits more strongly into demand, strengthening the wage-led channel: higher prices and output for a given wage bill.",
        "down": "Weaker wage-income transmission and a flatter demand path.",
        "min": 0.0,
        "step": 0.005,
    },
    "gamma_pi": {
        "label": "Profit-income demand weight (gamma_pi)",
        "formula": "Weight on aggregate operating profit (profit_income = sum over firms of (P - AC) * Q) in the demand shifter A = A_base + gamma * wage_bill + gamma_pi * profit_income. Like gamma it is a scaling weight, not a literal MPC. gamma_pi = 0 reproduces a pure wage-led regime; gamma_pi > 0 lets capital income also support demand.",
        "range_hint": "Between 0 and gamma (kept below gamma for the wage-led tilt). Also keep it below ~1/Q (about 0.035 here) for stability: profit contains P*Q, so too large a value makes the price feed back on itself and diverge. Current calibration ~0.008.",
        "up": "Profit income transmits more strongly into demand, cushioning the underconsumption effect of automation - but above gamma_pi * Q ~ 1 it destabilises the price loop.",
        "down": "Profit redistribution leaks more out of aggregate demand, amplifying the wage-led contraction.",
        "min": 0.0,
        "step": 0.0025,
    },
    "B": {
        "label": "Slope of the inverse demand curve",
        "formula": "Price is computed as p = max(0.01, A - B * Q).",
        "range_hint": "Usually positive. If B were 0, price would stop reacting to output.",
        "up": "Prices fall faster when output rises, so expansion becomes less attractive.",
        "down": "Prices are less sensitive to output, making expansion easier.",
        "min": 0.0,
        "step": 0.05,
    },
    "lam": {
        "label": "Speed of wage adjustment",
        "formula": "Wages move partially toward target: new wage = lam * target + (1 - lam) * old wage.",
        "range_hint": "Natural range: 0 to 1. At 1, wages jump immediately to the target.",
        "up": "Wages react faster to scarcity and slack.",
        "down": "Wages move more slowly, so shocks linger longer.",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    "mismatch_theta": {
        "label": "Tolerance for cross-skill matching",
        "formula": "Fallback hiring is allowed when mismatch cost stays within theta times the preferred cost.",
        "range_hint": "Usually at least 1. Around 1 means strict matching; higher values mean more flexibility.",
        "up": "More flexibility between skill groups and fewer unfilled tasks.",
        "down": "Stricter segmentation and more matching frictions.",
        "min": 1.0,
        "step": 0.1,
    },
    "a_h": {
        "label": "Base wage intercept for high-skilled labour",
        "formula": "High-skilled wage target follows w_h = a_h + b_h * employed_high.",
        "range_hint": "Usually positive. It is the wage floor when employment pressure is low.",
        "up": "High-skilled labour becomes structurally more expensive.",
        "down": "High-skilled labour becomes structurally cheaper.",
        "step": 0.1,
    },
    "b_h": {
        "label": "High-skilled wage slope",
        "formula": "This is the slope in w_h = a_h + b_h * employed_high.",
        "range_hint": "Usually non-negative. Higher values make wages more sensitive to tightness.",
        "up": "High-skilled wages climb faster when that labour market gets tight.",
        "down": "A flatter high-skilled wage response.",
        "min": 0.0,
        "step": 0.005,
    },
    "a_l": {
        "label": "Base wage intercept for low-skilled labour",
        "formula": "Low-skilled wage target follows w_l = a_l + b_l * employed_low.",
        "range_hint": "Usually positive. It acts like a low-skilled wage floor.",
        "up": "Low-skilled labour becomes more expensive and routine automation often looks more attractive.",
        "down": "Low-skilled labour becomes cheaper and automation pressure weakens.",
        "step": 0.1,
    },
    "b_l": {
        "label": "Low-skilled wage slope",
        "formula": "This is the slope in w_l = a_l + b_l * employed_low.",
        "range_hint": "Usually non-negative. Higher values make wages more sensitive to tightness.",
        "up": "Low-skilled wages react more strongly to labour-market tightness.",
        "down": "A flatter low-skilled wage response.",
        "min": 0.0,
        "step": 0.005,
    },
    "w_min": {
        "label": "Explicit minimum wage floor (minimumloon)",
        "formula": "Wage target is floored at max(w_min, a + b * L) for both skill groups.",
        "range_hint": "Current default is 0.0, so the extra wage floor is switched off. Set it above 0 to impose a minimumloon-style floor on both skill groups.",
        "up": "Wages cannot fall below a higher floor, which raises the cost of low-wage labour and can make automation look more attractive at low employment levels.",
        "down": "A lower floor has less bite; below the natural intercept it has no effect at all.",
        "min": 0.0,
        "step": 0.1,
    },
    "a_prod": {
        "label": "Steepness of human productivity over task complexity",
        "formula": "Human productivity scales exponentially with task complexity, using a_prod in the exponent.",
        "range_hint": "Usually non-negative. Larger values create sharper productivity differences across tasks.",
        "up": "Comparative advantages across tasks become sharper.",
        "down": "Human productivity varies less across task complexity.",
        "min": 0.0,
        "step": 0.01,
    },
    "xi_prod": {
        "label": "Low-skill non-routine productivity scaling",
        "formula": "xi_prod dampens low-skilled productivity on non-routine tasks: low-skill non-routine productivity is exp(xi_prod * a_prod * complexity). Routine productivity is skill-neutral.",
        "range_hint": "Usually between 0 and 1. At 1, low-skilled workers have the same productivity slope as high-skilled workers on non-routine tasks; lower values create a larger non-routine skill gap.",
        "up": "Low-skilled workers become more productive on non-routine tasks. Routine task productivity is unchanged.",
        "down": "Low-skilled workers become less productive on non-routine tasks. Routine task productivity is unchanged.",
        "min": 0.0,
        "step": 0.05,
    },
    "phi_base": {
        "label": "Baseline AI productivity",
        "formula": "AI productivity starts from phi_base and then declines with complexity.",
        "range_hint": "Usually positive. Higher values make AI stronger almost everywhere.",
        "up": "AI produces more per unit and automation usually becomes more attractive.",
        "down": "AI contributes less output and adoption pressure weakens.",
        "min": 0.0,
        "step": 0.1,
    },
    "phi_decay": {
        "label": "How fast AI productivity falls with complexity",
        "formula": "AI productivity declines roughly like phi_base * exp(-phi_decay * complexity).",
        "range_hint": "Usually non-negative. At 0, AI no longer loses productivity as tasks get more complex.",
        "up": "AI loses ground faster on complex tasks.",
        "down": "AI remains useful deeper into complex tasks.",
        "min": 0.0,
        "step": 0.01,
    },
    "nr_multiplier": {
        "label": "AI multiplier on non-routine tasks",
        "formula": "Non-routine AI productivity is routine AI productivity times nr_multiplier.",
        "range_hint": "< 1: AI disadvantage on non-routine, = 1: neutral, > 1: AI advantage on non-routine. For diagnostics, 0.5 to 1.5 is a sensible first range.",
        "up": "AI becomes relatively stronger on non-routine tasks.",
        "down": "AI becomes relatively weaker on non-routine tasks.",
        "min": 0.0,
        "step": 0.05,
    },
    "k_ai": {
        "label": "Initial AI rental cost",
        "formula": "This is the starting value of the public AI rental cost before learning reduces it over time.",
        "range_hint": "Usually positive. Higher values delay adoption.",
        "up": "AI starts more expensive, so adoption usually begins later.",
        "down": "AI is attractive earlier in the simulation.",
        "min": 0.0,
        "step": 0.5,
    },
    "k_ai_decay": {
        "label": "Speed of the AI learning curve",
        "formula": "AI cost falls over time according to an exponential learning curve using k_ai_decay.",
        "range_hint": "Usually non-negative. At 0, AI cost no longer declines over time.",
        "up": "AI costs fall faster over time.",
        "down": "AI remains expensive for longer.",
        "min": 0.0,
        "step": 0.005,
    },
    "k_ai_floor": {
        "label": "Long-run minimum AI cost",
        "formula": "This is the lower bound that AI cost approaches over time.",
        "range_hint": "Usually non-negative and below the initial k_ai value.",
        "up": "AI stays relatively expensive even in the long run.",
        "down": "AI can become much cheaper over time.",
        "min": 0.0,
        "step": 0.1,
    },
    "ai_irreversible": {
        "label": "AI investment is irreversible",
        "formula": "When True (NPV modes only): once a task is automated, it cannot be de-automated until the planning horizon T_horizon expires. When False: firms can freely replace AI with humans at any step, even in NPV modes.",
        "range_hint": "True/False toggle. Only relevant in NPV modes - ULC mode always has free reversal.",
        "up": "Stronger lock-in: firms cannot undo automation decisions cheaply, which may suppress early adoption but also prevents churning.",
        "down": "Free reversal: if wages fall after automation, firms can switch back to humans, creating more dynamic labour-market cycling.",
        "bool": True,
    },
    "I_base": {
        "label": "Fixed investment cost of automation",
        "formula": "NPV modes start with an upfront cost I_base before future savings are counted.",
        "range_hint": "Usually non-negative. Higher values create a stronger investment hurdle.",
        "up": "Higher entry barrier for NPV-based automation.",
        "down": "Firms can automate more easily.",
        "min": 0.0,
        "step": 0.5,
    },
    "complexity_scaling": {
        "label": "How much automation cost rises with complexity",
        "formula": "Automation cost is scaled upward for more complex tasks using complexity_scaling.",
        "range_hint": "Usually non-negative. At 0, complexity no longer raises the investment cost.",
        "up": "Complex tasks become much harder to automate profitably.",
        "down": "Less difference between simple and complex tasks.",
        "min": 0.0,
        "step": 0.01,
    },
    "r_discount": {
        "label": "Discount rate in NPV calculations",
        "formula": "Future cost savings are discounted by (1 + r_discount)^t.",
        "range_hint": "Usually 0 or above. Larger values make the future matter less.",
        "up": "Future savings matter less, so automation tends to slow down.",
        "down": "Future cost savings matter more, supporting adoption.",
        "min": 0.0,
        "step": 0.01,
    },
    "T_horizon": {
        "label": "Planning horizon for automation benefits",
        "formula": "Firms evaluate expected savings over T_horizon future periods.",
        "range_hint": "Must be at least 1. Larger values mean a longer planning horizon and longer runtimes.",
        "up": "Firms look further ahead, which often favours investment.",
        "down": "A more short-term perspective and often less automation.",
        "min": 1,
        "step": 1,
    },
    "eta_hurdle": {
        "label": "Extra hurdle for automation investment",
        "formula": "Automation only happens when NPV is sufficiently high relative to the hurdle condition.",
        "range_hint": "Usually 0 or above. Higher values mean stricter approval.",
        "up": "Stricter investment threshold and less adoption.",
        "down": "Firms accept automation more easily.",
        "min": 0.0,
        "step": 0.01,
    },
    "omega_adapt": {
        "label": "Update speed for expectations and smoothed trends",
        "formula": "Adaptive expectations are updated with a smoothing rule that weights new information by omega_adapt.",
        "range_hint": "Natural range: 0 to 1. At 1, only the newest information matters.",
        "up": "Expectations react faster to recent changes.",
        "down": "More inertia and smoother trend updates.",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    "employment_protection": {
        "label": "Enable employment protection (ontslagbescherming)",
        "formula": "When True, workers have contract types (flex/vast), severance costs are added to the NPV upfront cost, and the ketenregeling converts flex workers to vast or non-renews after chain_limit steps.",
        "range_hint": "True/False toggle.",
        "up": "Severance costs raise the effective cost of automation and slow adoption.",
        "down": "No contract distinction, severance, or chain-limit mechanism.",
        "bool": True,
    },
    "steps_per_year": {
        "label": "Model steps per calendar year",
        "formula": "Used to convert tenure (in steps) to years for severance calculations.",
        "range_hint": "Typically 12 (monthly) or 52 (weekly).",
        "up": "Finer time resolution; severance cost per step is smaller.",
        "down": "Coarser time resolution; each step represents more tenure.",
        "min": 1,
        "step": 1,
    },
    "severance_rate": {
        "label": "Severance multiplier (months of wage per year of tenure)",
        "formula": "Severance = severance_rate × wage × tenure_years. Default 1/3 ≈ Dutch transitievergoeding.",
        "range_hint": "0 disables severance. 1/3 is the statutory Dutch rate.",
        "up": "Automation becomes costlier for firms with many permanent workers.",
        "down": "Less friction from severance; closer to no-protection baseline.",
        "min": 0.0,
        "step": 0.05,
    },
    "chain_limit": {
        "label": "Ketenregeling chain limit (steps)",
        "formula": "Flex workers reaching chain_limit steps of tenure trigger conversion to vast (prob. p_convert) or non-renewal.",
        "range_hint": "Dutch law: 36 months. In model steps: 36 if steps_per_year=12.",
        "up": "Workers stay flex longer before the chain clause fires.",
        "down": "Chain clause fires sooner; faster conversion or turnover.",
        "min": 1,
        "step": 1,
    },
    "p_convert": {
        "label": "Probability of flex→vast conversion at chain limit",
        "formula": "At chain_limit, firms convert the worker to vast with probability p_convert; otherwise non-renew (if replacements available).",
        "range_hint": "0 to 1. 1 means every worker at the limit is converted.",
        "up": "More conversions; the permanent workforce grows faster.",
        "down": "More non-renewals; firms keep the workforce flexible by cycling workers.",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    "init_share_vast": {
        "label": "Initial share of employed workers on permanent contracts",
        "formula": "At warm-start, this fraction of employed workers are assigned contract_type='vast' with tenure drawn from [chain_limit, init_tenure_max_years × steps_per_year].",
        "range_hint": "0 to 1. Dutch context: roughly 0.6–0.7 permanent share.",
        "up": "Higher severance exposure from step 1; adoption may slow immediately.",
        "down": "Workforce starts more flex; severance costs are initially low.",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    "init_tenure_max_years": {
        "label": "Maximum initial tenure drawn for permanent workers (years)",
        "formula": "Permanent workers at warm-start draw tenure uniformly from [chain_limit, init_tenure_max_years × steps_per_year].",
        "range_hint": "Must be above chain_limit / steps_per_year. Typical: 5–15 years.",
        "up": "Workers start with longer tenure and higher severance exposure.",
        "down": "Permanent workers start with shorter tenure; severance costs are lower early on.",
        "min": 0.0,
        "step": 0.5,
    },
}

METRIC_GUIDE = {
    "ai_adoption_rate": {
        "what": "Share of all tasks that are currently automated by AI.",
        "formula": "automated tasks / total tasks",
        "interpretation": "0 means no tasks are automated. 1 means every task is automated.",
        "caveat": "This is a task share, not an employment share or output share.",
    },
    "new_reactive_automations_this_step": {
        "what": "Number of tasks that were automated this step after a worker left and the empty task was reevaluated.",
        "formula": "count of reactive automation decisions in the current step",
        "interpretation": "This isolates the substitution margin: automation triggered by replacement after exogenous separation.",
        "caveat": "A low value does not mean the channel is unimportant; opportunities only arise when separation happens first.",
    },
    "new_proactive_automations_this_step": {
        "what": "Number of tasks that were automated this step during bottleneck expansion or proactive reevaluation.",
        "formula": "count of proactive automation decisions in the current step",
        "interpretation": "This isolates the expansion margin: automation chosen as part of deliberate capacity expansion or proactive replacement.",
        "caveat": "This is event-count data, so spikes and quiet periods are normal.",
    },
    "employment_rate_high": {
        "what": "Fraction of all high-skilled workers that is employed.",
        "formula": "employed high-skilled / total high-skilled",
        "interpretation": "Closer to 1 means a tighter high-skilled labour market.",
        "caveat": "This does not show job quality, mismatch, or task composition within the high-skilled group.",
    },
    "employment_rate_low": {
        "what": "Fraction of all low-skilled workers that is employed.",
        "formula": "employed low-skilled / total low-skilled",
        "interpretation": "Closer to 1 means a tighter low-skilled labour market.",
        "caveat": "This does not show whether low-skilled workers are in preferred tasks or mismatch assignments.",
    },
    "wage_high": {
        "what": "Current market wage for high-skilled labour.",
        "formula": "w_h = a_h + b_h * employed_high, with gradual adjustment",
        "interpretation": "Higher values mean high-skilled labour is scarce or structurally expensive.",
        "caveat": "This is a model wage rule, not a bargaining outcome or a firm-specific wage distribution.",
    },
    "wage_low": {
        "what": "Current market wage for low-skilled labour.",
        "formula": "w_l = a_l + b_l * employed_low, with gradual adjustment",
        "interpretation": "Higher values mean low-skilled labour is scarce or structurally expensive.",
        "caveat": "If `w_min` binds, this series partly reflects the imposed floor rather than labour-market tightness alone.",
    },
    "skill_wage_premium": {
        "what": "Ratio between high-skilled and low-skilled wages.",
        "formula": "w_h / w_l",
        "interpretation": "Higher values mean a wider skill wage premium and therefore stronger wage polarisation.",
        "caveat": "Read this together with both wage levels; the ratio can move because high wages rise, low wages fall, or both.",
    },
    "total_output": {
        "what": "Aggregate production of all producers together.",
        "formula": "sum of firm outputs, where each firm uses a Leontief bottleneck technology",
        "interpretation": "Higher output means the economy is producing more final goods.",
        "caveat": "Because production is bottleneck-driven, a small task shortfall can move output sharply.",
    },
    "price": {
        "what": "Market price implied by inverse demand.",
        "formula": "p = max(0.01, A - B * Q)",
        "interpretation": "Higher price can reflect stronger demand or lower total output.",
        "caveat": "You should read price together with output and demand, because price alone does not tell you which force dominated.",
    },
    "labour_share": {
        "what": "Share of revenue that goes to labour income.",
        "formula": "wage bill / (price * total output)",
        "interpretation": "Higher labour share means workers receive a larger part of income.",
        "caveat": "This is an aggregate accounting share; it does not reveal how labour income is split across high and low skill.",
    },
    "ULC": {
        "what": "Aggregate unit labour cost.",
        "formula": "total labour costs / total output",
        "interpretation": "Higher values mean labour is expensive relative to how much is produced.",
        "caveat": "It is a macro average, so it can hide very different unit costs across task types.",
    },
    "wage_bill": {
        "what": "Total wages paid to employed workers.",
        "formula": "sum of all wages of employed workers",
        "interpretation": "This is total labour income, and it also feeds back into demand.",
        "caveat": "A falling wage bill can come from lower employment, lower wages, or both.",
    },
    "k_ai_current": {
        "what": "Current public AI rental cost after learning-by-doing over time.",
        "formula": "declines over time toward k_ai_floor using the AI learning curve",
        "interpretation": "Lower values make automation financially more attractive.",
        "caveat": "This path is exogenous to firms except through the common learning-curve structure.",
    },
    "A_demand": {
        "what": "Demand shifter used in the inverse demand equation.",
        "formula": "A = A_base + gamma * wage_bill + gamma_pi * profit_income (Kaleckian two-class consumption function).",
        "interpretation": "Higher A means the economy can support higher prices for a given output level. The split between the wage-income channel and the profit-income channel determines whether the demand regime is wage-led or profit-led.",
        "caveat": "It is not an independent demand shock series; it is mechanically linked to the wage bill AND aggregate profit income.",
    },
    "profit_income_lagged": {
        "what": "Aggregate operating profit income at the end of the previous step.",
        "formula": "sum over firms of (P - AC) * Q = total revenue minus total production costs",
        "interpretation": "Capital-income counterpart to the wage bill. Together they exhaust gross value added: W + Pi = P * Q. Feeds into A via the gamma_pi channel.",
        "caveat": "Operating profit can be negative when P < AC, in which case it acts as a demand drag. Calibrate so the initial profit margin (P - AC) > 0.",
    },
    "wage_demand_contribution": {
        "what": "Contribution of wage income to the demand shifter A this step.",
        "formula": "gamma * wage_bill_lagged",
        "interpretation": "How much of A comes from the labour-income channel.",
        "caveat": "Compare against profit_demand_contribution to gauge whether the demand regime is wage-led or profit-led.",
    },
    "profit_demand_contribution": {
        "what": "Contribution of profit income to the demand shifter A this step.",
        "formula": "gamma_pi * profit_income_lagged",
        "interpretation": "How much of A comes from the capital-income channel.",
        "caveat": "When gamma_pi = 0 this is zero by construction, reproducing the original wage-led specification.",
    },
    "avg_npv_routine": {
        "what": "Average NPV of routine tasks that are still not automated yet.",
        "formula": "mean of projected discounted cost savings minus upfront investment cost, computed only over remaining non-automated routine tasks",
        "interpretation": "This can stay negative even when firms did automate other tasks, because the attractive routine tasks may already have been removed from this remaining pool.",
        "caveat": "This is a selection-affected average over remaining tasks, not over all tasks.",
    },
    "avg_npv_nonroutine": {
        "what": "Average NPV of non-routine tasks that are still not automated yet.",
        "formula": "mean of projected discounted cost savings minus upfront investment cost, computed only over remaining non-automated non-routine tasks",
        "interpretation": "If this stays strongly negative, it means the remaining non-routine tasks are unattractive. It does not automatically mean no earlier investment happened elsewhere.",
        "caveat": "Like the routine version, this excludes tasks that were already automated.",
    },
    "share_adoptable_npv_routine": {
        "what": "Share of remaining non-automated routine tasks that meet the model's automation threshold.",
        "formula": "fraction of non-automated routine tasks where NPV > eta_hurdle * investment_cost",
        "interpretation": "Useful to see whether there are still routine tasks left that firms would actually approve under the current investment rule.",
        "caveat": "This is not the share of all routine tasks in the economy; it only refers to the remaining non-automated subset.",
    },
    "share_adoptable_npv_nonroutine": {
        "what": "Share of remaining non-automated non-routine tasks that meet the model's automation threshold.",
        "formula": "fraction of non-automated non-routine tasks where NPV > eta_hurdle * investment_cost",
        "interpretation": "If this is near zero, almost all remaining non-routine tasks fail the actual adoption rule under current assumptions.",
        "caveat": "Interpret this together with how many non-routine tasks are left; a high share over a tiny remaining pool can be misleading.",
    },
    "avg_realized_npv_this_step": {
        "what": "Average NPV of the tasks that were actually automated this step.",
        "formula": "mean stored NPV at the moment of investment for tasks with ai_investment_step = current step",
        "interpretation": "This directly reflects the attractiveness of investments that firms truly made right now.",
        "caveat": "This can be noisy because some steps have few or zero realized automations.",
    },
    "avg_realized_npv_cumulative": {
        "what": "Average NPV of all tasks that have actually been automated so far.",
        "formula": "mean stored NPV at investment time over all realized automation decisions",
        "interpretation": "This is often more informative than the remaining-task average, because it summarizes the quality of decisions that were actually taken.",
        "caveat": "It smooths over timing, so it can hide whether recent investments have become much better or worse.",
    },
    "Avg profit per producer": {
        "what": "Average profit level across producers.",
        "formula": "mean of (firm revenue - total firm costs) across all producers",
        "interpretation": "Higher values mean firms are, on average, earning more after labour and AI costs.",
        "caveat": "This is an average; dispersion across producers can still be large.",
    },
    "input_cost_ai_routine": {
        "what": "Average AI unit cost on routine tasks.",
        "formula": "mean(k_ai / productivity_ai(task)) over routine tasks",
        "interpretation": "Lower values mean AI can perform routine tasks more cheaply per unit of output.",
        "caveat": "This uses unit costs, not fixed investment costs, so it is not the same as the full NPV decision.",
    },
    "input_cost_ai_nonroutine": {
        "what": "Average AI unit cost on non-routine tasks.",
        "formula": "mean(k_ai / productivity_ai(task)) over non-routine tasks",
        "interpretation": "Lower values mean AI is becoming competitive on harder, non-routine work.",
        "caveat": "This differs from the routine version because AI productivity on non-routine tasks is scaled by `nr_multiplier`.",
    },
    "input_cost_human_high_routine": {
        "what": "Average high-skill human unit cost on routine tasks.",
        "formula": "mean(w_h / productivity_human(task, high)) over routine tasks",
        "interpretation": "Shows how expensive high-skilled labour is per unit on routine work.",
        "caveat": "High-skilled workers may not be the preferred input for routine tasks, so this is partly a counterfactual cost benchmark.",
    },
    "input_cost_human_low_routine": {
        "what": "Average low-skill human unit cost on routine tasks.",
        "formula": "mean(w_l / productivity_human(task, low)) over routine tasks",
        "interpretation": "Shows how expensive low-skilled labour is per unit on routine work.",
        "caveat": "This is an average over all routine tasks, so it hides the complexity gradient within routine work.",
    },
    "input_cost_human_high_nonroutine": {
        "what": "Average high-skill human unit cost on non-routine tasks.",
        "formula": "mean(w_h / productivity_human(task, high)) over non-routine tasks",
        "interpretation": "Shows the cost benchmark for the skill group that is usually better suited to non-routine work.",
        "caveat": "Because it is an average, it does not show whether only the easier non-routine tasks remain relevant for adoption.",
    },
    "input_cost_human_low_nonroutine": {
        "what": "Average low-skill human unit cost on non-routine tasks.",
        "formula": "mean(w_l / productivity_human(task, low)) over non-routine tasks",
        "interpretation": "Useful as a mismatch benchmark: how costly low-skilled labour is on non-routine work.",
        "caveat": "This often reflects mismatch productivity penalties more than actual equilibrium staffing.",
    },
    "Avg labour cost per producer": {
        "what": "Average wage bill borne by a producer.",
        "formula": "sum(firm labour costs) / number of producers",
        "interpretation": "Shows how much labour spending an average firm carries.",
        "caveat": "This excludes AI costs, so it is only one component of total firm cost.",
    },
    "new_automations_this_step": {
        "what": "Number of tasks automated in the current step.",
        "formula": "count(tasks with ai_investment_step = current_step)",
        "interpretation": "Tracks the overall pace of realized AI adoption.",
        "caveat": "In the dashboard this is often shown cumulatively for readability, so check the title carefully.",
    },
    "new_displaced_workers_this_step": {
        "what": "Number of workers displaced by automation in the current step.",
        "formula": "sum of workers removed from tasks when those tasks are automated in the current step",
        "interpretation": "This is the worker-flow analogue of automation events.",
        "caveat": "One automation can displace more than one worker, so do not compare it one-for-one with automation counts.",
    },
}


def _flatten_panel_metrics(panels) -> list[str]:
    metrics: list[str] = []
    for panel in panels:
        panel_metrics = panel["metrics"] if isinstance(panel, dict) else [panel[0]]
        for metric in panel_metrics:
            if metric not in metrics:
                metrics.append(metric)
    return metrics


def _render_plot_metric_table(metrics: list[str], title: str = "Plot reference"):
    rows = []
    for metric in metrics:
        info = METRIC_GUIDE.get(metric)
        if not info:
            continue
        rows.append(
            {
                "metric": METRIC_LABELS.get(metric, metric),
                "formula": info.get("formula", ""),
                "economic meaning": info.get("interpretation", info.get("what", "")),
            }
        )
    if not rows:
        return
    st.caption(title)
    table_df = pd.DataFrame(rows)
    table_html = table_df.to_html(index=False, escape=True, classes="plot-ref-table")
    wrapped_html = f"""
    <html>
    <head>
    <style>
    body {{
        margin: 0;
        padding: 0;
        background: transparent;
        color: #e5e7eb;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .plot-ref-table {{
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        font-size: 15px;
    }}
    .plot-ref-table th, .plot-ref-table td {{
        border: 1px solid rgba(148, 163, 184, 0.22);
        padding: 12px 14px;
        vertical-align: top;
        text-align: left;
        white-space: normal;
        word-break: break-word;
        overflow-wrap: anywhere;
    }}
    .plot-ref-table th {{
        background: rgba(30, 41, 59, 0.85);
        color: #f8fafc;
    }}
    .plot-ref-table td {{
        background: rgba(15, 23, 42, 0.58);
    }}
    .plot-ref-table th:nth-child(1), .plot-ref-table td:nth-child(1) {{ width: 24%; }}
    .plot-ref-table th:nth-child(2), .plot-ref-table td:nth-child(2) {{ width: 36%; }}
    .plot-ref-table th:nth-child(3), .plot-ref-table td:nth-child(3) {{ width: 40%; }}
    </style>
    </head>
    <body>
    {table_html}
    </body>
    </html>
    """
    table_height = max(180, 72 + len(rows) * 58)
    components.html(wrapped_html, height=table_height, scrolling=False)


def _default_run_label() -> str:
    return "dashboard_run"


def _setup_payload_from_state() -> dict:
    params = {}
    for key, default_value in BASE_PARAMS.items():
        state_key = f"param_{key}"
        params[key] = st.session_state.get(state_key, default_value)
    return {
        "run_label": st.session_state.get("run_label_input", _default_run_label()),
        "n_steps": int(st.session_state.get("steps_input", 1000)),
        "modes": normalize_modes(st.session_state.get("modes_input", MODES)),
        "params": params,
    }


def _list_saved_setups() -> list[dict]:
    if not SETUPS_DIR.exists():
        return []
    setups = []
    for path in sorted(SETUPS_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["file_name"] = path.stem
            setups.append(payload)
        except Exception:
            continue
    return setups


def _save_setup(setup_name: str):
    safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in setup_name).strip("_")
    safe_name = safe_name or "setup"
    SETUPS_DIR.mkdir(parents=True, exist_ok=True)
    payload = _setup_payload_from_state()
    payload["setup_name"] = setup_name.strip() or safe_name
    (SETUPS_DIR / f"{safe_name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_setup_into_state(setup_payload: dict):
    st.session_state["run_label_input"] = setup_payload.get("run_label", _default_run_label())
    st.session_state["steps_input"] = int(setup_payload.get("n_steps", 1000))
    st.session_state["modes_input"] = normalize_modes(setup_payload.get("modes", MODES))
    params = dict(setup_payload.get("params", {}))
    if "nr_penalty" in params and "nr_multiplier" not in params:
        params["nr_multiplier"] = params.pop("nr_penalty")
    for key, value in params.items():
        st.session_state[f"param_{key}"] = value


def _delete_setup(file_name: str) -> bool:
    path = SETUPS_DIR / f"{file_name}.json"
    if not path.exists():
        return False
    path.unlink()
    return True


def _load_run_into_setup(run: dict):
    st.session_state["run_label_input"] = run.get("run_label", _default_run_label())
    st.session_state["steps_input"] = int(run.get("n_steps", 1000))
    st.session_state["modes_input"] = normalize_modes(run.get("modes", MODES))
    params = dict(run.get("params", {}))
    if "nr_penalty" in params and "nr_multiplier" not in params:
        params["nr_multiplier"] = params.pop("nr_penalty")
    for key, value in params.items():
        st.session_state[f"param_{key}"] = value


def _apply_pending_sidebar_state():
    pending_setup = st.session_state.pop("pending_setup_payload", None)
    if pending_setup is not None:
        _load_setup_into_state(pending_setup)

    pending_run = st.session_state.pop("pending_run_to_load", None)
    if pending_run is not None:
        _load_run_into_setup(pending_run)
        bundle = load_saved_run_bundle(APP_DIR, pending_run["run_id"])
        if bundle is not None:
            st.session_state["latest_bundle"] = bundle
            st.session_state["latest_run_dir"] = pending_run.get("path")


def _parameter_groups():
    return {
        "Population": ["n_producers", "n_high_skilled", "n_low_skilled", "seed"],
        "Productivity": ["a_prod", "xi_prod", "phi_base", "phi_decay", "nr_multiplier"],
        "Wage setting": ["a_h", "b_h", "a_l", "b_l", "w_min", "lam"],
        "Goods market": ["A_base", "gamma", "gamma_pi", "B"],
        "AI adoption": ["k_ai", "k_ai_floor", "k_ai_decay", "p_evaluate", "delta", "mismatch_theta", "ai_irreversible"],
        "NPV parameters": ["I_base", "complexity_scaling", "r_discount", "T_horizon", "eta_hurdle", "omega_adapt"],
        "Employment Protection": ["employment_protection", "steps_per_year", "severance_rate", "chain_limit", "p_convert", "init_share_vast", "init_tenure_max_years"],
    }


def _inject_styles():
    st.markdown(
        """
        <style>
        .param-card {
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 14px;
            padding: 0.8rem 0.9rem 0.65rem 0.9rem;
            margin: 0.35rem 0 0.45rem 0;
            background: rgba(248, 250, 252, 0.04);
        }
        .param-title {
            font-size: 0.93rem;
            color: #e5e7eb;
            margin-bottom: 0.35rem;
            line-height: 1.35;
        }
        .param-line {
            font-size: 0.81rem;
            margin-bottom: 0.22rem;
            line-height: 1.35;
        }
        .param-neutral { color: #cbd5e1; }
        .param-up { color: #86efac; }
        .param-down { color: #fdba74; }
        .metric-card {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 14px;
            padding: 0.8rem 0.95rem;
            margin: 0.45rem 0;
            background: rgba(15, 23, 42, 0.22);
        }
        .metric-title {
            font-size: 0.95rem;
            margin-bottom: 0.3rem;
            color: #f8fafc;
        }
        .metric-body {
            font-size: 0.82rem;
            color: #cbd5e1;
            line-height: 1.45;
        }
        .loop-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 0.8rem;
            margin: 0.7rem 0 1rem 0;
        }
        .loop-card {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            background: rgba(15, 23, 42, 0.22);
        }
        .loop-title {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-bottom: 0.3rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .loop-value {
            font-size: 1.35rem;
            color: #f8fafc;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .loop-sub {
            font-size: 0.8rem;
            color: #cbd5e1;
            line-height: 1.4;
        }
        .loop-delta-neg { color: #fca5a5; }
        .loop-delta-pos { color: #86efac; }
        .info-card {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin: 0.45rem 0;
            background: rgba(15, 23, 42, 0.22);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_parameter_help(key: str):
    info = PARAMETER_META.get(key)
    if not info:
        return
    st.markdown(
        f"""
        <div class="param-card">
            <div class="param-title"><strong>{key}</strong> - {info['label']}</div>
            <div class="param-line param-neutral"><strong>Meaning:</strong> {info['formula']}</div>
            <div class="param-line param-neutral"><strong>Range / max:</strong> {info['range_hint']}</div>
            <div class="param-line param-up"><strong>Up:</strong> {info['up']}</div>
            <div class="param-line param-down"><strong>Down:</strong> {info['down']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_guide():
    st.subheader("Metric guide")
    st.caption("Use this as a reading guide for the plots and summary tables. If a graph name feels unclear, this section is the glossary for it.")
    for metric, info in METRIC_GUIDE.items():
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title"><strong>{METRIC_LABELS.get(metric, metric)}</strong> <span style="color:#94a3b8;">({metric})</span></div>
                <div class="metric-body"><strong>What it is:</strong> {info['what']}</div>
                <div class="metric-body"><strong>How it is computed:</strong> {info['formula']}</div>
                <div class="metric-body"><strong>How to read it:</strong> {info['interpretation']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_setup_diff(loaded_payload: dict):
    """Show which parameters changed relative to the current session state."""
    loaded_params = loaded_payload.get("params", {})
    changed = []
    for key, new_val in loaded_params.items():
        old_val = st.session_state.get(f"param_{key}", BASE_PARAMS.get(key))
        if old_val != new_val:
            changed.append({"parameter": key, "current": old_val, "→ new": new_val})
    if not changed:
        st.sidebar.caption("No parameter changes vs current settings.")
        return
    diff_df = pd.DataFrame(changed)
    st.sidebar.caption(f"{len(changed)} parameter(s) will change:")
    st.sidebar.dataframe(diff_df, use_container_width=True, hide_index=True)


def _uses_integer_number_input(value, meta: dict) -> bool:
    """Return True only when all number_input arguments should be ints."""
    step = meta.get("step", 1 if isinstance(value, int) and not isinstance(value, bool) else 0.01)
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and isinstance(step, int)
        and not isinstance(step, bool)
    )


def _build_sidebar():
    st.sidebar.header("Run settings")

    # ── Custom setup library ───────────────────────────────────────────────────
    st.sidebar.subheader("Custom setup library")
    setup_name = st.sidebar.text_input("Setup name", value="my_setup", key="setup_name_input")
    save_col, load_col = st.sidebar.columns(2)
    with save_col:
        if st.button("Save setup", use_container_width=True):
            _save_setup(setup_name)
            st.sidebar.success("Setup saved.")
            st.rerun()
    saved_setups = _list_saved_setups()
    setup_options = {item.get("setup_name", item["file_name"]): item for item in saved_setups}
    selected_setup_name = st.sidebar.selectbox(
        "Load setup",
        options=[""] + list(setup_options.keys()),
        key="setup_select_input",
    )
    with load_col:
        if selected_setup_name and st.button("Load setup", use_container_width=True):
            _load_setup_into_state(setup_options[selected_setup_name])
            st.sidebar.success("Setup loaded.")
            st.rerun()
    if selected_setup_name:
        with st.sidebar.expander("Preview: what changes?", expanded=False):
            _render_setup_diff(setup_options[selected_setup_name])
    if selected_setup_name and st.sidebar.button("Delete loaded setup", use_container_width=True, type="secondary"):
        deleted = _delete_setup(setup_options[selected_setup_name]["file_name"])
        if deleted:
            st.sidebar.success("Setup deleted.")
            st.rerun()
    st.sidebar.divider()

    run_label = st.sidebar.text_input("Run name", value=_default_run_label(), key="run_label_input")
    n_steps = st.sidebar.number_input("Steps", min_value=1, max_value=5000, value=1000, step=10, key="steps_input")
    st.sidebar.caption("Tip: use the same seed when you want a cleaner comparison between parameter changes.")
    selected_modes = st.sidebar.multiselect(
        "Modes",
        options=MODES,
        default=DEFAULT_DASHBOARD_MODES,
        format_func=lambda mode: MODE_LABELS.get(mode, mode),
        key="modes_input",
    )

    # ── Quick toggles (outside expanders for fast access) ─────────────────────
    st.sidebar.caption("Quick toggles")
    qt_col1, qt_col2 = st.sidebar.columns(2)
    with qt_col1:
        ep_on = st.toggle(
            "Employment protection",
            value=bool(st.session_state.get("param_employment_protection", BASE_PARAMS["employment_protection"])),
            key="ep_quick_toggle",
            help="Shortcut for employment_protection in the 'Employment Protection' parameter group.",
        )
        st.session_state["param_employment_protection"] = ep_on
    with qt_col2:
        ai_irrev_on = st.toggle(
            "AI irreversible",
            value=bool(st.session_state.get("param_ai_irreversible", BASE_PARAMS.get("ai_irreversible", False))),
            key="ai_irrev_quick_toggle",
            help="Shortcut for ai_irreversible in the 'AI adoption' parameter group. When on, automated tasks cannot be reversed until T_horizon expires.",
        )
        st.session_state["param_ai_irreversible"] = ai_irrev_on

    params = {}
    for group_name, keys in _parameter_groups().items():
        with st.sidebar.expander(group_name, expanded=False):
            for key in keys:
                meta = PARAMETER_META.get(key, {})
                value = BASE_PARAMS[key]
                _render_parameter_help(key)

                if isinstance(value, bool) or meta.get("bool"):
                    # Bool parameters use a checkbox widget
                    params[key] = st.checkbox(
                        f"Enable {key}",
                        value=st.session_state.get(f"param_{key}", bool(value)),
                        key=f"param_{key}",
                    )
                else:
                    is_int_input = _uses_integer_number_input(value, meta)
                    widget_kwargs = {
                        "label": f"Set {key}",
                        "label_visibility": "collapsed",
                        "value": int(value) if is_int_input else float(value),
                        "step": int(meta.get("step", 1)) if is_int_input else float(meta.get("step", 0.01)),
                        "key": f"param_{key}",
                    }
                    if "min" in meta:
                        widget_kwargs["min_value"] = int(meta["min"]) if is_int_input else float(meta["min"])
                    if "max" in meta:
                        widget_kwargs["max_value"] = int(meta["max"]) if is_int_input else float(meta["max"])
                    if not is_int_input:
                        widget_kwargs["format"] = "%.4f"

                    if is_int_input:
                        params[key] = int(st.number_input(**widget_kwargs))
                    else:
                        params[key] = float(st.number_input(**widget_kwargs))

    # --- Alpha distribution controls ---
    with st.sidebar.expander("Alpha distribution (routine-task share)", expanded=False):
        st.caption(
            "Controls how each firm's routine-task share (alpha) is drawn. "
            "'Uniform' draws from U(min, max). 'Gmyrek data' samples from the "
            "empirical automation-score distribution in tasks_by_4digit.xlsx."
        )
        alpha_source_options = ["uniform", "data"]
        alpha_source_labels  = {"uniform": "Uniform U(min, max)", "data": "Sample from Gmyrek data"}
        current_alpha_source = st.session_state.get("param_alpha_source", "uniform")
        alpha_source = st.radio(
            "Alpha source",
            options=alpha_source_options,
            format_func=lambda v: alpha_source_labels[v],
            index=alpha_source_options.index(current_alpha_source) if current_alpha_source in alpha_source_options else 0,
            key="param_alpha_source",
            horizontal=True,
        )
        params["alpha_source"] = alpha_source

        current_alpha_min = float(st.session_state.get("param_alpha_min", BASE_PARAMS.get("alpha_min", 0.0)))
        current_alpha_max = float(st.session_state.get("param_alpha_max", BASE_PARAMS.get("alpha_max", 1.0)))
        alpha_range = st.slider(
            "Alpha range [min, max]",
            min_value=0.0,
            max_value=1.0,
            value=(current_alpha_min, current_alpha_max),
            step=0.01,
            format="%.2f",
            key="param_alpha_range_slider",
            help=(
                "Constrains alpha values to this range. "
                "With Uniform: draws from U(min, max). "
                "With Gmyrek data: only uses occupation scores within this range."
            ),
        )
        params["alpha_min"] = alpha_range[0]
        params["alpha_max"] = alpha_range[1]
        # Keep individual session-state keys in sync for preset loading
        st.session_state["param_alpha_min"] = alpha_range[0]
        st.session_state["param_alpha_max"] = alpha_range[1]

    st.sidebar.markdown("---")
    log_decisions = st.sidebar.checkbox(
        "Log investment decisions",
        value=False,
        key="log_decisions_checkbox",
        help=(
            "When enabled, every automation evaluation is recorded "
            "(both 'automate' and 'not automate'). After the run a "
            "Download investment log (Excel) button appears. "
            "Slows the run slightly."
        ),
    )
    run_clicked = st.sidebar.button("Run simulation", use_container_width=True, type="primary")
    return run_clicked, run_label, int(n_steps), selected_modes, params, log_decisions


def _preview_signature(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def _render_preview_state(cache_key: str, current_signature: str, empty_message: str):
    cached = st.session_state.get(cache_key)
    if cached is None:
        st.info(empty_message)
        return None
    if cached.get("signature") != current_signature:
        st.caption("Showing the last loaded version of this preview. Click `↻` to refresh it with the current settings.")
    return cached


def _default_preview_wages(params: dict) -> tuple[float, float]:
    total_tasks = int(params["n_producers"]) * 20
    high_anchor = min(int(params["n_high_skilled"]), max(1, total_tasks // 2))
    low_anchor = min(int(params["n_low_skilled"]), max(1, total_tasks // 2))
    default_high = float(params["a_h"] + params["b_h"] * high_anchor)
    default_low = float(params["a_l"] + params["b_l"] * low_anchor)
    return default_high, default_low


def _render_setup_previews(params: dict):
    st.subheader("Setup previews")
    st.caption("These previews only refresh when you click the `↻` button inside that specific preview, so changing a parameter no longer reloads everything.")

    alpha_preview = st.slider(
        "Preview alpha for the 20-task productivity table",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Alpha is the share of tasks that become routine. The rest become non-routine.",
    )

    high_wage_min = float(params["a_h"])
    high_wage_max = float(params["a_h"] + params["b_h"] * int(params["n_high_skilled"]))
    low_wage_min = float(params["a_l"])
    low_wage_max = float(params["a_l"] + params["b_l"] * int(params["n_low_skilled"]))
    ai_cost_min = float(params["k_ai_floor"])
    ai_cost_max = float(params["k_ai"])
    default_high_wage, default_low_wage = _default_preview_wages(params)
    default_high_wage = round(min(max(default_high_wage, min(high_wage_min, high_wage_max)), max(high_wage_min, high_wage_max)), 2)
    default_low_wage = round(min(max(default_low_wage, min(low_wage_min, low_wage_max)), max(low_wage_min, low_wage_max)), 2)
    default_ai_cost = round(float(params["k_ai"]), 2)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "Productivity by task",
            "Cost comparison by task",
            "Automation frontier plot",
            "Wage and AI cost curves",
            "Keynesian loop",
            "NPV waterfall",
            "NPV heatmap",
            "Employment protection",
        ]
    )

    with tab1:
        st.caption("For the selected alpha, tasks 1-20 are split into routine and non-routine exactly like the model does inside each producer.")
        with st.form("productivity_preview_form", border=False):
            _, refresh_col = st.columns([8, 1])
            with refresh_col:
                refresh_productivity = st.form_submit_button("↻", help="Load or refresh this preview.")
            productivity_signature = _preview_signature({"alpha": alpha_preview, "params": params})
            if refresh_productivity:
                st.session_state["productivity_preview_cache"] = {
                    "signature": productivity_signature,
                    "productivity_df": build_productivity_preview(alpha_preview, params, n_tasks=20),
                }
        productivity_cached = _render_preview_state(
            "productivity_preview_cache",
            productivity_signature,
            "Click `↻` to load the productivity preview.",
        )
        if productivity_cached:
            productivity_df = productivity_cached["productivity_df"]
            productivity_fig = build_productivity_preview_figure(productivity_df)
            st.pyplot(productivity_fig, use_container_width=True)
            plt.close(productivity_fig)
            st.dataframe(productivity_df, use_container_width=True, hide_index=True)

    with tab2:
        st.caption("This tab helps you compare humans and AI task by task before running the model.")
        st.markdown(
            """
            `Unit cost` means `cost / productivity`.
            It answers: *how many euros do I need to pay for one unit of task output?*
            Lower is better.

            `Productivity per euro` means `productivity / cost`.
            It answers the same question from the other direction: *how much output do I get for one euro?*
            Higher is better.
            """
        )
        with st.form("cost_preview_form", border=False):
            metric_view = st.radio(
                "Comparison view",
                options=["unit_cost", "productivity_per_euro"],
                format_func=lambda value: "Lower is cheaper: cost per unit of output" if value == "unit_cost" else "Higher is better: output per euro",
                horizontal=True,
                key="cost_comparison_metric_view",
            )
            control_col1, control_col2, control_col3, control_col4 = st.columns([1, 1, 1, 0.25])
            with control_col1:
                preview_wage_high = st.slider("Preview high-skill wage", min_value=round(min(high_wage_min, high_wage_max), 2), max_value=round(max(high_wage_min, high_wage_max), 2), value=default_high_wage, step=0.05)
            with control_col2:
                preview_wage_low = st.slider("Preview low-skill wage", min_value=round(min(low_wage_min, low_wage_max), 2), max_value=round(max(low_wage_min, low_wage_max), 2), value=default_low_wage, step=0.05)
            with control_col3:
                preview_ai_cost = st.slider("Preview AI rental cost", min_value=round(min(ai_cost_min, ai_cost_max), 2), max_value=round(max(ai_cost_min, ai_cost_max), 2), value=default_ai_cost, step=0.05)
            with control_col4:
                refresh_cost = st.form_submit_button("↻", help="Load or refresh this preview.")
            cost_signature = _preview_signature({
                "alpha": alpha_preview,
                "params": params,
                "metric_view": metric_view,
                "wage_high": preview_wage_high,
                "wage_low": preview_wage_low,
                "ai_cost": preview_ai_cost,
            })
            if refresh_cost:
                assignment = estimate_initial_assignment(alpha_preview, params, n_tasks_per_producer=20)
                cost_df = build_cost_comparison_preview(alpha_preview, params, n_tasks=20, wage_high_override=preview_wage_high, wage_low_override=preview_wage_low, ai_cost_override=preview_ai_cost)
                st.session_state["cost_preview_cache"] = {
                    "signature": cost_signature,
                    "cost_df": cost_df,
                    "metric_view": metric_view,
                    "assignment": assignment,
                    "wage_high": preview_wage_high,
                    "wage_low": preview_wage_low,
                }
        cost_cached = _render_preview_state("cost_preview_cache", cost_signature, "Click `↻` to load the cost comparison preview.")
        if cost_cached:
            cost_df = cost_cached["cost_df"]
            assignment = cost_cached["assignment"]
            total_tasks = assignment["total_tasks"]
            scenario_counts = cost_df["cheapest_input"].value_counts()
            low_cheapest_tasks = int(scenario_counts.get("low", 0) * int(params["n_producers"]))
            high_cheapest_tasks = int(scenario_counts.get("high", 0) * int(params["n_producers"]))
            ai_cheapest_tasks = int(scenario_counts.get("AI", 0) * int(params["n_producers"]))
            high_supply = max(int(params["n_high_skilled"]), 1)
            low_supply = max(int(params["n_low_skilled"]), 1)
            if float(params["b_h"]) > 0:
                implied_high_employed_from_wage = max(0.0, min(float(params["n_high_skilled"]), (float(cost_cached["wage_high"]) - float(params["a_h"])) / float(params["b_h"])))
            else:
                implied_high_employed_from_wage = float(params["n_high_skilled"]) if cost_cached["wage_high"] >= float(params["a_h"]) else 0.0
            if float(params["b_l"]) > 0:
                implied_low_employed_from_wage = max(0.0, min(float(params["n_low_skilled"]), (float(cost_cached["wage_low"]) - float(params["a_l"])) / float(params["b_l"])))
            else:
                implied_low_employed_from_wage = float(params["n_low_skilled"]) if cost_cached["wage_low"] >= float(params["a_l"]) else 0.0
            implied_high_unemployment_rate = 100 * max(0.0, 1 - implied_high_employed_from_wage / high_supply)
            implied_low_unemployment_rate = 100 * max(0.0, 1 - implied_low_employed_from_wage / low_supply)
            illustrative_high_employment = min(high_cheapest_tasks, high_supply)
            illustrative_low_employment = min(low_cheapest_tasks, low_supply)
            illustrative_high_employment_rate = 100 * illustrative_high_employment / high_supply
            illustrative_low_employment_rate = 100 * illustrative_low_employment / low_supply
            illustrative_high_unemployment_rate = 100 - illustrative_high_employment_rate
            illustrative_low_unemployment_rate = 100 - illustrative_low_employment_rate
            ai_cheapest_share = 100 * ai_cheapest_tasks / max(total_tasks, 1)
            high_task_share = 100 * high_cheapest_tasks / max(total_tasks, 1)
            low_task_share = 100 * low_cheapest_tasks / max(total_tasks, 1)
            stat1, stat2, stat3 = st.columns(3)
            stat1.metric("Illustrative high-skill employment", f"{illustrative_high_employment_rate:.1f}%")
            stat2.metric("Illustrative low-skill employment", f"{illustrative_low_employment_rate:.1f}%")
            stat3.metric("Tasks where AI is cheapest", f"{ai_cheapest_share:.1f}%")
            st.caption(f"Under this preview, high-skill unemployment would be about {illustrative_high_unemployment_rate:.1f}% and low-skill unemployment about {illustrative_low_unemployment_rate:.1f}% if every task used the cheapest option shown here.")
            st.caption(f"For the loaded preview wages, the wage formulas imply about {implied_high_employed_from_wage:.0f} employed high-skill workers and {implied_low_employed_from_wage:.0f} employed low-skill workers.")
            st.caption(f"Under the loaded cost settings, the cheapest input would get about {ai_cheapest_share:.1f}% of tasks for AI, {high_task_share:.1f}% for high-skill labour, and {low_task_share:.1f}% for low-skill labour.")
            cost_fig = build_cost_comparison_figure(cost_df, metric=cost_cached["metric_view"])
            st.pyplot(cost_fig, use_container_width=True)
            plt.close(cost_fig)
            display_columns = ["task_id", "task_type", "complexity_index", "productivity_low", "productivity_high", "productivity_ai"]
            if cost_cached["metric_view"] == "unit_cost":
                display_columns.extend(["unit_cost_low", "unit_cost_high", "unit_cost_ai", "cheapest_input"])
            else:
                display_columns.extend(["productivity_per_euro_low", "productivity_per_euro_high", "productivity_per_euro_ai", "best_value_input"])
            st.dataframe(cost_df[display_columns], use_container_width=True, hide_index=True)

    with tab3:
        st.caption("The automation frontier shows how human and AI economics move across task complexity, separately for routine and non-routine tasks.")
        with st.form("frontier_preview_form", border=False):
            frontier_metric = st.radio("Frontier view", options=["unit_cost", "productivity_per_euro"], format_func=lambda value: "Unit cost" if value == "unit_cost" else "Productivity per euro", horizontal=True, key="frontier_metric_view")
            control_col1, control_col2, control_col3, control_col4 = st.columns([1, 1, 1, 0.25])
            with control_col1:
                frontier_wage_high = st.slider("Preview high-skill wage", min_value=round(min(high_wage_min, high_wage_max), 2), max_value=round(max(high_wage_min, high_wage_max), 2), value=default_high_wage, step=0.05, key="frontier_wage_high")
            with control_col2:
                frontier_wage_low = st.slider("Preview low-skill wage", min_value=round(min(low_wage_min, low_wage_max), 2), max_value=round(max(low_wage_min, low_wage_max), 2), value=default_low_wage, step=0.05, key="frontier_wage_low")
            with control_col3:
                frontier_ai_cost = st.slider("Preview AI rental cost", min_value=round(min(ai_cost_min, ai_cost_max), 2), max_value=round(max(ai_cost_min, ai_cost_max), 2), value=default_ai_cost, step=0.05, key="frontier_ai_cost")
            with control_col4:
                refresh_frontier = st.form_submit_button("↻", help="Load or refresh this preview.")
            frontier_signature = _preview_signature({"alpha": alpha_preview, "params": params, "metric": frontier_metric, "wage_high": frontier_wage_high, "wage_low": frontier_wage_low, "ai_cost": frontier_ai_cost})
            if refresh_frontier:
                st.session_state["frontier_preview_cache"] = {
                    "signature": frontier_signature,
                    "metric": frontier_metric,
                    "cost_df": build_cost_comparison_preview(alpha_preview, params, n_tasks=20, wage_high_override=frontier_wage_high, wage_low_override=frontier_wage_low, ai_cost_override=frontier_ai_cost),
                }
        frontier_cached = _render_preview_state("frontier_preview_cache", frontier_signature, "Click `↻` to load the automation frontier.")
        if frontier_cached:
            frontier_fig = build_automation_frontier_figure(frontier_cached["cost_df"], metric=frontier_cached["metric"])
            st.pyplot(frontier_fig, use_container_width=True)
            plt.close(frontier_fig)

    with tab4:
        st.caption("The wage lines show the target wage curves. The two horizontal lines show the AI cost at the start and the long-run floor.")
        with st.form("wage_preview_form", border=False):
            _, refresh_col = st.columns([8, 1])
            with refresh_col:
                refresh_wage = st.form_submit_button("↻", help="Load or refresh this preview.")
            wage_signature = _preview_signature({"params": params})
            if refresh_wage:
                st.session_state["wage_preview_cache"] = {"signature": wage_signature}
        wage_cached = _render_preview_state("wage_preview_cache", wage_signature, "Click `↻` to load the wage and AI cost preview.")
        if wage_cached:
            wage_fig = build_wage_preview_figure(params, n_high_skilled=int(params["n_high_skilled"]), n_low_skilled=int(params["n_low_skilled"]))
            st.pyplot(wage_fig, use_container_width=True)
            plt.close(wage_fig)

    with tab5:
        st.caption("A quick pre-run stress test of the Keynesian demand feedback using your current settings.")
        with st.form("keynesian_preview_form", border=False):
            total_tasks_guess = int(params["n_producers"]) * 20
            total_employed_guess = min(total_tasks_guess, int(params["n_high_skilled"]) + int(params["n_low_skilled"]))
            displaced_workers = st.slider("Illustrative displaced workers", min_value=0, max_value=max(0, total_employed_guess), value=min(100, total_employed_guess), step=1)
            default_q = max(1.0, round(total_employed_guess / max(int(params["n_producers"]), 1) * 1.5, 1))
            demo_q = st.slider("Illustrative output level Q", min_value=1.0, max_value=max(5.0, float(total_tasks_guess)), value=float(default_q), step=0.5)
            refresh_keynes = st.form_submit_button("↻", help="Load or refresh this preview.")
            keynes_signature = _preview_signature({"alpha": alpha_preview, "params": params, "displaced_workers": displaced_workers, "demo_q": demo_q})
            if refresh_keynes:
                assignment_guess = estimate_initial_assignment(alpha_preview, params, n_tasks_per_producer=20)
                st.session_state["keynes_preview_cache"] = {
                    "signature": keynes_signature,
                    "assignment": assignment_guess,
                    "displaced_workers": displaced_workers,
                    "demo_q": demo_q,
                }
        keynes_cached = _render_preview_state("keynes_preview_cache", keynes_signature, "Click `↻` to load the Keynesian loop preview.")
        if keynes_cached:
            assignment = keynes_cached["assignment"]
            employed_high = assignment["employed_high"]
            employed_low = assignment["employed_low"]
            total_employed = assignment["filled_tasks"]
            total_tasks = assignment["total_tasks"]
            avg_wage = (assignment["wage_high"] * employed_high + assignment["wage_low"] * employed_low) / max(total_employed, 1)
            st.caption(f"Initial employment is capped by available tasks: {int(params['n_producers'])} firms x 20 tasks = {total_tasks} tasks. This preview currently fills about {total_employed} of those tasks ({employed_high} high-skill, {employed_low} low-skill).")
            initial_wage_bill = total_employed * avg_wage
            A_initial = float(params["A_base"] + params["gamma"] * initial_wage_bill)
            p_initial = max(0.01, A_initial - params["B"] * float(keynes_cached["demo_q"]))
            post_wage_bill = max(0.0, initial_wage_bill - int(keynes_cached["displaced_workers"]) * avg_wage)
            A_after = float(params["A_base"] + params["gamma"] * post_wage_bill)
            p_after = max(0.01, A_after - params["B"] * float(keynes_cached["demo_q"]))
            delta_p = p_after - p_initial
            pct_change_p = (delta_p / p_initial * 100) if p_initial else 0.0
            delta_class = "loop-delta-pos" if delta_p >= 0 else "loop-delta-neg"
            max_post_wage_bill = 0.0
            max_A_after = float(params["A_base"] + params["gamma"] * max_post_wage_bill)
            max_p_after = max(0.01, max_A_after - params["B"] * float(keynes_cached["demo_q"]))
            max_delta_p = max_p_after - p_initial
            max_pct_change_p = (max_delta_p / p_initial * 100) if p_initial else 0.0
            st.markdown(f"""
            <div class="loop-grid">
                <div class="loop-card"><div class="loop-title">Step 1 - Initial employment</div><div class="loop-value">{total_employed}</div><div class="loop-sub">The model can only fill available tasks at the start. With {int(params['n_producers'])} firms and 20 tasks each, there are {total_tasks} tasks in total. This preview currently fills about {employed_high} high-skill and {employed_low} low-skill tasks.</div></div>
                <div class="loop-card"><div class="loop-title">Step 2 - Wage bill Y</div><div class="loop-value">{initial_wage_bill:.2f}</div><div class="loop-sub">Approximate average wage is {avg_wage:.2f}, so {total_employed} employed workers x {avg_wage:.2f} is about {initial_wage_bill:.2f}</div></div>
                <div class="loop-card"><div class="loop-title">Step 3 - Demand shifter A (wage channel only)</div><div class="loop-value">{A_initial:.2f}</div><div class="loop-sub">Wage-led part of the Kaleckian loop: A_wage = {params['A_base']:.2f} + {params['gamma']:.4f} x {initial_wage_bill:.2f} = {A_initial:.2f}. The full demand shifter also includes gamma_pi x profit_income, which is computed inside the model and not previewed here.</div></div>
                <div class="loop-card"><div class="loop-title">Step 4 - Illustrative price p</div><div class="loop-value">{p_initial:.2f}</div><div class="loop-sub">Using your chosen output assumption Q of about {float(keynes_cached['demo_q']):.2f}: p = {A_initial:.2f} - {params['B']:.2f} x {float(keynes_cached['demo_q']):.2f} = {p_initial:.2f}</div></div>
            </div>
            <div class="loop-grid">
                <div class="loop-card"><div class="loop-title">Shock - displaced workers</div><div class="loop-value">{int(keynes_cached['displaced_workers'])}</div><div class="loop-sub">This slider lets you estimate how a labour-displacement shock travels through Employment -> Y -> A -> p before running the full model.</div></div>
                <div class="loop-card"><div class="loop-title">After shock - new Y</div><div class="loop-value">{post_wage_bill:.2f}</div><div class="loop-sub">Wage bill falls by about {int(keynes_cached['displaced_workers']) * avg_wage:.2f}, so Y goes from {initial_wage_bill:.2f} to {post_wage_bill:.2f}.</div></div>
                <div class="loop-card"><div class="loop-title">After shock - new A (wage channel only)</div><div class="loop-value">{A_after:.2f}</div><div class="loop-sub">Lower Y means lower wage-led demand: A_wage = {params['A_base']:.2f} + {params['gamma']:.4f} x {post_wage_bill:.2f} = {A_after:.2f}. In the full model the profit-income channel (gamma_pi x Pi) partially offsets this drop, because AI redistributes income from labour to capital.</div></div>
                <div class="loop-card"><div class="loop-title">After shock - price effect</div><div class="loop-value {delta_class}">{delta_p:.2f}</div><div class="loop-sub">If Q stays around {float(keynes_cached['demo_q']):.2f}, price changes by {delta_p:.2f}, which is about {pct_change_p:.2f}% relative to the initial price.</div></div>
            </div>
            <div class="loop-grid">
                <div class="loop-card"><div class="loop-title">Maximum Keynesian hit</div><div class="loop-value">{max_delta_p:.2f}</div><div class="loop-sub">This is the largest pure Keynesian price drop in this simplified preview: if all current labour income disappeared, Y would fall to 0, A would fall to {max_A_after:.2f}, and price would move from {p_initial:.2f} to {max_p_after:.2f} if Q stayed unchanged.</div></div>
                <div class="loop-card"><div class="loop-title">Maximum percentage effect</div><div class="loop-value {delta_class}">{max_pct_change_p:.2f}%</div><div class="loop-sub">This is not a full equilibrium result. It is a simple upper-bound style check for how strongly the Keynesian channel can bite through Y -> A -> p.</div></div>
                <div class="loop-card"><div class="loop-title">Why output and price matter</div><div class="loop-value">Model impact</div><div class="loop-sub">Lower price reduces firm revenue for a given output level. That can squeeze profits, reduce the ability to keep workers hired, and feed back into future wage income and demand. Output matters because price is computed from p = A - B x Q: higher Q pushes price down, while lower Q pushes price up.</div></div>
                <div class="loop-card"><div class="loop-title">How to read this preview</div><div class="loop-value">Rule of thumb</div><div class="loop-sub">A higher gamma makes the Keynesian loop stronger. A higher B makes price more sensitive to output. A bigger displacement shock reduces Y more strongly. A larger assumed Q makes price more exposed to demand weakness.</div></div>
            </div>
            """, unsafe_allow_html=True)

    with tab6:
        npv_modes = ["npv_naive", "npv_adaptive", "npv_mean_field"]
        st.caption("This tab walks through one stylized automation decision for one task.")
        st.markdown("""
            Read the waterfall like this:

            `Upfront investment`
            This is the one-off cost of automating the task today.

            `PV human cost avoided`
            This is the present value of the labour cost you avoid if the task is no longer done by a worker.

            `PV AI cost incurred`
            This is the present value of the AI rental cost you expect to pay after automating the task.

            `NPV`
            `NPV = - investment cost + PV(human cost avoided) - PV(AI cost incurred)`
            """)
        with st.form("npv_waterfall_form", border=False):
            wf_col1, wf_col2, wf_col3 = st.columns(3)
            with wf_col1:
                wf_wage_high = st.slider("Preview high-skill wage", min_value=round(min(high_wage_min, high_wage_max), 2), max_value=round(max(high_wage_min, high_wage_max), 2), value=default_high_wage, step=0.05, key="npv_waterfall_wage_high")
            with wf_col2:
                wf_wage_low = st.slider("Preview low-skill wage", min_value=round(min(low_wage_min, low_wage_max), 2), max_value=round(max(low_wage_min, low_wage_max), 2), value=default_low_wage, step=0.05, key="npv_waterfall_wage_low")
            with wf_col3:
                wf_ai_cost = st.slider("Preview AI rental cost", min_value=round(min(ai_cost_min, ai_cost_max), 2), max_value=round(max(ai_cost_min, ai_cost_max), 2), value=default_ai_cost, step=0.05, key="npv_waterfall_ai_cost")
            col1, col2, col3, col4 = st.columns([1, 1, 1, 0.25])
            with col1:
                npv_mode = st.selectbox("Expectation mode", options=npv_modes, format_func=lambda x: MODE_LABELS.get(x, x), key="npv_preview_mode")
                npv_task_type = st.selectbox("Task type", options=["routine", "non_routine"], key="npv_preview_task_type")
            with col2:
                npv_complexity = st.slider("Task complexity", min_value=1, max_value=20, value=5, step=1, key="npv_preview_complexity")
                adaptive_trend_high = st.number_input("Adaptive wage trend high-skill", value=0.0, step=0.01, format="%.4f", key="npv_preview_trend_high")
            with col3:
                adaptive_trend_low = st.number_input("Adaptive wage trend low-skill", value=0.0, step=0.01, format="%.4f", key="npv_preview_trend_low")
                mean_field_displacement_flow = st.number_input("Mean-field expected worker displacement flow", value=0.0, step=0.1, format="%.3f", key="npv_preview_automation_flow")
            with col4:
                load_npv_preview = st.form_submit_button("↻", help="Load or refresh this preview.")
            st.markdown("**Severance preview (Employment protection)** — pretend the task is currently filled by N vast workers with the chosen tenure. Severance enters the NPV as an extra upfront cost: NPV = −(I + severance) + PV(savings).")
            sv_col1, sv_col2, sv_col3, sv_col4 = st.columns(4)
            default_sev_rate = float(params.get("severance_rate", 1.0 / 3.0))
            default_max_tenure = max(20.0, float(params.get("init_tenure_max_years", 10.0)))
            with sv_col1:
                wf_severance_tenure = st.slider(
                    "Vast worker tenure (years)",
                    min_value=0.0, max_value=default_max_tenure, value=0.0, step=0.5,
                    key="npv_waterfall_sev_tenure",
                    help="Years of tenure of the worker(s) currently on this task. Higher tenure → larger severance liability.",
                )
            with sv_col2:
                wf_severance_n_workers = st.number_input(
                    "Vast workers on task", min_value=0, max_value=20, value=1, step=1,
                    key="npv_waterfall_sev_n_workers",
                    help="Number of vast (permanent) workers currently assigned to the task. Severance scales linearly with this count.",
                )
            with sv_col3:
                wf_severance_skill = st.selectbox(
                    "Severance wage basis", options=["high", "low"], index=0,
                    key="npv_waterfall_sev_skill",
                    help="Use the high-skill or low-skill preview wage when computing severance.",
                )
            with sv_col4:
                wf_severance_rate = st.slider(
                    "Severance rate override",
                    min_value=0.0, max_value=max(5.0, default_sev_rate * 3.0),
                    value=float(default_sev_rate), step=0.05,
                    key="npv_waterfall_sev_rate",
                    help=f"Months-of-wage per year of tenure. Default = parameter value ({default_sev_rate:.3f}). Bump it up to see how the cost channel bites.",
                )
            npv_signature = _preview_signature({"mode": npv_mode, "task_type": npv_task_type, "complexity_index": npv_complexity, "adaptive_trend_high": adaptive_trend_high, "adaptive_trend_low": adaptive_trend_low, "mean_field_displacement_flow": mean_field_displacement_flow, "alpha_preview": alpha_preview, "wage_high": wf_wage_high, "wage_low": wf_wage_low, "ai_cost": wf_ai_cost, "sev_tenure": wf_severance_tenure, "sev_n_workers": wf_severance_n_workers, "sev_skill": wf_severance_skill, "sev_rate": wf_severance_rate, "params": params})
            if load_npv_preview:
                st.session_state["npv_waterfall_loaded"] = {"signature": npv_signature, "mode": npv_mode, "task_type": npv_task_type, "complexity_index": npv_complexity, "adaptive_trend_high": adaptive_trend_high, "adaptive_trend_low": adaptive_trend_low, "mean_field_displacement_flow": mean_field_displacement_flow, "alpha_preview": alpha_preview, "wage_high": wf_wage_high, "wage_low": wf_wage_low, "ai_cost": wf_ai_cost, "sev_tenure": wf_severance_tenure, "sev_n_workers": wf_severance_n_workers, "sev_skill": wf_severance_skill, "sev_rate": wf_severance_rate}
        loaded_preview = _render_preview_state("npv_waterfall_loaded", npv_signature, "Click `↻` to load the NPV waterfall preview.")
        if loaded_preview:
            loaded_preview["mode"] = normalize_mode(loaded_preview["mode"])
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            summary_col1.metric("Loaded mode", MODE_LABELS.get(loaded_preview["mode"], loaded_preview["mode"]))
            summary_col2.metric("Loaded task", loaded_preview["task_type"].replace("_", " "))
            summary_col3.metric("Loaded complexity", int(loaded_preview["complexity_index"]))
            npv_preview = compute_example_npv_preview(
                params=params, alpha=loaded_preview["alpha_preview"],
                task_type=loaded_preview["task_type"],
                complexity_index=int(loaded_preview["complexity_index"]),
                mode=loaded_preview["mode"],
                adaptive_trend_high=float(loaded_preview["adaptive_trend_high"]),
                adaptive_trend_low=float(loaded_preview["adaptive_trend_low"]),
                mean_field_displacement_flow=float(loaded_preview.get("mean_field_displacement_flow", loaded_preview.get("rational_automation_flow", 0.0))),
                wage_high_override=float(loaded_preview["wage_high"]),
                wage_low_override=float(loaded_preview["wage_low"]),
                ai_cost_override=float(loaded_preview["ai_cost"]),
                severance_tenure_years=float(loaded_preview.get("sev_tenure", 0.0)),
                severance_n_workers=int(loaded_preview.get("sev_n_workers", 1)),
                severance_skill=str(loaded_preview.get("sev_skill", "high")),
                severance_rate_override=float(loaded_preview.get("sev_rate", default_sev_rate)),
            )
            task_label = f"{loaded_preview['task_type']}, complexity {loaded_preview['complexity_index']}"
            waterfall_fig = build_npv_waterfall_figure(npv_preview, MODE_LABELS.get(loaded_preview["mode"], loaded_preview["mode"]), task_label)
            st.pyplot(waterfall_fig, use_container_width=True)
            plt.close(waterfall_fig)
            sev_cost = float(npv_preview.get("severance_cost", 0.0))
            sev_caption = ""
            if sev_cost > 0:
                sev_caption = (
                    f" Severance = {npv_preview.get('severance_rate_used', 0.0):.3f} × €"
                    f"{npv_preview.get('severance_wage', 0.0):.2f} × "
                    f"{npv_preview.get('severance_tenure_years', 0.0):.1f}y × "
                    f"{npv_preview.get('severance_n_workers', 0)} workers = €{sev_cost:.2f}."
                )
            st.caption(
                f"NPV = -(I + severance) + discounted human cost avoided - discounted AI cost incurred = "
                f"{npv_preview['npv']:.2f}.{sev_caption}"
            )
            st.dataframe(npv_preview["annual_table"], use_container_width=True, hide_index=True)

    with tab7:
        npv_modes = ["npv_naive", "npv_adaptive", "npv_mean_field"]
        st.caption("This heatmap shows where automation looks attractive across task type and complexity.")
        with st.form("npv_heatmap_form", border=False):
            heatmap_mode = st.selectbox("Heatmap expectation mode", options=npv_modes, format_func=lambda x: MODE_LABELS.get(x, x), key="npv_heatmap_mode")
            hm_col1, hm_col2, hm_col3, hm_col4 = st.columns([1, 1, 1, 0.25])
            with hm_col1:
                hm_wage_high = st.slider("Preview high-skill wage", min_value=round(min(high_wage_min, high_wage_max), 2), max_value=round(max(high_wage_min, high_wage_max), 2), value=default_high_wage, step=0.05, key="npv_heatmap_wage_high")
            with hm_col2:
                hm_wage_low = st.slider("Preview low-skill wage", min_value=round(min(low_wage_min, low_wage_max), 2), max_value=round(max(low_wage_min, low_wage_max), 2), value=default_low_wage, step=0.05, key="npv_heatmap_wage_low")
            with hm_col3:
                hm_ai_cost = st.slider("Preview AI rental cost", min_value=round(min(ai_cost_min, ai_cost_max), 2), max_value=round(max(ai_cost_min, ai_cost_max), 2), value=default_ai_cost, step=0.05, key="npv_heatmap_ai_cost")
            with hm_col4:
                refresh_heatmap = st.form_submit_button("↻", help="Load or refresh this preview.")
            heatmap_signature = _preview_signature({"alpha": alpha_preview, "params": params, "mode": heatmap_mode, "wage_high": hm_wage_high, "wage_low": hm_wage_low, "ai_cost": hm_ai_cost, "adaptive_trend_high": float(st.session_state.get('npv_preview_trend_high', 0.0)), "adaptive_trend_low": float(st.session_state.get('npv_preview_trend_low', 0.0)), "mean_field_displacement_flow": float(st.session_state.get('npv_preview_automation_flow', 0.0))})
            if refresh_heatmap:
                st.session_state["npv_heatmap_loaded"] = {"signature": heatmap_signature, "alpha": alpha_preview, "mode": heatmap_mode, "wage_high": hm_wage_high, "wage_low": hm_wage_low, "ai_cost": hm_ai_cost, "adaptive_trend_high": float(st.session_state.get('npv_preview_trend_high', 0.0)), "adaptive_trend_low": float(st.session_state.get('npv_preview_trend_low', 0.0)), "mean_field_displacement_flow": float(st.session_state.get('npv_preview_automation_flow', 0.0))}
        heatmap_loaded = _render_preview_state("npv_heatmap_loaded", heatmap_signature, "Click `↻` to load the NPV heatmap.")
        if heatmap_loaded:
            heatmap_fig = build_npv_heatmap_figure(params=params, alpha=heatmap_loaded["alpha"], mode=heatmap_loaded["mode"], adaptive_trend_high=float(heatmap_loaded["adaptive_trend_high"]), adaptive_trend_low=float(heatmap_loaded["adaptive_trend_low"]), mean_field_displacement_flow=float(heatmap_loaded["mean_field_displacement_flow"]), max_complexity=20, wage_high_override=float(heatmap_loaded["wage_high"]), wage_low_override=float(heatmap_loaded["wage_low"]), ai_cost_override=float(heatmap_loaded["ai_cost"]))
            st.pyplot(heatmap_fig, use_container_width=True)
            plt.close(heatmap_fig)
            st.caption("This is usually easier to read than one average NPV number, because it shows where in the task space the investment logic becomes favourable.")

    with tab8:
        _render_ep_preview(params)


def _render_ep_preview(params: dict):
    """Static employment-protection previews before running the model."""
    import numpy as np

    ep_on = bool(params.get("employment_protection", True))

    st.markdown(
        """
        **How to use this tab**

        These previews are computed directly from your current parameter settings — no simulation needed.
        They answer three questions that are hard to read off from parameter sliders alone:

        1. **How does the contract mix at the start of the model affect severance exposure?**
        2. **How much does a worker's tenure raise the cost of automating their task?**
        3. **How does that extra cost shift the automation decision threshold?**
        4. **When does the contract chain clause fire, and what happens to the worker?**

        Use the sliders inside each tab to explore different wage and tenure scenarios.
        """
    )

    if not ep_on:
        st.info("Employment protection is currently disabled. Enable the 'Employment protection' toggle at the top of the sidebar to see these previews.")
        return

    sev_rate = float(params.get("severance_rate", 1 / 3))
    chain_limit = int(params.get("chain_limit", 36))
    steps_per_year = int(params.get("steps_per_year", 12))
    p_convert = float(params.get("p_convert", 0.5))
    init_share_vast = float(params.get("init_share_vast", 0.6))
    init_tenure_max_years = float(params.get("init_tenure_max_years", 10.0))
    chain_years = chain_limit / max(steps_per_year, 1)
    avg_tenure_vast = (chain_years + init_tenure_max_years) / 2

    default_high_wage, default_low_wage = _default_preview_wages(params)
    I_base = float(params.get("I_base", 20.0))
    complexity_scaling = float(params.get("complexity_scaling", 0.15))

    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "Contract composition",
        "Severance cost by tenure",
        "Automation cost uplift",
        "Contract chain limit timeline",
    ])

    # ── 1. Initial contract composition ───────────────────────────────────────
    with subtab1:
        st.markdown(
            f"""
            **What this shows**

            At the start of the simulation, employed workers are split into permanent (**vast**) and
            flexible (**flex**) contracts. The parameter `init_share_vast = {init_share_vast:.0%}` controls this split.

            - **Vast workers** are already on permanent contracts. They start with tenure drawn uniformly
              from **{chain_years:.1f} to {init_tenure_max_years:.1f} years** (average ≈ {avg_tenure_vast:.1f} years).
              This tenure immediately creates a severance liability: if a firm automates this worker's task,
              it must pay severance **right from step 1**.
            - **Flex workers** start with zero tenure, so they carry no initial severance cost.
              They only become expensive to replace once they accumulate tenure toward the chain limit.

            **The higher `init_share_vast`, the more severance friction the model starts with.**
            """
        )
        fig_comp, axes_comp = plt.subplots(1, 2, figsize=(11, 4))

        # Left: horizontal stacked bar
        ax_bar = axes_comp[0]
        ax_bar.barh(["Workforce"], [init_share_vast * 100], color="#2563eb", label=f"Permanent (vast)  {init_share_vast:.0%}")
        ax_bar.barh(["Workforce"], [(1 - init_share_vast) * 100], left=[init_share_vast * 100], color="#f97316", label=f"Flexible (flex)  {1 - init_share_vast:.0%}")
        ax_bar.set_xlim(0, 100)
        ax_bar.set_xlabel("Share of employed workforce (%)")
        ax_bar.set_title("Initial contract composition", fontsize=11)
        ax_bar.legend(loc="lower right", fontsize=9)
        for spine in ax_bar.spines.values():
            spine.set_visible(False)

        # Right: tenure distribution for vast workers (uniform)
        ax_ten = axes_comp[1]
        tenure_range = np.linspace(chain_years, init_tenure_max_years, 200)
        ax_ten.fill_between(tenure_range, 0, 1, color="#2563eb", alpha=0.35, label="Vast workers' initial tenure (uniform)")
        ax_ten.axvline(avg_tenure_vast, color="#2563eb", linewidth=2, linestyle="--", label=f"Average ≈ {avg_tenure_vast:.1f} yr")
        ax_ten.set_xlim(0, init_tenure_max_years * 1.1)
        ax_ten.set_ylim(0, 1.4)
        ax_ten.set_yticks([])
        ax_ten.set_xlabel("Years of tenure at simulation start")
        ax_ten.set_title("Tenure distribution of permanent workers at t=0", fontsize=11)
        ax_ten.legend(fontsize=9)
        ax_ten.grid(axis="x", alpha=0.2)

        fig_comp.tight_layout()
        st.pyplot(fig_comp, use_container_width=True)
        plt.close(fig_comp)

    # ── 2. Severance cost by tenure ────────────────────────────────────────────
    with subtab2:
        st.markdown(
            f"""
            **What this shows**

            When a firm automates a task that is currently filled by a worker, it must pay a **one-off severance
            payment** to that worker. The formula is:

            > **Severance = `severance_rate` × wage × tenure_years**

            With `severance_rate = {sev_rate:.3f}` (≈ {sev_rate:.2%}), a worker earning wage **w** who has been
            employed for **t years** costs **{sev_rate:.3f} × w × t** to let go.

            - Move the wage sliders to see how wages scale the cost.
            - The **orange dashed line** marks the average initial tenure of permanent workers ({avg_tenure_vast:.1f} yr),
              so you can read off a realistic starting severance burden.
            - Flex workers typically have low tenure at first, so their severance cost starts near zero.
            """
        )
        sv_col1, sv_col2 = st.columns(2)
        with sv_col1:
            sv_wage_high = st.slider(
                "High-skill wage (€)",
                min_value=float(params["a_h"]),
                max_value=max(float(params["a_h"]) + 0.01, float(params["a_h"] + params["b_h"] * int(params["n_high_skilled"]))),
                value=round(default_high_wage, 2),
                step=0.05,
                key="ep_prev_wage_high",
            )
        with sv_col2:
            sv_wage_low = st.slider(
                "Low-skill wage (€)",
                min_value=float(params["a_l"]),
                max_value=max(float(params["a_l"]) + 0.01, float(params["a_l"] + params["b_l"] * int(params["n_low_skilled"]))),
                value=round(default_low_wage, 2),
                step=0.05,
                key="ep_prev_wage_low",
            )

        tenure_cont = np.linspace(0, max(init_tenure_max_years, chain_years + 1), 200)
        sev_high_cont = sev_rate * sv_wage_high * tenure_cont
        sev_low_cont = sev_rate * sv_wage_low * tenure_cont

        fig_sev, ax_sev = plt.subplots(figsize=(10, 4.5))
        ax_sev.plot(tenure_cont, sev_high_cont, color="#2563eb", linewidth=2.2, label=f"High-skill  (wage = €{sv_wage_high:.2f})")
        ax_sev.plot(tenure_cont, sev_low_cont, color="#f97316", linewidth=2.2, label=f"Low-skill  (wage = €{sv_wage_low:.2f})")
        ax_sev.axvline(avg_tenure_vast, color="#94a3b8", linewidth=1.5, linestyle="--", label=f"Avg. initial tenure of permanent workers  ({avg_tenure_vast:.1f} yr)")
        ax_sev.axvline(chain_years, color="#dc2626", linewidth=1.5, linestyle=":", label=f"Chain limit  ({chain_years:.1f} yr)")

        # Annotate severance at avg tenure
        sev_h_at_avg = sev_rate * sv_wage_high * avg_tenure_vast
        sev_l_at_avg = sev_rate * sv_wage_low * avg_tenure_vast
        ax_sev.annotate(f"€{sev_h_at_avg:.2f}", xy=(avg_tenure_vast, sev_h_at_avg), xytext=(avg_tenure_vast + 0.3, sev_h_at_avg + 0.2), fontsize=8, color="#2563eb")
        ax_sev.annotate(f"€{sev_l_at_avg:.2f}", xy=(avg_tenure_vast, sev_l_at_avg), xytext=(avg_tenure_vast + 0.3, sev_l_at_avg - 0.3), fontsize=8, color="#f97316")

        ax_sev.set_xlabel("Worker tenure at the time of automation (years)")
        ax_sev.set_ylabel("€ (severance payment)")
        ax_sev.set_title(
            f"One-off severance cost a firm pays to automate an occupied task  (severance_rate = {sev_rate:.3f})",
            fontsize=11,
        )
        ax_sev.legend(fontsize=8)
        ax_sev.grid(alpha=0.2)
        fig_sev.tight_layout()
        st.pyplot(fig_sev, use_container_width=True)
        plt.close(fig_sev)

        m1, m2 = st.columns(2)
        m1.metric("Severance at avg tenure — high-skill", f"€{sev_h_at_avg:.3f}")
        m2.metric("Severance at avg tenure — low-skill", f"€{sev_l_at_avg:.3f}")

    # ── 3. Automation cost uplift ──────────────────────────────────────────────
    with subtab3:
        st.markdown(
            f"""
            **What this shows**

            In the NPV modes, a firm automates a task only if the NPV of future cost savings exceeds the
            **upfront investment cost** plus any hurdle.  Employment protection adds the severance payment on
            top of the base investment cost when the task is currently occupied.

            > **Effective investment cost = I_base × (1 + complexity_scaling × complexity) + severance**

            The left panel shows the absolute increase in cost across complexity levels.
            The right panel shows the same increase **as a percentage of the base cost**,
            making it easier to judge how much more attractive a task needs to be before the firm will automate.

            **Use the sliders to set a representative wage and tenure** (e.g. the average permanent worker).
            The dashed grey line is the cost without employment protection; the red line is the cost with it.
            The red area is the range of automation decisions that are **blocked by employment protection**
            — cases that would have been profitable without severance but now fall below the cost threshold.
            """
        )
        complexity_range = np.arange(1, 21)
        inv_cost = I_base * (1 + complexity_scaling * complexity_range)

        nbv_col1, nbv_col2 = st.columns(2)
        with nbv_col1:
            nbv_tenure = st.slider(
                "Representative tenure (years)",
                min_value=0.0,
                max_value=max(float(init_tenure_max_years), chain_years + 0.1),
                value=round(avg_tenure_vast, 1),
                step=0.5,
                key="ep_prev_npv_tenure",
                help=f"Set to the average tenure of permanent workers at the start ({avg_tenure_vast:.1f} yr) for a realistic baseline.",
            )
        with nbv_col2:
            nbv_wage = st.slider(
                "Representative wage (€)",
                min_value=float(params["a_l"]),
                max_value=max(float(params["a_l"]) + 0.01, float(params["a_h"] + params["b_h"] * int(params["n_high_skilled"]))),
                value=round((default_high_wage + default_low_wage) / 2, 2),
                step=0.05,
                key="ep_prev_npv_wage",
            )

        sev_cost = sev_rate * nbv_wage * nbv_tenure
        total_cost = inv_cost + sev_cost
        pct_increase = 100 * sev_cost / np.maximum(inv_cost, 1e-9)

        fig_npv, (ax_npv1, ax_npv2) = plt.subplots(1, 2, figsize=(13, 4.5))

        ax_npv1.plot(complexity_range, inv_cost, label="Without employment protection", color="#94a3b8", linewidth=1.8, linestyle="--")
        ax_npv1.plot(complexity_range, total_cost, label=f"With employment protection  (+€{sev_cost:.2f})", color="#ef4444", linewidth=2.2)
        ax_npv1.fill_between(complexity_range, inv_cost, total_cost, alpha=0.15, color="#ef4444", label="Automation blocked by severance")
        ax_npv1.set_xlabel("Task complexity index")
        ax_npv1.set_ylabel("€ (upfront investment cost  I + severance)")
        ax_npv1.set_title("Upfront investment cost with and without employment protection", fontsize=10)
        ax_npv1.legend(fontsize=8)
        ax_npv1.grid(alpha=0.2)
        ax_npv1.annotate(
            f"Severance = €{sev_cost:.2f}\n({sev_rate:.2%} × €{nbv_wage:.2f} × {nbv_tenure:.1f}yr)",
            xy=(10, (inv_cost[9] + total_cost[9]) / 2),
            xytext=(13, inv_cost[9] - 2),
            fontsize=8,
            color="#ef4444",
            arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1),
        )

        ax_npv2.bar(complexity_range, pct_increase, color="#f97316", width=0.7, alpha=0.85)
        ax_npv2.axhline(pct_increase.mean(), color="#dc2626", linewidth=1.5, linestyle="--", label=f"Average  {pct_increase.mean():.1f}%")
        ax_npv2.set_xlabel("Task complexity index")
        ax_npv2.set_ylabel("Severance as % of base investment cost")
        ax_npv2.set_title(
            "How much does employment protection raise the\nautomation investment hurdle? (%)",
            fontsize=10,
        )
        ax_npv2.legend(fontsize=8)
        ax_npv2.grid(axis="y", alpha=0.2)
        fig_npv.suptitle(
            f"Tenure = {nbv_tenure:.1f} yr  |  wage = €{nbv_wage:.2f}  |  severance = €{sev_cost:.2f}",
            fontsize=10,
        )
        fig_npv.tight_layout()
        st.pyplot(fig_npv, use_container_width=True)
        plt.close(fig_npv)

        m1, m2, m3 = st.columns(3)
        m1.metric("Severance for this worker", f"€{sev_cost:.3f}")
        m2.metric("Cost uplift at complexity 1", f"+{pct_increase[0]:.1f}%", delta=f"€{inv_cost[0]:.2f} → €{total_cost[0]:.2f}", delta_color="inverse")
        m3.metric("Cost uplift at complexity 20", f"+{pct_increase[-1]:.1f}%", delta=f"€{inv_cost[-1]:.2f} → €{total_cost[-1]:.2f}", delta_color="inverse")

    # ── 4. Contract chain limit timeline ──────────────────────────────────────
    with subtab4:
        st.markdown(
            f"""
            **What this shows**

            Flex workers can be hired and fired freely, but Dutch law (the *ketenregeling*) limits how long
            a worker can stay on a flex contract. Once a flex worker reaches `chain_limit = {chain_limit} steps`
            (= **{chain_years:.1f} years** at {steps_per_year} steps/year), the firm must choose:

            - With probability **`p_convert` = {p_convert:.0%}**: convert the worker to a **permanent (vast) contract**.
              This raises future severance exposure — the worker now accumulates tenure on a vast contract.
            - With probability **{1 - p_convert:.0%}**: **do not renew** the contract.
              The worker leaves the firm; the task becomes vacant and can be re-filled or automated.

            **Why does this matter for automation?**
            Near the chain limit, firms may decide to automate a task *before* it triggers conversion,
            because once a worker becomes permanent they become substantially more expensive to remove.
            The chain limit thus creates a **pre-emptive automation incentive**.
            """
        )
        # ── Redesigned timeline: left = flex bar, right = outcome boxes ──────
        # Use a figure split into two axes: left (timeline) and right (outcomes)
        fig_keten, (ax_left, ax_right) = plt.subplots(
            1, 2, figsize=(13, 3.8),
            gridspec_kw={"width_ratios": [1.0, 1.2]},
        )
        fig_keten.suptitle("Contract chain limit mechanism (ketenregeling)", fontsize=12, fontweight="bold")

        # ── Left panel: flex tenure timeline ──────────────────────────────────
        ax_left.set_xlim(0, chain_years * 1.15)
        ax_left.set_ylim(0, 1)
        ax_left.set_yticks([])
        for spine in ax_left.spines.values():
            spine.set_visible(False)

        # Flex bar
        bar_y, bar_h = 0.42, 0.22
        ax_left.barh(bar_y + bar_h / 2, chain_years, height=bar_h, left=0,
                     color="#f97316", alpha=0.82, zorder=2)
        ax_left.text(chain_years / 2, bar_y + bar_h / 2,
                     "Flex contract period", ha="center", va="center",
                     fontsize=10, color="white", fontweight="bold")

        # Chain limit vertical line
        ax_left.axvline(chain_years, color="#1e293b", linewidth=2.5, linestyle="--", zorder=3)
        ax_left.text(chain_years, bar_y + bar_h + 0.14,
                     f"Chain limit\n{chain_years:.1f} yr  ({chain_limit} steps)",
                     ha="center", va="bottom", fontsize=9,
                     color="#1e293b", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1e293b", alpha=0.9))

        # x-axis labels: just 0, midpoint, chain_years
        ax_left.set_xticks([0, chain_years / 2, chain_years])
        ax_left.set_xticklabels(["0 yr", f"{chain_years / 2:.1f} yr", f"{chain_years:.1f} yr"])
        ax_left.set_xlabel("Years of flex tenure", fontsize=9)
        ax_left.xaxis.set_tick_params(which="both", bottom=True)

        # ── Right panel: outcome boxes ─────────────────────────────────────────
        ax_right.set_xlim(0, 1)
        ax_right.set_ylim(0, 1)
        ax_right.set_yticks([])
        ax_right.set_xticks([])
        for spine in ax_right.spines.values():
            spine.set_visible(False)
        ax_right.set_xlabel(" ", fontsize=9)  # keep height consistent

        # Incoming arrow from left panel (uses figure coords via annotation connection)
        branch_x = 0.08
        top_y, bot_y = 0.72, 0.28

        # Vertical connector line
        ax_right.plot([branch_x, branch_x], [bot_y, top_y], color="#475569", linewidth=2, solid_capstyle="round")

        # Top branch: permanent conversion
        ax_right.annotate(
            "",
            xy=(branch_x + 0.06, top_y),
            xytext=(branch_x, top_y),
            arrowprops=dict(arrowstyle="->", color="#2563eb", lw=2.0),
        )
        ax_right.text(
            branch_x + 0.08, top_y,
            f"Permanent (vast) contract\nprobability: p_convert = {p_convert:.0%}\n"
            f"→ Worker stays; future severance exposure rises",
            ha="left", va="center", fontsize=9, color="#1e3a8a",
            bbox=dict(boxstyle="round,pad=0.45", fc="#dbeafe", ec="#2563eb", alpha=0.92),
        )

        # Bottom branch: non-renewal
        ax_right.annotate(
            "",
            xy=(branch_x + 0.06, bot_y),
            xytext=(branch_x, bot_y),
            arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2.0),
        )
        ax_right.text(
            branch_x + 0.08, bot_y,
            f"Contract not renewed\nprobability: {1 - p_convert:.0%}\n"
            f"→ Worker leaves; task vacant (hire or automate)",
            ha="left", va="center", fontsize=9, color="#7f1d1d",
            bbox=dict(boxstyle="round,pad=0.45", fc="#fee2e2", ec="#dc2626", alpha=0.92),
        )

        # Label for the branch column
        ax_right.text(0.0, 0.5, "At chain\nlimit:", ha="center", va="center",
                      fontsize=9, color="#475569", style="italic")

        fig_keten.tight_layout()
        st.pyplot(fig_keten, use_container_width=True)
        plt.close(fig_keten)

        keten_df = pd.DataFrame([
            {"Parameter": "chain_limit", "Value (steps)": chain_limit, "Value (years)": f"{chain_years:.2f}"},
            {"Parameter": "steps_per_year", "Value (steps)": steps_per_year, "Value (years)": "—"},
            {"Parameter": "p_convert", "Value (steps)": f"{p_convert:.0%}", "Value (years)": "prob. of permanent conversion"},
            {"Parameter": "Non-renewal prob.", "Value (steps)": f"{1 - p_convert:.0%}", "Value (years)": "prob. of contract non-renewal"},
        ])
        st.dataframe(keten_df, use_container_width=False, hide_index=True)


def _render_graph_maker(
    series_dict: dict,
    key_prefix: str = "gm",
    default_metrics: list | None = None,
):
    """
    Custom graph builder — reusable for both the dashboard run and the experimenter.

    Parameters
    ----------
    series_dict : dict
        Mapping of label → pd.DataFrame (columns = metrics, index = step).
        For the dashboard run pass {MODE_LABELS[mode]: df for mode, df in bundle.results.items()}.
        For the experimenter pass {exp_name: df for exp_name, df in ...}.
    key_prefix : str
        Unique prefix for all Streamlit widget keys (avoids collisions between sections).
    default_metrics : list | None
        Pre-selected metrics. Defaults to the first two available.
    """
    import math

    st.markdown("### Custom graph builder")
    st.caption(
        "Pick any combination of metrics and series, then choose how to display them. "
        "**Grid** gives each metric its own panel. "
        "**Overlaid** puts everything on one set of axes (useful when metrics share the same scale). "
        "**2 × 1** forces exactly two rows, one panel per metric column."
    )

    if not series_dict:
        st.info("No results available yet — run the simulation first.")
        return

    # Collect available metrics (present in at least one series)
    all_cols: set[str] = set()
    for df in series_dict.values():
        all_cols.update(df.columns)
    available = sorted(c for c in all_cols if c not in {"step", "adoption_mode", "ai_irreversible"})

    if not available:
        st.info("No plottable metrics found.")
        return

    _defaults = default_metrics or available[:min(2, len(available))]

    gm_col1, gm_col2, gm_col3 = st.columns([2, 2, 1])
    with gm_col1:
        selected_metrics = st.multiselect(
            "Metrics",
            options=available,
            default=[m for m in _defaults if m in available],
            format_func=lambda m: METRIC_LABELS.get(m, m),
            key=f"{key_prefix}_metrics",
        )
    with gm_col2:
        all_series_names = list(series_dict.keys())
        selected_series = st.multiselect(
            "Series (modes / experiments)",
            options=all_series_names,
            default=all_series_names,
            key=f"{key_prefix}_series",
        )
    with gm_col3:
        layout = st.radio(
            "Layout",
            options=["Grid", "Overlaid", "2 × 1"],
            key=f"{key_prefix}_layout",
            index=0,
        )
        smooth_win = st.number_input(
            "Smoothing window",
            min_value=1, max_value=200, value=1, step=1,
            key=f"{key_prefix}_smooth",
            help="Rolling average window applied to all series. Set to 1 for no smoothing.",
        )

    if not selected_metrics:
        st.info("Select at least one metric.")
        return
    if not selected_series:
        st.info("Select at least one series.")
        return

    n = len(selected_metrics)

    # ── Build figure ──────────────────────────────────────────────────────────
    # Assign colors by metric and linestyles by series. In overlaid plots this
    # keeps multiple metrics visually distinct even when only one series exists.
    _PALETTE = ["#2563eb", "#f97316", "#16a34a", "#dc2626", "#7c3aed", "#0891b2", "#be185d", "#ca8a04"]
    _LS_CYCLE = ["-", "--", "-.", ":"]
    metric_styles = {
        metric: {"color": _PALETTE[i % len(_PALETTE)]}
        for i, metric in enumerate(selected_metrics)
    }
    series_styles = {
        name: {
            "color": _PALETTE[i % len(_PALETTE)],
            "ls": _LS_CYCLE[i % len(_LS_CYCLE)],
        }
        for i, name in enumerate(selected_series)
    }

    def _get_series(name: str, metric: str) -> pd.Series | None:
        df = series_dict.get(name)
        if df is None or metric not in df.columns:
            return None
        s = df[metric].dropna()
        if s.empty:
            return None
        if smooth_win > 1:
            s = s.rolling(smooth_win, min_periods=1).mean()
        return s

    if layout == "Overlaid":
        # ── Dual y-axis logic ─────────────────────────────────────────────────
        # For each metric compute its representative scale (median of abs values
        # across all selected series).  If the ratio between the largest and
        # smallest scale exceeds 5× we split: smaller-scale metrics → left axis,
        # larger-scale metrics → right axis.  Same-scale metrics share an axis.
        import math as _math

        def _metric_scale(metric: str) -> float:
            vals = []
            for sname in selected_series:
                s = _get_series(sname, metric)
                if s is not None and not s.empty:
                    vals.extend(s.abs().dropna().tolist())
            return float(np.median(vals)) if vals else 1.0

        scales = {m: _metric_scale(m) for m in selected_metrics}
        scale_vals = [v for v in scales.values() if v > 0]
        use_dual = (
            len(selected_metrics) >= 2
            and len(scale_vals) >= 2
            and (max(scale_vals) / max(min(scale_vals), 1e-12)) > 5.0
        )

        if use_dual:
            # Sort metrics: lowest median → left axis, rest → right axis
            sorted_metrics = sorted(selected_metrics, key=lambda m: scales.get(m, 0))
            # Find the natural break: left = metrics within 5× of the minimum scale
            min_scale = scales[sorted_metrics[0]]
            left_metrics  = [m for m in sorted_metrics if scales.get(m, 0) <= min_scale * 5.0]
            right_metrics = [m for m in sorted_metrics if scales.get(m, 0) >  min_scale * 5.0]
        else:
            left_metrics  = selected_metrics
            right_metrics = []

        fig_gm, ax_left = plt.subplots(figsize=(13, 4.5))
        ax_right = ax_left.twinx() if right_metrics else None

        # Extra line-styles for right axis to avoid identical look
        _LS_RIGHT = ["--", "-.", ":", "-"]

        def _plot_on_axis(ax, metrics_subset, ls_override=None):
            for metric in metrics_subset:
                for si, sname in enumerate(selected_series):
                    s = _get_series(sname, metric)
                    if s is None:
                        continue
                    label = f"{METRIC_LABELS.get(metric, metric)} — {sname}"
                    st_val = series_styles[sname]
                    ls = ls_override[si % len(ls_override)] if ls_override else st_val["ls"]
                    ax.plot(s.index, s, label=label,
                            color=metric_styles[metric]["color"], linestyle=ls, linewidth=1.8)

        _plot_on_axis(ax_left, left_metrics)
        if ax_right and right_metrics:
            _plot_on_axis(ax_right, right_metrics, ls_override=_LS_RIGHT)
            ax_right.set_ylabel(
                "  /  ".join(METRIC_LABELS.get(m, m) for m in right_metrics),
                fontsize=9, color="#6b7280",
            )
            # Combined legend from both axes
            h1, l1 = ax_left.get_legend_handles_labels()
            h2, l2 = ax_right.get_legend_handles_labels()
            ax_left.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
        else:
            ax_left.legend(fontsize=8, loc="best")

        ax_left.set_xlabel("Model step")
        ax_left.set_ylabel(
            "  /  ".join(METRIC_LABELS.get(m, m) for m in left_metrics),
            fontsize=9,
        )
        ax_left.grid(alpha=0.2)
        _title_suffix = f"  ({smooth_win}-step rolling avg)" if smooth_win > 1 else ""
        _dual_note = "  —  dual y-axis" if use_dual else ""
        ax_left.set_title(f"Custom graph{_title_suffix}{_dual_note}", fontsize=11)
        fig_gm.tight_layout()
        st.pyplot(fig_gm, use_container_width=True)
        plt.close(fig_gm)

    else:
        # Grid or 2×1
        if layout == "2 × 1":
            ncols = 1
            nrows = min(n, 2)
            metrics_to_show = selected_metrics[:2]
        else:
            ncols = min(n, 3)
            nrows = math.ceil(n / ncols)
            metrics_to_show = selected_metrics

        fig_gm, axes_gm = plt.subplots(
            nrows, ncols,
            figsize=(max(6 * ncols, 8), 4 * nrows),
            squeeze=False,
        )
        axes_flat = axes_gm.flatten()

        for i, metric in enumerate(metrics_to_show):
            ax = axes_flat[i]
            any_plotted = False
            for sname in selected_series:
                s = _get_series(sname, metric)
                if s is None:
                    continue
                st_val = series_styles[sname]
                ax.plot(s.index, s, label=sname, color=st_val["color"], linestyle=st_val["ls"], linewidth=1.8)
                any_plotted = True
            title = METRIC_LABELS.get(metric, metric)
            if smooth_win > 1:
                title += f"  ({smooth_win}-step avg)"
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Model step")
            ax.grid(alpha=0.2)
            if any_plotted:
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")

        # Hide unused axes
        for j in range(len(metrics_to_show), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig_gm.tight_layout()
        st.pyplot(fig_gm, use_container_width=True)
        plt.close(fig_gm)


def _render_run_overview(summary: pd.DataFrame):
    focus_columns = [
        "ai_adoption_rate_final",
        "employment_rate_low_final",
        "employment_rate_high_final",
        "gini_income_final",
        "skill_wage_premium_final",
        "price_final",
        "total_output_final",
    ]
    available_columns = [column for column in focus_columns if column in summary.columns]
    if available_columns:
        st.dataframe(summary[available_columns], use_container_width=True)
    else:
        st.dataframe(summary, use_container_width=True)


def _render_current_run(run_dir: Path | None):
    bundle = st.session_state.get("latest_bundle")
    if bundle is None:
        st.info("No run yet. Choose your parameters on the left and click `Run simulation`.")
        return

    summary = build_summary_dataframe(bundle)
    st.subheader("Latest run")
    if run_dir is not None:
        st.caption(f"Saved in `{run_dir}`")

    _render_run_overview(summary)

    # ── Investment decision log download ──────────────────────────────────────
    log_df = build_investment_log_dataframe(bundle)
    if not log_df.empty:
        with st.expander("Investment decision log", expanded=False):
            st.caption(
                f"**{len(log_df):,} evaluation records** across all modes. "
                "Each row is one automation evaluation (both 'automate' and 'not automate'). "
                "Download the Excel file to inspect the full breakdown."
            )
            # Quick preview: last 200 rows of the 'evaluated' decisions
            preview = log_df[log_df.get("evaluated", pd.Series(dtype=bool)) == True].tail(200)  # noqa: E712
            if not preview.empty:
                st.dataframe(preview, use_container_width=True, hide_index=True)
            excel_bytes = build_investment_log_excel_bytes(log_df)
            st.download_button(
                label="Download investment log (Excel)",
                data=excel_bytes,
                file_name=f"investment_log_{bundle.run_label or 'run'}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    elif st.session_state.get("log_decisions_checkbox", False):
        st.info("Investment log enabled but no decisions were recorded yet — re-run the simulation.")

    st.caption(
        "Reading guide: `Labour market plots` show the main outcomes you would usually discuss first. "
        "`Macro plots` show economy-wide mechanisms like labour share, profits and adoption channels. "
        "`Other plots` contain diagnostics that help explain why the headline outcomes move."
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Labour market plots", "Macro plots", "Other plots", "Automation complexity",
        "Firm inspector", "Employment protection",
        "Alpha × workforce", "Complexity × workforce",
    ])
    with tab1:
        st.caption("Start here for the main story: employment, wages, output, prices and overall AI adoption.")
        _lm_rolling_on = st.toggle(
            "Rolling average",
            value=False,
            key="lm_rolling_on",
            help="Smooth all lines with a rolling average to reduce step-to-step noise.",
        )
        _lm_rolling_win = 1
        if _lm_rolling_on:
            _lm_rolling_win = st.slider(
                "Window (steps)",
                min_value=2, max_value=50, value=10, step=1,
                key="lm_rolling_win",
            )
        labour_fig = build_labour_market_figure(bundle.results, rolling=_lm_rolling_win)
        st.pyplot(labour_fig, use_container_width=True)
        plt.close(labour_fig)
        _render_plot_metric_table(
            _flatten_panel_metrics(LABOUR_MARKET_PANELS),
            "Plot reference: metric, formula, economic meaning",
        )
    with tab2:
        st.caption("These plots show the macro transmission: income distribution, firm profitability, and the reactive/proactive AI channels.")
        _mac_rolling_on = st.toggle(
            "Rolling average",
            value=False,
            key="mac_rolling_on",
            help="Smooth all lines with a rolling average to reduce step-to-step noise.",
        )
        _mac_rolling_win = 1
        if _mac_rolling_on:
            _mac_rolling_win = st.slider(
                "Window (steps)",
                min_value=2, max_value=50, value=10, step=1,
                key="mac_rolling_win",
            )
        macro_fig = build_macro_figure(bundle.results, rolling=_mac_rolling_win)
        st.pyplot(macro_fig, use_container_width=True)
        plt.close(macro_fig)
        _render_plot_metric_table(
            _flatten_panel_metrics(MACRO_PANELS),
            "Plot reference: metric, formula, economic meaning",
        )
    with tab4:
        st.caption(
            "**Top row** — Automation timing scatter: each dot is one task that was automated. "
            "The x-axis shows when it happened, the y-axis shows how complex it was. "
            "Orange = routine, blue = non-routine. Simple tasks should cluster on the left, complex ones on the right. "
            "Only NPV modes record individual timing, so the ULC baseline has no scatter panel. "
            "**Bottom row** — End-state heatmap: each cell shows the fraction of tasks at that "
            "complexity level that ended up automated. 0% = white, 100% = dark green."
        )
        complexity_fig = build_automation_complexity_figure(bundle)
        st.pyplot(complexity_fig, use_container_width=True)
        plt.close(complexity_fig)
    with tab3:
        st.caption(
            "These are diagnostic plots. They are useful once you want to explain `why` adoption, wages or employment moved. "
            "Spiky event plots are normal here because they count discrete automation events."
        )
        other_figures = build_other_figures(bundle.results)
        if not other_figures:
            st.info("No additional plots are available for this run.")
        else:
            subtabs = st.tabs(list(other_figures.keys()))
            for subtab, (group_name, figure) in zip(subtabs, other_figures.items()):
                with subtab:
                    if group_name == "Adoption diagnostics":
                        st.caption(
                            "Read these as the investment side of the model. "
                            "`Share meeting adoption threshold` tells you how much automation opportunity is still left among the non-automated tasks."
                        )
                    elif group_name == "Cost diagnostics":
                        st.caption(
                            "These are input-cost plots. They compare the running cost of labour and AI, so they help explain why a mode starts preferring one input over another."
                        )
                    elif group_name == "Flow diagnostics":
                        st.caption(
                            "These are now shown cumulatively to make them easier to read over long runs. "
                            "They count realized automation events, split into the reactive and proactive adoption channels."
                        )
                    else:
                        st.caption("Additional diagnostics that are available in the model output but are less central than the main labour and macro views.")
                    st.pyplot(figure, use_container_width=True)
                    plt.close(figure)
                    group_metrics = OTHER_PLOT_GROUPS.get(group_name, {}).get("metrics", [])
                    if group_name == "Additional diagnostics":
                        excluded_metrics = {"w_min", "new_displaced_workers_this_step", "adoption_mode", "ai_irreversible"}
                        shown_metrics = []
                        for df in bundle.results.values():
                            for column in df.columns:
                                if column in excluded_metrics:
                                    continue
                                if column not in shown_metrics:
                                    shown_metrics.append(column)
                    else:
                        shown_metrics = group_metrics
                    _render_plot_metric_table(
                        shown_metrics,
                        "Plot reference: metric, formula, economic meaning",
                    )

    with tab5:
        st.caption(
            "Inspect individual firms at the end of the simulation. "
            "Select an adoption mode and a firm to see its tasks, automation status, and assigned workers."
        )
        if not bundle.models:
            st.info("Firm inspector is not available for loaded runs — model objects are not persisted. Re-run the simulation to use this feature.")
        else:
            inspector_mode = st.selectbox(
                "Adoption mode",
                options=bundle.modes,
                format_func=lambda m: MODE_LABELS.get(m, m),
                key="inspector_mode",
            )
            inspector_model = bundle.models[inspector_mode]
            producer_options = [
                (p.unique_id, f"Firm {p.unique_id}  (α={p.alpha:.2f}, output={p.output:.1f}, n_workers={sum(len(t.employees) for t in p.tasks)}, n_ai={sum(t.n_ai for t in p.tasks)})")
                for p in inspector_model.producers
            ]
            selected_pid = st.selectbox(
                "Select firm",
                options=[pid for pid, _ in producer_options],
                format_func=lambda pid: next(label for p_id, label in producer_options if p_id == pid),
                key="inspector_firm",
            )
            producer = next(p for p in inspector_model.producers if p.unique_id == selected_pid)

            n_tasks_total = len(producer.tasks)
            n_auto = sum(1 for t in producer.tasks if t.automated)
            n_workers_total = sum(len(t.employees) for t in producer.tasks)
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Tasks", n_tasks_total)
            col_b.metric("Automated", f"{n_auto} ({n_auto/n_tasks_total:.0%})")
            col_c.metric("Workers assigned", n_workers_total)
            col_d.metric("Output", f"{producer.output:.2f}")

            task_rows = []
            for t in sorted(producer.tasks, key=lambda t: (t.task_type, t.complexity_index)):
                n_high = sum(1 for w in t.employees if w.skill_level == "high")
                n_low  = sum(1 for w in t.employees if w.skill_level == "low")
                if t.automated:
                    status = "AI"
                    input_detail = f"{t.n_ai} AI unit(s)"
                    total_prod = t.n_ai * producer.productivity_ai(t)
                elif t.employees:
                    def _fmt_worker(w, _model=inspector_model):
                        parts = [w.skill_level]
                        if _model.employment_protection:
                            parts.append(w.contract_type)
                            tenure_yr = w.tenure / max(1, _model.steps_per_year)
                            parts.append(f"tenure={tenure_yr:.1f}y")
                        parts.append(f"wage={w.wage:.2f}")
                        return f"W{w.unique_id} ({', '.join(parts)})"
                    input_detail = ", ".join(_fmt_worker(w) for w in t.employees)
                    status = "Human"
                    total_prod = sum(producer.productivity_human(t, w.skill_level) for w in t.employees)
                else:
                    status = "Empty"
                    input_detail = "—"
                    total_prod = 0.0

                task_rows.append({
                    "Task ID": t.task_id,
                    "Type": t.task_type,
                    "Complexity": t.complexity_index,
                    "Status": status,
                    "n_workers_high": n_high,
                    "n_workers_low": n_low,
                    "n_ai": t.n_ai if t.automated else 0,
                    "Total productivity": round(total_prod, 3),
                    "Input": input_detail,
                })

            task_df = pd.DataFrame(task_rows)

            def _color_status(val):
                if val == "AI":
                    return "background-color: #d1fae5; color: #065f46"
                if val == "Empty":
                    return "background-color: #fee2e2; color: #991b1b"
                return ""

            st.dataframe(
                task_df.style.applymap(_color_status, subset=["Status"]),
                use_container_width=True,
                hide_index=True,
            )

    with tab6:
        st.caption(
            "Employment protection diagnostics: permanent/flex contract shares by skill group, "
            "tenure by contract type, contract chain-limit conversion and non-renewal events, "
            "and average severance cost per task."
        )
        ep_metrics = [
            "share_vast_high", "share_vast_low", "share_flex_high", "share_flex_low",
            "avg_tenure_years", "avg_tenure_vast_years", "avg_tenure_flex_years",
            "avg_severance_per_task", "conversions_this_step", "non_renewals_this_step",
        ]
        any_ep_data = any(
            metric in df.columns
            for df in bundle.results.values()
            for metric in ep_metrics
        )
        if not any_ep_data:
            st.info("No employment protection data found. Enable 'Employment Protection' in the sidebar and re-run the simulation.")
        else:
            # ── EP summary card ────────────────────────────────────────────────
            st.subheader("Employment protection summary")
            summary_cols = st.columns(4)
            for col_idx, (mode, df) in enumerate(bundle.results.items()):
                if col_idx >= 4:
                    break
                with summary_cols[col_idx]:
                    mode_label = MODE_LABELS.get(mode, mode)
                    # Final-step values
                    final_vast_high = float(df["share_vast_high"].iloc[-1]) if "share_vast_high" in df.columns else None
                    final_vast_low = float(df["share_vast_low"].iloc[-1]) if "share_vast_low" in df.columns else None
                    total_sev = float(df["avg_severance_per_task"].sum()) if "avg_severance_per_task" in df.columns else None
                    total_conv = int(df["conversions_this_step"].sum()) if "conversions_this_step" in df.columns else None
                    total_nonren = int(df["non_renewals_this_step"].sum()) if "non_renewals_this_step" in df.columns else None
                    lines = [f"**{mode_label}**"]
                    if final_vast_high is not None:
                        lines.append(f"Permanent share high-skill (end): {final_vast_high:.0%}")
                    if final_vast_low is not None:
                        lines.append(f"Permanent share low-skill (end): {final_vast_low:.0%}")
                    if total_sev is not None:
                        lines.append(f"Total severance paid: €{total_sev:.2f}")
                    if total_conv is not None:
                        lines.append(f"Chain limit conversions: {total_conv:,}")
                    if total_nonren is not None:
                        lines.append(f"Non-renewals: {total_nonren:,}")
                    st.markdown(
                        "<div class='metric-card'>" + "<br>".join(lines) + "</div>",
                        unsafe_allow_html=True,
                    )

            st.divider()
            ep_fig = build_employment_protection_figure(bundle.results)
            st.pyplot(ep_fig, use_container_width=True)
            plt.close(ep_fig)

    with tab7:
        st.markdown("#### Task status by alpha range")
        st.caption(
            "Each task at the last simulation step is assigned one status: "
            "**AI** (automated), **High-skill** (only high-skill workers), "
            "**Low-skill** (only low-skill workers), **Mixed** (both high- and low-skill), "
            "or **Empty** (no workers, not automated). "
            "Tasks belonging to firms with similar alpha values are grouped into fixed bins of 0.05 (0→1, 20 bins total)."
        )
        alpha_st_fig = build_alpha_status_figure(bundle)
        st.pyplot(alpha_st_fig, use_container_width=True)
        plt.close(alpha_st_fig)

        st.markdown("#### Absolute worker count by alpha range")
        st.caption(
            "Total worker-slots (AI units + high-skill + low-skill workers) in firms grouped "
            "by their alpha value. Alpha is binned in fixed steps of 0.05 from 0 to 1 (20 bins). "
            "High-alpha firms (many routine tasks) are expected to be more AI-heavy."
        )
        alpha_wf_fig = build_alpha_workforce_figure(bundle)
        st.pyplot(alpha_wf_fig, use_container_width=True)
        plt.close(alpha_wf_fig)

    with tab8:
        st.markdown("#### Task status by complexity")
        st.caption(
            "Each task at the last simulation step is assigned exactly one status: "
            "**AI** (automated), **High-skill** (only high-skill workers), "
            "**Low-skill** (only low-skill workers), **Mixed** (both high- and low-skill workers), "
            "or **Empty** (no workers, not automated). "
            "The bars show how many tasks fall into each status for every complexity level (1 = simplest)."
        )
        status_fig = build_task_complexity_status_figure(bundle)
        st.pyplot(status_fig, use_container_width=True)
        plt.close(status_fig)

        st.markdown("#### Absolute worker count by task complexity")
        st.caption(
            "Each panel shows one adoption mode and one task type (routine / non-routine). "
            "The x-axis is the task complexity index (1 = simplest). "
            "Bar height is the total number of worker-slots (AI units + high-skill + low-skill workers) "
            "across all tasks of that complexity, reflecting both task frequency and staffing intensity."
        )
        complexity_wf_fig = build_task_complexity_workforce_figure(bundle)
        st.pyplot(complexity_wf_fig, use_container_width=True)
        plt.close(complexity_wf_fig)

    # ── Custom graph builder (below all tabs) ─────────────────────────────────
    st.divider()
    with st.expander("Custom graph builder", expanded=False):
        gm_series = {
            MODE_LABELS.get(mode, mode): df
            for mode, df in bundle.results.items()
        }
        _render_graph_maker(
            gm_series,
            key_prefix="run_gm",
            default_metrics=["employment_rate_low", "employment_rate_high"],
        )


def _render_saved_runs_panel():
    st.subheader("Saved runs")
    history = load_run_history(APP_DIR)
    if history:
        history_df = pd.DataFrame(
            [
                {
                    "run": item["run_label"],
                    "created_at": item["created_at"],
                    "steps": item["n_steps"],
                    "modes": ", ".join(item["modes"]),
                }
                for item in history
            ]
        )
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        reload_options = {
            f"{item['run_label']} ({item['created_at']})": item
            for item in history
        }
        selected_reload = st.selectbox(
            "Load setup from old run",
            options=[""] + list(reload_options.keys()),
            help="This fills the sidebar with the parameters, modes and step count from an old run, and restores that run's plots in the dashboard.",
        )
        if selected_reload and st.button("Load selected run into setup", use_container_width=True):
            st.session_state["pending_run_to_load"] = reload_options[selected_reload]
            st.rerun()

        rename_options = {
            f"{item['run_label']} ({item['created_at']})": item["run_id"]
            for item in history
        }
        selected_rename = st.selectbox(
            "Rename saved run",
            options=[""] + list(rename_options.keys()),
            help="Change the visible name of a saved run without touching its results.",
        )
        new_run_name = st.text_input(
            "New saved run name",
            value="",
            key="rename_run_name_input",
        )
        if selected_rename and new_run_name.strip() and st.button("Rename selected run", use_container_width=True):
            renamed = rename_run_in_history(APP_DIR, rename_options[selected_rename], new_run_name)
            if renamed:
                active_bundle = st.session_state.get("latest_bundle")
                if active_bundle is not None and getattr(active_bundle, "run_id", None) == rename_options[selected_rename]:
                    active_bundle.run_label = new_run_name.strip()
                    st.session_state["latest_bundle"] = active_bundle
                    st.session_state["pending_setup_payload"] = {
                        "run_label": new_run_name.strip(),
                        "n_steps": st.session_state.get("steps_input", 1000),
                        "modes": st.session_state.get("modes_input", MODES),
                        "params": {key: st.session_state.get(f"param_{key}", value) for key, value in BASE_PARAMS.items()},
                    }
                st.success("Saved run renamed.")
                st.rerun()
            else:
                st.warning("Could not rename this run.")

        delete_options = {
            f"{item['run_label']} ({item['created_at']})": item["run_id"]
            for item in history
        }
        runs_to_delete = st.multiselect(
            "Delete old runs",
            options=list(delete_options.keys()),
            help="This removes the saved run folder from dashboard_runs.",
        )
        if runs_to_delete and st.button("Delete selected runs", type="secondary", use_container_width=True):
            deleted = 0
            for label in runs_to_delete:
                if delete_run_from_history(APP_DIR, delete_options[label]):
                    deleted += 1
            st.success(f"Deleted {deleted} run(s).")
            st.rerun()
    else:
        st.info("No saved runs yet.")


def _render_compare_mode_across_runs(history: list[dict], metric: str):
    run_options = {f"{item['run_label']} ({item['created_at']})": item["run_id"] for item in history}
    selected_mode = st.selectbox(
        "Mode",
        options=MODES,
        format_func=lambda mode: MODE_LABELS.get(mode, mode),
        key="compare_same_mode_mode",
    )
    selected_run_labels = st.multiselect(
        "Runs",
        options=list(run_options.keys()),
        default=list(run_options.keys())[: min(4, len(run_options))],
        key="compare_same_mode_runs",
    )
    if not selected_run_labels:
        st.info("Select at least one run.")
        return

    frame = load_combined_history_frame(
        APP_DIR,
        run_ids=[run_options[label] for label in selected_run_labels],
        metric=metric,
        modes=[selected_mode],
    )
    if frame.empty:
        st.warning("No comparison data was found for this selection.")
        return

    display_frame = frame.sort_values("step").copy()
    if metric in CUMULATIVE_CHANNEL_METRICS:
        display_frame["value"] = display_frame.groupby(["run_id", "run_display", "mode"])["value"].cumsum()
        st.caption("The same mode across multiple runs. For automation channels, this is shown as cumulative lines.")
    else:
        st.caption("The same mode across multiple runs. Each run gets its own color.")
    comparison_fig = build_comparison_figure(
        frame,
        metric,
        color_by="run",
        title_suffix=MODE_LABELS.get(selected_mode, selected_mode),
    )
    st.pyplot(comparison_fig, use_container_width=True)
    plt.close(comparison_fig)

    final_step_table = (
        display_frame.sort_values("step")
        .groupby(["run_display"], as_index=False)
        .tail(1)[["run_display", "value"]]
        .rename(columns={"run_display": "run", "value": "final_value"})
        .set_index("run")
    )
    st.caption("Final values at the last step.")
    st.dataframe(final_step_table, use_container_width=True)


def _render_compare_modes_within_run(history: list[dict], metric: str):
    run_options = {f"{item['run_label']} ({item['created_at']})": item["run_id"] for item in history}
    selected_run_label = st.selectbox(
        "Run",
        options=list(run_options.keys()),
        key="compare_single_run",
    )
    selected_modes = st.multiselect(
        "Modes",
        options=MODES,
        default=MODES,
        format_func=lambda mode: MODE_LABELS.get(mode, mode),
        key="compare_single_run_modes",
    )
    if not selected_modes:
        st.info("Select at least one mode.")
        return

    frame = load_combined_history_frame(
        APP_DIR,
        run_ids=[run_options[selected_run_label]],
        metric=metric,
        modes=selected_modes,
    )
    if frame.empty:
        st.warning("No comparison data was found for this selection.")
        return

    display_frame = frame.sort_values("step").copy()
    if metric in CUMULATIVE_CHANNEL_METRICS:
        display_frame["value"] = display_frame.groupby(["run_id", "run_display", "mode"])["value"].cumsum()
        st.caption("Multiple modes within exactly one run. For automation channels, this is shown as cumulative lines.")
    else:
        st.caption("Multiple modes within exactly one run. Colors and line styles correspond to the mode.")
    comparison_fig = build_comparison_figure(
        frame,
        metric,
        color_by="mode",
        title_suffix=selected_run_label,
    )
    st.pyplot(comparison_fig, use_container_width=True)
    plt.close(comparison_fig)

    final_step_table = (
        display_frame.sort_values("step")
        .groupby(["mode"], as_index=False)
        .tail(1)[["mode", "value"]]
        .assign(mode=lambda df: df["mode"].map(MODE_LABELS))
        .rename(columns={"mode": "mode", "value": "final_value"})
        .set_index("mode")
    )
    st.caption("Final values at the last step.")
    st.dataframe(final_step_table, use_container_width=True)


def _render_history():
    history = load_run_history(APP_DIR)
    st.subheader("Compare runs")
    st.caption("Choose the comparison type first, so runs and modes do not get mixed together.")
    if not history:
        st.info("There are no saved dashboard runs to compare yet.")
        return

    metric_options = list(PLOT_METRICS)
    metric = st.selectbox(
        "Metric",
        options=metric_options,
        format_func=lambda item: METRIC_LABELS.get(item, item),
        index=metric_options.index("ai_adoption_rate"),
        key="compare_metric",
    )

    tab1, tab2 = st.tabs(["Same mode across runs", "Modes within one run"])
    with tab1:
        _render_compare_mode_across_runs(history, metric)
    with tab2:
        _render_compare_modes_within_run(history, metric)


def _render_mode_info():
    st.subheader("Adoption modes explained")
    st.caption("This page explains the economic logic behind ULC and the three NPV variants.")

    st.markdown(
        """
        <div class="info-card">
            <strong>ULC (baseline)</strong><br/>
            Firms compare current unit costs only. If AI is cheaper than the cheapest human input for a task right now, they automate.
            This is the most short-run and myopic rule.
        </div>
        <div class="info-card">
            <strong>NPV - naive expectations</strong><br/>
            Firms evaluate automation as an investment. They compare upfront cost against discounted future savings, but assume current wages continue unchanged.
            This is forward-looking, but still simple-minded about the future.
        </div>
        <div class="info-card">
            <strong>NPV - adaptive expectations</strong><br/>
            Firms still use NPV, but now extrapolate recent wage trends.
            If wages have recently been rising, they expect higher future labour costs and automation becomes more attractive.
        </div>
        <div class="info-card">
            <strong>NPV - mean-field expectations</strong><br/>
            Firms use a mean-field style forecast based on labour-market conditions and recent worker-displacement flow.
            This usually makes them more strategic about how future wages may respond to automation itself.
        </div>
        """,
        unsafe_allow_html=True,
    )

    comparison_df = pd.DataFrame(
        [
            {"Mode": "ULC", "Decision rule": "Current unit cost only", "Forward-looking": "No", "Wage expectations": "None", "Typical behaviour": "Fast and reactive"},
            {"Mode": "NPV - naive", "Decision rule": "Discounted investment logic", "Forward-looking": "Yes", "Wage expectations": "Current wages stay constant", "Typical behaviour": "More cautious than ULC"},
            {"Mode": "NPV - adaptive", "Decision rule": "Discounted investment logic", "Forward-looking": "Yes", "Wage expectations": "Recent wage trend continues", "Typical behaviour": "Responds to recent momentum"},
            {"Mode": "NPV - mean-field", "Decision rule": "Discounted investment logic", "Forward-looking": "Yes", "Wage expectations": "Projected from macro/labour conditions", "Typical behaviour": "Most strategic / model-consistent"},
        ]
    )
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="info-card">
            <strong>Rule of thumb</strong><br/>
            ULC asks: "Is AI cheaper right now?"<br/>
            NPV asks: "Is it worth paying the investment cost today for future savings?"<br/>
            The difference between the three NPV modes is mainly how firms imagine future wages will evolve.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Experimenter
# =============================================================================

_NUMERIC_PARAMS = list(NUMERIC_PARAMS)
_BOOL_PARAMS = list(BOOL_PARAMS)


def _render_experimenter(sidebar_params: dict, sidebar_n_steps: int, sidebar_modes: list):
    """
    Multi-configuration experiment runner with incremental caching and save/load.

    Results are stored keyed by config fingerprint so re-clicking Run only
    executes experiments whose config has changed or that are genuinely new.
    """
    st.subheader("Experimenter")
    st.caption(
        "Add named configurations and run them. Already-run experiments are cached - "
        "only new or changed ones are re-executed when you click Run."
    )

    # Initialise session state
    if "exp_configs" not in st.session_state:
        st.session_state["exp_configs"] = []
    # Cache: {fingerprint: {"name": str, "bundle": RunBundle}}
    if "exp_cache" not in st.session_state:
        st.session_state["exp_cache"] = {}

    # =========================================================
    # Load saved batch
    # =========================================================
    # Ensure the saved-batch directory exists so a previously saved experiment
    # can be dropped straight into it and loaded — without first having to add
    # and save a run in this session.
    (APP_DIR / "dashboard_experimenter").mkdir(parents=True, exist_ok=True)
    batches = list_experiment_batches(APP_DIR)
    # Always show the loader (even with no batches yet) rather than hiding it
    # until the first batch has been saved.
    if True:
        with st.expander("Load saved batch", expanded=False):
            if not batches:
                st.caption(
                    "No saved batches yet. Drop a saved experiment-batch folder into "
                    "`outputs/dashboard_experimenter/` and reload this page, or save one "
                    "from a completed run below."
                )
            batch_options = {
                f"{b['batch_name']}{' [setup only]' if b.get('config_only') else ''} ({b['created_at']})": b for b in batches
            }
            selected_batch = st.selectbox(
                "Saved batch",
                options=[""] + list(batch_options.keys()),
                key="exp_load_select",
            )
            col_load, col_load_rerun, col_del_batch = st.columns(3)
            with col_load:
                if selected_batch and st.button("Load batch", use_container_width=True, help="Restore saved results — no re-simulation needed."):
                    b = batch_options[selected_batch]
                    if b.get("config_only"):
                        st.warning("This saved item contains configs only. Use 'Load & rerun' to simulate it.")
                        return
                    loaded_configs, loaded_results = load_experiment_batch(b["path"])
                    st.session_state["exp_configs"] = loaded_configs
                    # Restore cache from loaded results
                    new_cache = {}
                    for cfg, item in zip(loaded_configs, loaded_results):
                        fp = _config_fingerprint(cfg)
                        new_cache[fp] = item
                    st.session_state["exp_cache"] = new_cache
                    st.success(f"Loaded '{b['batch_name']}' — {len(loaded_configs)} experiments restored.")
                    st.rerun()
            with col_load_rerun:
                if selected_batch and st.button("Load & rerun", use_container_width=True, help="Load configs but clear cached results so everything is re-simulated."):
                    b = batch_options[selected_batch]
                    loaded_configs, _loaded_results = load_experiment_batch(b["path"])
                    st.session_state["exp_configs"] = loaded_configs
                    st.session_state["exp_cache"] = {}  # empty cache → all configs become pending
                    st.success(f"Loaded '{b['batch_name']}' — {len(loaded_configs)} configs queued for rerun.")
                    st.rerun()
            with col_del_batch:
                if selected_batch and st.button("Delete batch", type="secondary", use_container_width=True):
                    if delete_experiment_batch(batch_options[selected_batch]["path"]):
                        st.success("Batch deleted.")
                        st.rerun()

    # =========================================================
    # Add new experiment
    # =========================================================
    with st.expander("Add experiment", expanded=len(st.session_state["exp_configs"]) == 0):
        new_name = st.text_input(
            "Experiment name",
            value=f"Experiment {len(st.session_state['exp_configs']) + 1}",
            key="exp_new_name",
        )
        st.caption("Leave parameter overrides empty to use sidebar settings unchanged.")

        override_keys = st.multiselect(
            "Numeric parameters to override",
            options=_NUMERIC_PARAMS,
            format_func=lambda k: f"{k}  (current: {sidebar_params.get(k, BASE_PARAMS.get(k))})",
            key="exp_override_keys",
        )

        overrides: dict = {}
        if override_keys:
            ov_cols = st.columns(min(3, len(override_keys)))
            for i, key in enumerate(override_keys):
                with ov_cols[i % len(ov_cols)]:
                    default_val = sidebar_params.get(key, BASE_PARAMS.get(key, 0))
                    meta = PARAMETER_META.get(key, {})
                    is_int = _uses_integer_number_input(BASE_PARAMS.get(key), meta)
                    widget_kwargs = dict(
                        label=key,
                        value=int(default_val) if is_int else float(default_val),
                        step=int(meta.get("step", 1)) if is_int else float(meta.get("step", 0.01)),
                        key=f"exp_ov_{key}",
                    )
                    if "min" in meta:
                        widget_kwargs["min_value"] = int(meta["min"]) if is_int else float(meta["min"])
                    if "max" in meta:
                        widget_kwargs["max_value"] = int(meta["max"]) if is_int else float(meta["max"])
                    if not is_int:
                        widget_kwargs["format"] = "%.4f"
                    overrides[key] = int(st.number_input(**widget_kwargs)) if is_int else float(st.number_input(**widget_kwargs))

        if _BOOL_PARAMS:
            st.caption("Boolean overrides")
            bool_cols = st.columns(min(4, len(_BOOL_PARAMS)))
            for i, key in enumerate(_BOOL_PARAMS):
                with bool_cols[i % len(bool_cols)]:
                    current_bool = bool(sidebar_params.get(key, BASE_PARAMS.get(key, False)))
                    overrides[key] = st.checkbox(
                        key,
                        value=current_bool,
                        key=f"exp_ov_bool_{key}",
                        help=PARAMETER_META.get(key, {}).get("formula", ""),
                    )

        st.caption("Alpha distribution override")
        _exp_alpha_src_opts = ["uniform", "data"]
        _exp_alpha_src_labels = {"uniform": "Uniform U(min, max)", "data": "Gmyrek data"}
        _exp_cur_src = sidebar_params.get("alpha_source", "uniform")
        overrides["alpha_source"] = st.radio(
            "Alpha source",
            options=_exp_alpha_src_opts,
            format_func=lambda v: _exp_alpha_src_labels[v],
            index=_exp_alpha_src_opts.index(_exp_cur_src) if _exp_cur_src in _exp_alpha_src_opts else 0,
            horizontal=True,
            key="exp_ov_alpha_source",
        )
        _exp_cur_min = float(sidebar_params.get("alpha_min", 0.0))
        _exp_cur_max = float(sidebar_params.get("alpha_max", 1.0))
        _exp_alpha_range = st.slider(
            "Alpha range [min, max]",
            min_value=0.0, max_value=1.0,
            value=(_exp_cur_min, _exp_cur_max),
            step=0.01, format="%.2f",
            key="exp_ov_alpha_range",
        )
        overrides["alpha_min"] = _exp_alpha_range[0]
        overrides["alpha_max"] = _exp_alpha_range[1]

        exp_modes = st.multiselect(
            "Modes for this experiment",
            options=MODES,
            default=sidebar_modes or MODES,
            format_func=lambda m: MODE_LABELS.get(m, m),
            key="exp_new_modes",
        )
        exp_steps = st.number_input(
            "Steps", min_value=1, max_value=5000, value=sidebar_n_steps, step=10, key="exp_new_steps"
        )

        if st.button("Add to list", type="primary"):
            merged_params = {**sidebar_params, **overrides}
            st.session_state["exp_configs"].append({
                "name": new_name.strip() or f"Experiment {len(st.session_state['exp_configs']) + 1}",
                "params": merged_params,
                "modes": exp_modes or MODES,
                "n_steps": int(exp_steps),
                "overrides": overrides,
            })
            st.rerun()

    # =========================================================
    # Queue table
    # =========================================================
    configs = st.session_state["exp_configs"]
    if not configs:
        st.info("No experiments queued yet.")
        return

    if "exp_editing_idx" not in st.session_state:
        st.session_state["exp_editing_idx"] = None

    cache = st.session_state["exp_cache"]
    st.markdown(f"**{len(configs)} experiment(s) queued**")
    to_remove = []
    for idx, cfg in enumerate(configs):
        fp = _config_fingerprint(cfg)
        cached = fp in cache
        status_icon = "✅" if cached else "⏳"
        col_status, col_name, col_overrides, col_modes, col_steps, col_edit, col_del = st.columns([0.5, 2, 4, 3, 0.8, 0.5, 0.5])
        col_status.markdown(status_icon, help="✅ cached  ⏳ needs run")
        col_name.markdown(f"**{cfg['name']}**")
        override_summary = ", ".join(f"{k}={v}" for k, v in cfg.get("overrides", {}).items()) or "sidebar defaults"
        col_overrides.caption(override_summary)
        col_modes.caption(", ".join(MODE_LABELS.get(m, m) for m in cfg["modes"]))
        col_steps.caption(f"{cfg['n_steps']}s")
        if col_edit.button("✏", key=f"exp_edit_btn_{idx}", help="Edit this experiment"):
            st.session_state["exp_editing_idx"] = idx if st.session_state["exp_editing_idx"] != idx else None
            st.rerun()
        if col_del.button("✕", key=f"exp_del_{idx}"):
            to_remove.append(idx)

        # Inline edit form
        if st.session_state.get("exp_editing_idx") == idx:
            with st.container(border=True):
                st.caption(f"Editing **{cfg['name']}**")
                edit_name = st.text_input("Name", value=cfg["name"], key=f"edit_name_{idx}")

                current_overrides = cfg.get("overrides", {})
                edit_override_keys = st.multiselect(
                    "Numeric parameters to override",
                    options=_NUMERIC_PARAMS,
                    default=[k for k in current_overrides if k in _NUMERIC_PARAMS],
                    format_func=lambda k: f"{k}  (base: {BASE_PARAMS.get(k)})",
                    key=f"edit_ov_keys_{idx}",
                )
                edit_overrides: dict = {}
                if edit_override_keys:
                    ov_cols = st.columns(min(3, len(edit_override_keys)))
                    for i, key in enumerate(edit_override_keys):
                        with ov_cols[i % len(ov_cols)]:
                            default_val = current_overrides.get(key, cfg["params"].get(key, BASE_PARAMS.get(key, 0)))
                            meta = PARAMETER_META.get(key, {})
                            is_int = _uses_integer_number_input(BASE_PARAMS.get(key), meta)
                            widget_kwargs = dict(
                                label=key,
                                value=int(default_val) if is_int else float(default_val),
                                step=int(meta.get("step", 1)) if is_int else float(meta.get("step", 0.01)),
                                key=f"edit_ov_{idx}_{key}",
                            )
                            if "min" in meta:
                                widget_kwargs["min_value"] = int(meta["min"]) if is_int else float(meta["min"])
                            if "max" in meta:
                                widget_kwargs["max_value"] = int(meta["max"]) if is_int else float(meta["max"])
                            if not is_int:
                                widget_kwargs["format"] = "%.4f"
                            edit_overrides[key] = int(st.number_input(**widget_kwargs)) if is_int else float(st.number_input(**widget_kwargs))

                if _BOOL_PARAMS:
                    bool_cols = st.columns(min(4, len(_BOOL_PARAMS)))
                    for i, key in enumerate(_BOOL_PARAMS):
                        with bool_cols[i % len(bool_cols)]:
                            current_bool = bool(current_overrides.get(key, cfg["params"].get(key, BASE_PARAMS.get(key, False))))
                            edit_overrides[key] = st.checkbox(
                                key,
                                value=current_bool,
                                key=f"edit_ov_bool_{idx}_{key}",
                            )

                st.caption("Alpha distribution override")
                _edit_alpha_src_opts = ["uniform", "data"]
                _edit_alpha_src_labels = {"uniform": "Uniform U(min, max)", "data": "Gmyrek data"}
                _edit_cur_src = current_overrides.get("alpha_source", cfg["params"].get("alpha_source", "uniform"))
                edit_overrides["alpha_source"] = st.radio(
                    "Alpha source",
                    options=_edit_alpha_src_opts,
                    format_func=lambda v: _edit_alpha_src_labels[v],
                    index=_edit_alpha_src_opts.index(_edit_cur_src) if _edit_cur_src in _edit_alpha_src_opts else 0,
                    horizontal=True,
                    key=f"edit_ov_alpha_source_{idx}",
                )
                _edit_cur_min = float(current_overrides.get("alpha_min", cfg["params"].get("alpha_min", 0.0)))
                _edit_cur_max = float(current_overrides.get("alpha_max", cfg["params"].get("alpha_max", 1.0)))
                _edit_alpha_range = st.slider(
                    "Alpha range [min, max]",
                    min_value=0.0, max_value=1.0,
                    value=(_edit_cur_min, _edit_cur_max),
                    step=0.01, format="%.2f",
                    key=f"edit_ov_alpha_range_{idx}",
                )
                edit_overrides["alpha_min"] = _edit_alpha_range[0]
                edit_overrides["alpha_max"] = _edit_alpha_range[1]

                edit_modes = st.multiselect(
                    "Modes",
                    options=MODES,
                    default=cfg["modes"],
                    format_func=lambda m: MODE_LABELS.get(m, m),
                    key=f"edit_modes_{idx}",
                )
                edit_steps = st.number_input(
                    "Steps", min_value=1, max_value=5000, value=cfg["n_steps"], step=10,
                    key=f"edit_steps_{idx}",
                )

                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("Save changes", type="primary", key=f"edit_save_{idx}"):
                        old_fp = _config_fingerprint(cfg)
                        merged = {**sidebar_params, **edit_overrides}
                        st.session_state["exp_configs"][idx] = {
                            "name": edit_name.strip() or cfg["name"],
                            "params": merged,
                            "modes": edit_modes or MODES,
                            "n_steps": int(edit_steps),
                            "overrides": edit_overrides,
                        }
                        # Invalidate old cache entry so it reruns with new config
                        st.session_state["exp_cache"].pop(old_fp, None)
                        st.session_state["exp_editing_idx"] = None
                        st.rerun()
                with col_cancel:
                    if st.button("Cancel", key=f"edit_cancel_{idx}"):
                        st.session_state["exp_editing_idx"] = None
                        st.rerun()

    if to_remove:
        removed_fps = {_config_fingerprint(configs[i]) for i in to_remove}
        st.session_state["exp_configs"] = [c for i, c in enumerate(configs) if i not in to_remove]
        for fp in removed_fps:
            st.session_state["exp_cache"].pop(fp, None)
        st.rerun()

    pending = [cfg for cfg in configs if _config_fingerprint(cfg) not in cache]
    n_pending = len(pending)
    n_cached = len(configs) - n_pending

    exp_log_decisions = st.checkbox(
        "Log investment decisions for this batch",
        value=False,
        key="exp_log_decisions",
        help="Records every automation evaluation (both yes and no). Enables the Excel download below. Slows the run.",
    )

    col_run, col_rerun, col_clear = st.columns([3, 2, 1])
    with col_run:
        run_label_text = (
            f"Run {n_pending} new experiment(s)" if n_pending else "Nothing new to run (all cached)"
        )
        run_clicked = st.button(
            run_label_text,
            type="primary",
            use_container_width=True,
            disabled=(n_pending == 0),
        )
    with col_rerun:
        if configs and st.button("Force rerun all", use_container_width=True, help="Clear cached results and re-simulate all loaded experiments."):
            st.session_state["exp_cache"] = {}
            st.rerun()
    with col_clear:
        if st.button("Clear all", type="secondary", use_container_width=True):
            st.session_state["exp_configs"] = []
            st.session_state["exp_cache"] = {}
            st.rerun()

    if n_cached:
        st.caption(f"{n_cached} experiment(s) already cached - skipping re-run. Use 'Force rerun all' to re-simulate.")

    st.divider()
    show_extended_analysis = st.checkbox(
        "Extended analysis - multiple runs per experiment",
        value=True,
        key="exp_show_multirun_analysis",
        help=(
            "Run every queued configuration across multiple seeds and compare "
            "configurations with mean +/- sd trajectories and a paired permutation test."
        ),
    )
    if show_extended_analysis:
        # Extended multi-run analysis: mean +/- 1 sd ribbons, full-results
        # export, and the paired permutation test between configurations.
        render_multirun_analysis(configs, sidebar_params, APP_DIR)

    if run_clicked and pending:
        exp_bar = st.progress(0.0, text="Starting...")

        def exp_progress(label, done, total):
            exp_bar.progress(done / max(total, 1), text=f"Running: {label} ({done}/{total})")

        new_results = run_experiment_batch(
            configs=pending, progress_callback=exp_progress, log_decisions=exp_log_decisions
        )
        for cfg, item in zip(pending, new_results):
            fp = _config_fingerprint(cfg)
            st.session_state["exp_cache"][fp] = item
        exp_bar.progress(1.0, text="Done.")

    # Assemble ordered results from cache (preserving queue order)
    exp_results = []
    for cfg in configs:
        fp = _config_fingerprint(cfg)
        if fp in st.session_state["exp_cache"]:
            exp_results.append(st.session_state["exp_cache"][fp])

    if not exp_results:
        return

    # =========================================================
    # Save batch
    # =========================================================
    st.divider()
    with st.expander("Save this batch", expanded=False):
        save_name = st.text_input("Batch name", value="my_experiment_batch", key="exp_save_name")
        if st.button("Save batch", type="primary"):
            save_experiment_batch(
                exp_results=exp_results,
                configs=configs,
                base_dir=APP_DIR,
                batch_name=save_name.strip() or "batch",
            )
            st.success(f"Batch '{save_name}' saved.")

    # =========================================================
    # Results
    # =========================================================
    st.subheader("Results")
    available_modes = list(dict.fromkeys(m for item in exp_results for m in item["bundle"].modes))
    compare_mode = st.selectbox(
        "Compare experiments in mode",
        options=available_modes,
        format_func=lambda m: MODE_LABELS.get(m, m),
        key="exp_compare_mode",
    )

    st.markdown("**Final-step summary**")
    summary_df = build_experiment_final_table(exp_results, compare_mode)
    st.dataframe(summary_df, use_container_width=True)

    st.markdown("**Time-series comparison**")

    # ── Rolling average toggle ────────────────────────────────────────────────
    _exp_rolling_col1, _exp_rolling_col2 = st.columns([1, 3])
    with _exp_rolling_col1:
        _exp_rolling_on = st.toggle(
            "Rolling average",
            value=False,
            key="exp_rolling_on",
            help="Smooth each series with a centred moving average. Useful to cut through step-to-step noise and compare trends across experiments.",
        )
    with _exp_rolling_col2:
        if _exp_rolling_on:
            _exp_rolling_win = st.slider(
                "Window (steps)", min_value=2, max_value=50, value=10, step=1,
                key="exp_rolling_win",
            )
        else:
            _exp_rolling_win = 1

    metric_opts = PLOT_METRICS
    # Handle "Select all" via a flag set before the widget renders to avoid
    # StreamlitAPIException (cannot modify widget key after instantiation)
    if st.session_state.pop("_exp_select_all_pending", False):
        st.session_state["exp_compare_metrics"] = list(PLOT_METRICS)
    _ecol1, _ecol2 = st.columns([6, 1])
    with _ecol1:
        selected_metrics = st.multiselect(
            "Metrics to plot",
            options=metric_opts,
            default=DEFAULT_SELECTED_METRICS,
            format_func=lambda m: METRIC_LABELS.get(m, m),
            key="exp_compare_metrics",
        )
    with _ecol2:
        st.write("")  # vertical alignment
        if st.button("Select all", key="exp_select_all_metrics"):
            st.session_state["_exp_select_all_pending"] = True
            st.rerun()
    for metric in selected_metrics:
        fig = build_experiment_comparison_figure(exp_results, metric, compare_mode, rolling=_exp_rolling_win)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Investment log download for experiments ───────────────────────────────
    any_log = any(
        model.investment_log
        for item in exp_results
        for model in item["bundle"].models.values()
    )
    if any_log:
        with st.expander("Investment decision logs — all experiments", expanded=False):
            st.caption("Download a single Excel workbook with one sheet per experiment plus combined sheets.")
            exp_log_items = []
            all_dfs = []
            for item in exp_results:
                df = build_investment_log_dataframe(item["bundle"], experiment_name=item["name"])
                if not df.empty:
                    exp_log_items.append({"name": item["name"], "df": df})
                    all_dfs.append(df)
            if all_dfs:
                combined_log = pd.concat(all_dfs, ignore_index=True)
                excel_bytes = build_investment_log_excel_bytes(combined_log, experiment_logs=exp_log_items)
                st.download_button(
                    label="Download investment log — all experiments (Excel)",
                    data=excel_bytes,
                    file_name="investment_log_experiments.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

    st.divider()
    st.markdown("**Automation complexity — per experiment**")
    st.caption(
        "Select an experiment to inspect its automation timing and end-state heatmap. "
        "The timing scatter (top, NPV modes only) shows when each task was automated versus how complex it was. "
        "The heatmap (bottom) shows the fraction of tasks automated per complexity level at the end of the run."
    )
    exp_names = [item["name"] for item in exp_results]
    selected_exp_name = st.selectbox(
        "Experiment to inspect",
        options=exp_names,
        key="exp_complexity_select",
    )
    selected_exp_item = next((item for item in exp_results if item["name"] == selected_exp_name), None)
    if selected_exp_item is not None:
        complexity_fig = build_automation_complexity_figure(selected_exp_item["bundle"])
        st.pyplot(complexity_fig, use_container_width=True)
        plt.close(complexity_fig)

    st.divider()
    st.markdown("**Alpha × workforce — per experiment**")
    st.caption(
        "Alpha is binned in fixed steps of 0.05 from 0 to 1 (20 bins total). "
        "**Top figure:** each task is classified as AI / High-skill / Low-skill / Mixed / Empty — "
        "shows which task statuses dominate in high- vs. low-alpha firms. "
        "**Bottom figure:** absolute worker-slots (AI units + high-skill + low-skill) in each alpha bin."
    )
    exp_aw_name = st.selectbox(
        "Experiment (alpha × workforce)",
        options=[item["name"] for item in exp_results],
        key="exp_alpha_wf_select",
    )
    exp_aw_item = next((item for item in exp_results if item["name"] == exp_aw_name), None)
    if exp_aw_item is not None:
        st.markdown("##### Task status by alpha range")
        ast_fig = build_alpha_status_figure(exp_aw_item["bundle"])
        st.pyplot(ast_fig, use_container_width=True)
        plt.close(ast_fig)
        st.markdown("##### Absolute worker count by alpha range")
        aw_fig = build_alpha_workforce_figure(exp_aw_item["bundle"])
        st.pyplot(aw_fig, use_container_width=True)
        plt.close(aw_fig)

    st.divider()
    st.markdown("**Task status by complexity — per experiment**")
    st.caption(
        "Each task at the last simulation step is classified as AI, High-skill only, "
        "Low-skill only, Mixed (both), or Empty (no workers). "
        "The bars show how many tasks fall into each status per complexity level."
    )
    exp_ts_name = st.selectbox(
        "Experiment (task status by complexity)",
        options=[item["name"] for item in exp_results],
        key="exp_task_status_select",
    )
    exp_ts_item = next((item for item in exp_results if item["name"] == exp_ts_name), None)
    if exp_ts_item is not None:
        ts_fig = build_task_complexity_status_figure(exp_ts_item["bundle"])
        st.pyplot(ts_fig, use_container_width=True)
        plt.close(ts_fig)

    st.divider()
    st.markdown("**Absolute worker count by task complexity — per experiment**")
    st.caption(
        "Shows the absolute number of AI units, high-skill, and low-skill worker-slots "
        "across all task complexity levels at the **last** simulation step. "
        "Bar height reflects both task frequency and staffing intensity."
    )
    exp_tw_name = st.selectbox(
        "Experiment (worker count by complexity)",
        options=[item["name"] for item in exp_results],
        key="exp_task_wf_select",
    )
    exp_tw_item = next((item for item in exp_results if item["name"] == exp_tw_name), None)
    if exp_tw_item is not None:
        tw_fig = build_task_complexity_workforce_figure(exp_tw_item["bundle"])
        st.pyplot(tw_fig, use_container_width=True)
        plt.close(tw_fig)

    st.divider()
    st.markdown("**Firm inspector**")
    st.caption("Inspect individual firms at the end of the run. Select an experiment, mode and firm to see task status and assigned workers.")

    insp_exp_name = st.selectbox(
        "Experiment",
        options=[item["name"] for item in exp_results],
        key="exp_inspector_exp",
    )
    insp_exp_item = next((item for item in exp_results if item["name"] == insp_exp_name), None)
    if insp_exp_item is not None:
        insp_bundle = insp_exp_item["bundle"]
        if not insp_bundle.models:
            st.info("Firm inspector is not available for loaded batches — model objects are not persisted. Re-run the experiment to use this feature.")
        else:
            insp_mode = st.selectbox(
                "Adoption mode",
                options=insp_bundle.modes,
                format_func=lambda m: MODE_LABELS.get(m, m),
                key="exp_inspector_mode",
            )
            insp_model = insp_bundle.models[insp_mode]
            producer_options = [
                (p.unique_id, f"Firm {p.unique_id}  (α={p.alpha:.2f}, output={p.output:.1f}, n_workers={sum(len(t.employees) for t in p.tasks)}, n_ai={sum(t.n_ai for t in p.tasks)})")
                for p in insp_model.producers
            ]
            selected_pid = st.selectbox(
                "Select firm",
                options=[pid for pid, _ in producer_options],
                format_func=lambda pid: next(label for p_id, label in producer_options if p_id == pid),
                key="exp_inspector_firm",
            )
            producer = next(p for p in insp_model.producers if p.unique_id == selected_pid)

            n_tasks_total = len(producer.tasks)
            n_auto = sum(1 for t in producer.tasks if t.automated)
            n_workers_total = sum(len(t.employees) for t in producer.tasks)
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Tasks", n_tasks_total)
            col_b.metric("Automated", f"{n_auto} ({n_auto/n_tasks_total:.0%})")
            col_c.metric("Workers assigned", n_workers_total)
            col_d.metric("Output", f"{producer.output:.2f}")

            task_rows = []
            for t in sorted(producer.tasks, key=lambda t: (t.task_type, t.complexity_index)):
                n_high = sum(1 for w in t.employees if w.skill_level == "high")
                n_low  = sum(1 for w in t.employees if w.skill_level == "low")
                if t.automated:
                    status = "AI"
                    input_detail = f"{t.n_ai} AI unit(s)"
                    total_prod = t.n_ai * producer.productivity_ai(t)
                elif t.employees:
                    def _fmt_worker2(w, _model=insp_model):
                        parts = [w.skill_level]
                        if _model.employment_protection:
                            parts.append(w.contract_type)
                            tenure_yr = w.tenure / max(1, _model.steps_per_year)
                            parts.append(f"tenure={tenure_yr:.1f}y")
                        parts.append(f"wage={w.wage:.2f}")
                        return f"W{w.unique_id} ({', '.join(parts)})"
                    input_detail = ", ".join(_fmt_worker2(w) for w in t.employees)
                    status = "Human"
                    total_prod = sum(producer.productivity_human(t, w.skill_level) for w in t.employees)
                else:
                    status = "Empty"
                    input_detail = "—"
                    total_prod = 0.0
                task_rows.append({
                    "Task ID": t.task_id,
                    "Type": t.task_type,
                    "Complexity": t.complexity_index,
                    "Status": status,
                    "n_workers_high": n_high,
                    "n_workers_low": n_low,
                    "n_ai": t.n_ai if t.automated else 0,
                    "Total productivity": round(total_prod, 3),
                    "Input": input_detail,
                })

            task_df = pd.DataFrame(task_rows)

            def _color_status(val):
                if val == "AI":
                    return "background-color: #d1fae5; color: #065f46"
                if val == "Empty":
                    return "background-color: #fee2e2; color: #991b1b"
                return ""

            st.dataframe(
                task_df.style.applymap(_color_status, subset=["Status"]),
                use_container_width=True,
                hide_index=True,
            )

    # ── Custom graph builder (across all experiments, pick mode) ──────────────
    st.divider()
    with st.expander("Custom graph builder", expanded=False):
        st.caption(
            "Pick a metric, a mode and which experiments to include. "
            "Each experiment becomes one line in the graph."
        )
        if exp_results:
            exp_mode_opts = list({m for item in exp_results for m in item["bundle"].modes})
            gm_exp_mode = st.selectbox(
                "Adoption mode",
                options=exp_mode_opts,
                format_func=lambda m: MODE_LABELS.get(m, m),
                key="exp_gm_mode",
            )
            gm_exp_series = {
                item["name"]: item["bundle"].results.get(gm_exp_mode, pd.DataFrame())
                for item in exp_results
                if gm_exp_mode in item["bundle"].results
            }
            _render_graph_maker(
                gm_exp_series,
                key_prefix="exp_gm",
                default_metrics=["ai_adoption_rate", "employment_rate_low"],
            )
        else:
            st.info("No experiment results available yet.")


# =============================================================================
# Sensitivity analysis
# =============================================================================

def _render_sensitivity(sidebar_params: dict):
    """
    Sensitivity analysis: single-parameter sweep or multi-parameter (LHS / Monte Carlo / Factorial).
    Supports save/load of both setups and results.
    """
    st.subheader("Sensitivity analysis")

    # =========================================================
    # Load / Delete saved sweeps and setups
    # =========================================================
    saved_sweeps = list_sensitivity_sweeps(APP_DIR)
    saved_setups = list_sensitivity_setups(APP_DIR)

    load_col, setup_col = st.columns(2)
    with load_col:
        if saved_sweeps:
            with st.expander("Load saved sweep (results)", expanded=False):
                sweep_options = {
                    f"{s['sweep_name']}  [{s.get('sweep_type','single')}]  ({s['created_at']})": s
                    for s in saved_sweeps
                }
                selected_sweep = st.selectbox(
                    "Saved sweep", options=[""] + list(sweep_options.keys()), key="sens_load_select"
                )
                col_load_s, col_del_s = st.columns(2)
                with col_load_s:
                    if selected_sweep and st.button("Load sweep", use_container_width=True):
                        s = sweep_options[selected_sweep]
                        st.session_state["sensitivity_results"] = load_sensitivity_sweep(s["path"])
                        st.success(f"Loaded '{s['sweep_name']}'.")
                        st.rerun()
                with col_del_s:
                    if selected_sweep and st.button("Delete sweep", type="secondary", use_container_width=True):
                        if delete_sensitivity_sweep(sweep_options[selected_sweep]["path"]):
                            st.success("Deleted.")
                            st.rerun()

    with setup_col:
        if saved_setups:
            with st.expander("Load saved setup (config only)", expanded=False):
                setup_options = {
                    f"{s.get('name','setup')}  [{s.get('sweep_type','single')}]  ({s.get('created_at','')})": s
                    for s in saved_setups
                }
                selected_setup = st.selectbox(
                    "Saved setup", options=[""] + list(setup_options.keys()), key="sens_setup_load_select"
                )
                if selected_setup and st.button("Load setup into form", use_container_width=True, key="sens_load_setup_btn"):
                    st.session_state["sens_loaded_setup"] = setup_options[selected_setup]
                    st.rerun()

    loaded_setup = st.session_state.get("sens_loaded_setup", {})

    # =========================================================
    # Sweep type selection
    # =========================================================
    sweep_type = st.radio(
        "Sweep type",
        ["Single parameter", "Multi-parameter"],
        horizontal=True,
        key="sens_sweep_type",
        index=0 if loaded_setup.get("sweep_type", "single") == "single" else 1,
    )

    # Shared settings
    sh_col1, sh_col2 = st.columns(2)
    with sh_col1:
        sens_mode = st.selectbox(
            "Adoption mode",
            options=MODES,
            format_func=lambda m: MODE_LABELS.get(m, m),
            key="sens_mode",
            index=MODES.index(loaded_setup["mode"]) if loaded_setup.get("mode") in MODES else 0,
        )
    with sh_col2:
        sens_steps = st.number_input(
            "Simulation steps", min_value=1, max_value=5000,
            value=int(loaded_setup.get("n_steps", 150)), step=10, key="sens_n_steps"
        )

    # =========================================================
    # Single-parameter setup
    # =========================================================
    param_values: list = []
    param_configs: list = []
    method = "single"

    if sweep_type == "Single parameter":
        method = "single"
        col1, col2 = st.columns(2)
        with col1:
            default_param = loaded_setup.get("param_name", _NUMERIC_PARAMS[0])
            p_idx = _NUMERIC_PARAMS.index(default_param) if default_param in _NUMERIC_PARAMS else 0
            sens_param = st.selectbox(
                "Parameter to vary",
                options=_NUMERIC_PARAMS,
                format_func=lambda k: f"{k}  (current: {sidebar_params.get(k, BASE_PARAMS.get(k))})",
                key="sens_param",
                index=p_idx,
            )
        current_val = float(sidebar_params.get(sens_param, BASE_PARAMS.get(sens_param, 1.0)))
        meta = PARAMETER_META.get(sens_param, {})
        with col2:
            range_mode = st.radio(
                "Range definition", ["Min / Max / N points", "Explicit list"],
                horizontal=True, key="sens_range_mode",
            )
        if range_mode == "Min / Max / N points":
            loaded_range = loaded_setup.get("param_range", {})
            r1, r2, r3 = st.columns(3)
            with r1:
                sens_min = st.number_input("Min",
                    value=float(loaded_range.get("min", meta.get("min", max(0.0, current_val * 0.25)))),
                    step=float(meta.get("step", 0.01)), format="%.4f", key="sens_min")
            with r2:
                sens_max = st.number_input("Max",
                    value=float(loaded_range.get("max", current_val * 2.0 if current_val > 0 else 2.0)),
                    step=float(meta.get("step", 0.01)), format="%.4f", key="sens_max")
            with r3:
                sens_n = st.number_input("N points", min_value=2, max_value=50,
                    value=int(loaded_range.get("n", 8)), step=1, key="sens_n")
            n_pts = int(sens_n)
            param_values = [sens_min + (sens_max - sens_min) * i / max(n_pts - 1, 1) for i in range(n_pts)]
            if _uses_integer_number_input(BASE_PARAMS.get(sens_param), meta):
                param_values = sorted(set(int(round(v)) for v in param_values))
        else:
            raw_list = st.text_input(
                "Values (comma-separated)",
                value=loaded_setup.get("explicit_values") or ", ".join(
                    str(round(v, 4)) for v in [current_val * 0.5, current_val, current_val * 1.5, current_val * 2.0]
                ),
                key="sens_explicit_list",
            )
            try:
                if _uses_integer_number_input(BASE_PARAMS.get(sens_param), meta):
                    param_values = [int(v.strip()) for v in raw_list.split(",") if v.strip()]
                else:
                    param_values = [float(v.strip()) for v in raw_list.split(",") if v.strip()]
            except ValueError:
                st.error("Could not parse the value list.")
        if param_values:
            st.caption(f"Will run {len(param_values)} simulations: {', '.join(str(v) for v in param_values)}")

    # =========================================================
    # Multi-parameter setup
    # =========================================================
    else:
        method = st.selectbox(
            "Sampling method",
            options=["lhs", "monte_carlo", "factorial"],
            format_func={"lhs": "LHS (Latin Hypercube)", "monte_carlo": "Monte Carlo", "factorial": "Full factorial"}.__getitem__,
            key="sens_multi_method",
            index=["lhs", "monte_carlo", "factorial"].index(loaded_setup.get("method", "lhs")),
        )
        n_samples = st.number_input(
            "N samples (LHS/MC) or grid points per param (factorial)",
            min_value=2, max_value=1000,
            value=int(loaded_setup.get("n_samples", 10)), step=1, key="sens_multi_n"
        )
        sens_seed = st.number_input("Random seed", min_value=0, value=int(loaded_setup.get("seed", 42)), step=1, key="sens_seed")

        st.markdown("**Parameters to vary** — add one row per parameter")
        if "sens_param_configs" not in st.session_state:
            st.session_state["sens_param_configs"] = loaded_setup.get("param_configs", [])

        # Add / remove params
        add_col, _ = st.columns([1, 3])
        with add_col:
            new_param_key = st.selectbox(
                "Add parameter",
                options=[""] + [p for p in _NUMERIC_PARAMS if p not in [pc["name"] for pc in st.session_state["sens_param_configs"]]],
                format_func=lambda k: k if k else "— select —",
                key="sens_add_param_key",
            )
            if new_param_key and st.button("Add", key="sens_add_param_btn"):
                cur = float(sidebar_params.get(new_param_key, BASE_PARAMS.get(new_param_key, 1.0)))
                meta = PARAMETER_META.get(new_param_key, {})
                st.session_state["sens_param_configs"].append({
                    "name": new_param_key,
                    "min": float(meta.get("min", max(0.0, cur * 0.25))),
                    "max": float(cur * 2.0) if cur > 0 else 2.0,
                    "is_int": _uses_integer_number_input(BASE_PARAMS.get(new_param_key), meta),
                })
                st.rerun()

        to_remove_pc = []
        for i, pc in enumerate(st.session_state["sens_param_configs"]):
            pc_col1, pc_col2, pc_col3, pc_col4 = st.columns([2, 1.5, 1.5, 0.4])
            pc_col1.markdown(f"**{pc['name']}**")
            pc["min"] = float(pc_col2.number_input("Min", value=float(pc["min"]), format="%.4f", key=f"pc_min_{i}"))
            pc["max"] = float(pc_col3.number_input("Max", value=float(pc["max"]), format="%.4f", key=f"pc_max_{i}"))
            if pc_col4.button("✕", key=f"pc_del_{i}"):
                to_remove_pc.append(i)
        if to_remove_pc:
            st.session_state["sens_param_configs"] = [pc for j, pc in enumerate(st.session_state["sens_param_configs"]) if j not in to_remove_pc]
            st.rerun()

        param_configs = st.session_state["sens_param_configs"]
        if method == "factorial":
            n_runs = int(n_samples) ** max(len(param_configs), 1)
        else:
            n_runs = int(n_samples)
        st.caption(f"Will run **{n_runs}** simulations with {len(param_configs)} parameter(s).")
        sens_param = ""  # unused for multi

    # =========================================================
    # Alpha distribution override (shared; affects all runs in the sweep)
    # =========================================================
    with st.expander("Alpha distribution for this sweep", expanded=False):
        st.caption(
            "Sets the alpha source and range for **all** runs in this sweep. "
            "To sweep over `alpha_min` or `alpha_max` themselves, add them as parameters above."
        )
        _sens_alpha_src_opts = ["uniform", "data"]
        _sens_alpha_src_labels = {"uniform": "Uniform U(min, max)", "data": "Gmyrek data"}
        _sens_cur_src = sidebar_params.get("alpha_source", "uniform")
        sens_alpha_source = st.radio(
            "Alpha source",
            options=_sens_alpha_src_opts,
            format_func=lambda v: _sens_alpha_src_labels[v],
            index=_sens_alpha_src_opts.index(_sens_cur_src) if _sens_cur_src in _sens_alpha_src_opts else 0,
            horizontal=True,
            key="sens_alpha_source",
        )
        _sens_cur_min = float(sidebar_params.get("alpha_min", 0.0))
        _sens_cur_max = float(sidebar_params.get("alpha_max", 1.0))
        _sens_alpha_range = st.slider(
            "Alpha range [min, max]",
            min_value=0.0, max_value=1.0,
            value=(_sens_cur_min, _sens_cur_max),
            step=0.01, format="%.2f",
            key="sens_alpha_range",
            help="Constrains alpha values. With 'Gmyrek data': only occupations in this score range are sampled.",
        )
        sens_alpha_min = _sens_alpha_range[0]
        sens_alpha_max = _sens_alpha_range[1]
    # Merge alpha settings into the base_params used for sweeps
    sidebar_params = {
        **sidebar_params,
        "alpha_source": sens_alpha_source,
        "alpha_min": sens_alpha_min,
        "alpha_max": sens_alpha_max,
    }

    # =========================================================
    # Output metrics selector (shared)
    # =========================================================
    metrics_for_final = st.multiselect(
        "Metrics for final-value & time-series plots",
        options=PLOT_METRICS,
        default=DEFAULT_SELECTED_METRICS,
        format_func=lambda m: METRIC_LABELS.get(m, m),
        key="sens_final_metrics",
    )

    # =========================================================
    # Save setup
    # =========================================================
    with st.expander("Save current setup (config only)", expanded=False):
        setup_save_name = st.text_input("Setup name", value="my_sensitivity_setup", key="sens_setup_save_name")
        if st.button("Save setup", key="sens_setup_save_btn"):
            setup_dict = {
                "sweep_type": "single" if sweep_type == "Single parameter" else "multi",
                "mode": sens_mode,
                "n_steps": int(sens_steps),
                "param_name": sens_param if sweep_type == "Single parameter" else "",
                "param_configs": param_configs,
                "method": method,
                "n_samples": int(n_samples) if sweep_type != "Single parameter" else None,
                "seed": int(sens_seed) if sweep_type != "Single parameter" else 42,
            }
            save_sensitivity_setup(setup_dict, APP_DIR, setup_save_name.strip() or "setup")
            st.success(f"Setup '{setup_save_name}' saved.")

    # =========================================================
    # Run button
    # =========================================================
    can_run = (sweep_type == "Single parameter" and len(param_values) > 0) or \
              (sweep_type != "Single parameter" and len(param_configs) > 0)
    if not can_run:
        st.warning("Define at least one parameter range before running." if sweep_type != "Single parameter" else "Define a valid parameter range first.")
        return

    if st.button("Run sensitivity sweep", type="primary"):
        sens_bar = st.progress(0.0, text="Starting sweep...")

        def sens_progress(label, done, total):
            sens_bar.progress(done / max(total, 1), text=f"{label} ({done}/{total})")

        if sweep_type == "Single parameter":
            st.session_state["sensitivity_results"] = {
                "sweep_type": "single",
                "param_name": sens_param,
                "param_values": param_values,
                "mode": sens_mode,
                "base_params": sidebar_params,
                "sweep": run_sensitivity_sweep(
                    base_params=sidebar_params,
                    param_name=sens_param,
                    param_values=param_values,
                    n_steps=int(sens_steps),
                    mode=sens_mode,
                    progress_callback=sens_progress,
                ),
            }
        else:
            st.session_state["sensitivity_results"] = {
                "sweep_type": "multi",
                "param_name": ", ".join(pc["name"] for pc in param_configs),
                "param_configs": param_configs,
                "method": method,
                "mode": sens_mode,
                "base_params": sidebar_params,
                "sweep": run_sensitivity_sweep_multi(
                    base_params=sidebar_params,
                    param_configs=param_configs,
                    method=method,
                    n_samples=int(n_samples),
                    n_steps=int(sens_steps),
                    mode=sens_mode,
                    seed=int(sens_seed),
                    progress_callback=sens_progress,
                ),
            }
        sens_bar.progress(1.0, text="Sweep complete.")

    sens_state = st.session_state.get("sensitivity_results")
    if not sens_state:
        return

    # =========================================================
    # Save results
    # =========================================================
    with st.expander("Save sweep results", expanded=False):
        default_name = sens_state.get("param_name", "sweep").replace(", ", "_")
        sens_save_name = st.text_input("Sweep name", value=f"{default_name}_sweep", key="sens_save_name")
        if st.button("Save sweep", type="primary", key="sens_save_btn"):
            save_sensitivity_sweep(sens_state=sens_state, base_dir=APP_DIR, sweep_name=sens_save_name.strip() or "sweep")
            st.success(f"Sweep '{sens_save_name}' saved.")

    # =========================================================
    # Results display
    # =========================================================
    sweep = sens_state["sweep"]
    mode = sens_state["mode"]
    param_name = sens_state.get("param_name", "")
    is_multi_result = sens_state.get("sweep_type") == "multi"

    st.divider()
    st.subheader(f"Results — {param_name}  ({MODE_LABELS.get(mode, mode)})")
    st.caption(f"{len(sweep)} runs · mode: {MODE_LABELS.get(mode, mode)}" +
               (f" · method: {sens_state.get('method','')}" if is_multi_result else ""))

    res_tab1, res_tab2, res_tab3, res_tab4, res_tab5 = st.tabs(["Time-series", "Final-value plots", "Summary table", "Alpha × workforce", "Complexity × workforce"])

    with res_tab1:
        st.markdown("**Multi-metric time-series across all runs**")
        st.caption("Each line = one run. Use the filter below to highlight only runs that meet your conditions.")

        # Timestep range filter
        all_steps = []
        for item in sweep:
            if not item["df"].empty:
                all_steps = list(item["df"].index)
                break
        max_step = max(all_steps) if all_steps else int(sens_steps)
        ts_filter_col1, ts_filter_col2 = st.columns(2)
        with ts_filter_col1:
            ts_step_min = st.number_input("Show from step", min_value=0, max_value=max_step, value=0, step=10, key="sens_ts_min")
        with ts_filter_col2:
            ts_step_max = st.number_input("Show until step", min_value=0, max_value=max_step, value=max_step, step=10, key="sens_ts_max")

        # Handle "Select all" before the widget renders; Streamlit does not
        # allow changing a widget-backed session_state key after instantiation.
        if st.session_state.pop("_sens_ts_select_all_pending", False):
            st.session_state["sens_ts_metrics_multi"] = list(PLOT_METRICS)
        _sens_metric_col1, _sens_metric_col2 = st.columns([6, 1])
        with _sens_metric_col1:
            ts_metrics = st.multiselect(
                "Metrics to plot",
                options=PLOT_METRICS,
                default=metrics_for_final[:4] if metrics_for_final else ["ai_adoption_rate"],
                format_func=lambda m: METRIC_LABELS.get(m, m),
                key="sens_ts_metrics_multi",
            )
        with _sens_metric_col2:
            st.write("")
            if st.button("Select all", key="sens_ts_select_all_metrics"):
                st.session_state["_sens_ts_select_all_pending"] = True
                st.rerun()

        _sens_rolling_col1, _sens_rolling_col2 = st.columns([1, 3])
        with _sens_rolling_col1:
            _sens_rolling_on = st.toggle(
                "Rolling average",
                value=False,
                key="sens_rolling_on",
                help="Smooth each run's line with a centred moving average. Useful to cut through step-to-step noise.",
            )
        with _sens_rolling_col2:
            if _sens_rolling_on:
                _sens_rolling_win = st.slider(
                    "Window (steps)", min_value=2, max_value=50, value=10, step=1,
                    key="sens_rolling_win",
                )
            else:
                _sens_rolling_win = 1

        # ── Conditional run filter ────────────────────────────────────────────
        with st.expander("Filter runs by condition", expanded=False):
            st.caption(
                "Define one or more conditions. Only runs that satisfy **all** conditions "
                "will be highlighted; the rest are shown in grey."
            )
            OPERATORS = [">", ">=", "<", "<=", "=="]

            if "sens_filter_conditions" not in st.session_state:
                st.session_state["sens_filter_conditions"] = []

            cond_list = st.session_state["sens_filter_conditions"]

            # Add / remove condition rows
            add_col, clear_col = st.columns([1, 1])
            with add_col:
                if st.button("+ Add condition", key="sens_cond_add"):
                    cond_list.append({"metric": PLOT_METRICS[0], "step": 0, "op": ">", "value": 0.0})
            with clear_col:
                if st.button("Clear all conditions", key="sens_cond_clear"):
                    cond_list.clear()

            conds_to_remove = []
            for ci, cond in enumerate(cond_list):
                cc1, cc2, cc3, cc4, cc5 = st.columns([3, 2, 1, 2, 1])
                with cc1:
                    cond["metric"] = st.selectbox(
                        "Metric", PLOT_METRICS,
                        index=PLOT_METRICS.index(cond["metric"]) if cond["metric"] in PLOT_METRICS else 0,
                        format_func=lambda m: METRIC_LABELS.get(m, m),
                        key=f"sens_cond_metric_{ci}",
                    )
                with cc2:
                    cond["step"] = st.number_input(
                        "At step", min_value=0, max_value=max_step,
                        value=min(int(cond["step"]), max_step),
                        step=1, key=f"sens_cond_step_{ci}",
                    )
                with cc3:
                    cond["op"] = st.selectbox(
                        "Op", OPERATORS,
                        index=OPERATORS.index(cond["op"]) if cond["op"] in OPERATORS else 0,
                        key=f"sens_cond_op_{ci}",
                    )
                with cc4:
                    cond["value"] = st.number_input(
                        "Threshold", value=float(cond["value"]),
                        step=0.01, format="%.4f", key=f"sens_cond_val_{ci}",
                    )
                with cc5:
                    if st.button("✕", key=f"sens_cond_rm_{ci}"):
                        conds_to_remove.append(ci)

            for ci in reversed(conds_to_remove):
                cond_list.pop(ci)

            # Compute matching indices
            highlight_indices = None
            matched_runs = []
            if cond_list:
                import operator as _op
                _OP_MAP = {">": _op.gt, ">=": _op.ge, "<": _op.lt, "<=": _op.le, "==": _op.eq}
                matching = []
                for idx, item in enumerate(sweep):
                    df = item["df"]
                    ok = True
                    for cond in cond_list:
                        m, step_t, op_str, thresh = cond["metric"], int(cond["step"]), cond["op"], float(cond["value"])
                        if m not in df.columns:
                            ok = False
                            break
                        # Find closest step index
                        pos = df.index.searchsorted(step_t, side="left")
                        if pos >= len(df.index):
                            closest_idx = df.index[-1]
                        else:
                            closest_idx = df.index[pos]
                        val = pd.to_numeric(df.loc[closest_idx, m], errors="coerce")
                        if pd.isna(val) or not _OP_MAP[op_str](val, thresh):
                            ok = False
                            break
                    if ok:
                        matching.append(idx)
                        matched_runs.append(item)

                highlight_indices = matching if matching else []
                st.info(f"{len(matching)} / {len(sweep)} runs match all conditions.")

        # ── Charts ────────────────────────────────────────────────────────────
        for ts_m in ts_metrics:
            fig_ts = build_sensitivity_multirun_timeseries(
                sweep, ts_m,
                step_min=int(ts_step_min),
                step_max=int(ts_step_max) if ts_step_max < max_step else None,
                highlight_indices=highlight_indices,
                rolling=_sens_rolling_win,
            )
            st.pyplot(fig_ts, use_container_width=True)
            plt.close(fig_ts)

        # ── Matching runs parameter table ─────────────────────────────────────
        if highlight_indices is not None and matched_runs:
            st.markdown("**Matching runs — parameter values**")
            param_rows = []
            for item in matched_runs:
                if "param_values" in item:
                    row = dict(item["param_values"])
                else:
                    pn = sens_state.get("param_name", "param")
                    row = {pn: item["param_value"]}
                # Add final-step values for each selected metric, read from df directly
                # so any metric in PLOT_METRICS works (not just KEY_METRICS subset)
                _df = item["df"]
                for ts_m in ts_metrics:
                    if ts_m in _df.columns:
                        raw = _df[ts_m].iloc[-1]
                        row[METRIC_LABELS.get(ts_m, ts_m)] = round(float(raw), 4) if raw == raw else float("nan")
                    else:
                        row[METRIC_LABELS.get(ts_m, ts_m)] = float("nan")
                param_rows.append(row)
            st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)
        elif highlight_indices is not None and not matched_runs:
            st.warning("No runs match the current filter conditions.")

    with res_tab2:
        if not is_multi_result and metrics_for_final:
            st.markdown("**Final-step outcome vs parameter value**")
            final_fig = build_sensitivity_final_figure(sweep, param_name, metrics_for_final)
            st.pyplot(final_fig, use_container_width=True)
            plt.close(final_fig)
        elif is_multi_result:
            st.info("Final-value scatter is designed for single-param sweeps. Use the Summary table tab to compare multi-param run outcomes.")

    with res_tab3:
        st.markdown("**Summary table — final-step values per run**")
        rows = []
        for item in sweep:
            if "param_values" in item:
                row = {k: v for k, v in item["param_values"].items()}
            else:
                row = {param_name: item["param_value"]}
            row.update({METRIC_LABELS.get(m, m): round(v, 4) for m, v in item["final"].items()})
            rows.append(row)
        summary_df = pd.DataFrame(rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    def _sens_run_label(item: dict, idx: int) -> str:
        if "param_values" in item:
            return f"Run {idx+1}: " + ", ".join(f"{k}={v}" for k, v in item["param_values"].items())
        return f"Run {idx+1}: {param_name}={item.get('param_value', '?')}"

    _has_models = any(item.get("model") is not None for item in sweep)

    with res_tab4:
        if not _has_models:
            st.info("Model objects are not available for loaded sweeps. Re-run the sweep to see this plot.")
        else:
            st.caption(
                "Alpha is binned in fixed steps of 0.05 from 0 to 1 (20 bins total). "
                "**Top:** task status per alpha bin (AI / High-skill / Low-skill / Mixed / Empty). "
                "**Bottom:** absolute worker-slots per alpha bin."
            )
            run_labels_aw = [_sens_run_label(item, i) for i, item in enumerate(sweep) if item.get("model") is not None]
            run_items_aw  = [item for item in sweep if item.get("model") is not None]
            sel_aw = st.selectbox("Select run", options=range(len(run_labels_aw)),
                                  format_func=lambda i: run_labels_aw[i], key="sens_aw_run")
            if run_items_aw:
                mini_bundle = make_single_run_bundle(run_items_aw[sel_aw], mode)
                st.markdown("##### Task status by alpha range")
                ast_fig = build_alpha_status_figure(mini_bundle)
                st.pyplot(ast_fig, use_container_width=True)
                plt.close(ast_fig)
                st.markdown("##### Absolute worker count by alpha range")
                aw_fig = build_alpha_workforce_figure(mini_bundle)
                st.pyplot(aw_fig, use_container_width=True)
                plt.close(aw_fig)

    with res_tab5:
        if not _has_models:
            st.info("Model objects are not available for loaded sweeps. Re-run the sweep to see this plot.")
        else:
            run_labels_tw = [_sens_run_label(item, i) for i, item in enumerate(sweep) if item.get("model") is not None]
            run_items_tw  = [item for item in sweep if item.get("model") is not None]
            sel_tw = st.selectbox("Select run", options=range(len(run_labels_tw)),
                                  format_func=lambda i: run_labels_tw[i], key="sens_tw_run")
            if run_items_tw:
                mini_bundle = make_single_run_bundle(run_items_tw[sel_tw], mode)

                st.markdown("#### Task status by complexity")
                st.caption(
                    "Each task is classified as AI, High-skill only, Low-skill only, Mixed (both), "
                    "or Empty (no workers). The bars show how many tasks fall into each status "
                    "per complexity level (1 = simplest)."
                )
                ts_fig = build_task_complexity_status_figure(mini_bundle)
                st.pyplot(ts_fig, use_container_width=True)
                plt.close(ts_fig)

                st.markdown("#### Absolute worker count by task complexity")
                st.caption(
                    "Total number of AI units, high-skill, and low-skill worker-slots across all "
                    "tasks of each complexity level. Bar height reflects both task frequency and "
                    "staffing intensity."
                )
                tw_fig = build_task_complexity_workforce_figure(mini_bundle)
                st.pyplot(tw_fig, use_container_width=True)
                plt.close(tw_fig)


@st.cache_data(show_spinner=False)
def _load_ofat_timeseries(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Step" in df.columns:
        df = df.set_index("Step")
    elif df.columns[0].lower() == "unnamed: 0":
        df = df.set_index(df.columns[0])
    return df


_PERCENT_METRICS = {"ai_adoption_rate", "employment_rate_high", "employment_rate_low", "labour_share"}


def _format_parameter_label(parameter: str, include_raw_name: bool = False) -> str:
    meta = PARAMETER_META.get(parameter, {})
    label = meta.get("label") or parameter.replace("_", " ").title()
    return f"{label} ({parameter})" if include_raw_name and label != parameter else label


def _find_ofat_baseline_row(group: pd.DataFrame, metric_col: str) -> pd.Series | None:
    if group.empty:
        return None

    finite = group[np.isfinite(pd.to_numeric(group[metric_col], errors="coerce"))].copy()
    if finite.empty:
        return None

    labels = finite.get("factor_label", pd.Series(index=finite.index, dtype=object)).astype(str).str.lower()
    default_mask = labels.str.contains("default") | np.isclose(
        pd.to_numeric(finite.get("factor_pct"), errors="coerce").fillna(np.inf),
        0.0,
    )
    if default_mask.any():
        return finite.loc[default_mask].sort_values("grid_index").iloc[0]

    param_values = pd.to_numeric(finite["param_value"], errors="coerce")
    default_values = pd.to_numeric(finite["default_value"], errors="coerce")
    if param_values.notna().any() and default_values.notna().any():
        distance = (param_values - default_values).abs()
        if distance.notna().any():
            return finite.loc[distance.idxmin()]

    return finite.sort_values("grid_index").iloc[len(finite) // 2]


def _build_ofat_tornado_data(
    aggregated: pd.DataFrame,
    metric_col: str,
    modes: list[str],
    effect_scale: str,
    ranking_basis: str,
    include_bool: bool,
    include_zero_effects: bool,
) -> pd.DataFrame:
    rows = []
    std_col = metric_col.replace("final_mean__", "final_std__", 1)
    source = aggregated[aggregated["mode"].isin(modes)].copy()
    if not include_bool:
        source = source[~source["parameter"].isin(_BOOL_PARAMS)]

    for (parameter, mode), group in source.groupby(["parameter", "mode"], dropna=False):
        group = group.copy()
        group[metric_col] = pd.to_numeric(group[metric_col], errors="coerce")
        group = group[np.isfinite(group[metric_col])]
        if group.empty:
            continue

        baseline_row = _find_ofat_baseline_row(group, metric_col)
        if baseline_row is None:
            continue

        baseline = float(baseline_row[metric_col])
        low_row = group.loc[group[metric_col].idxmin()]
        high_row = group.loc[group[metric_col].idxmax()]
        low_value = float(low_row[metric_col])
        high_value = float(high_row[metric_col])
        low_effect = low_value - baseline
        high_effect = high_value - baseline
        low_std = float(low_row.get(std_col, np.nan)) if std_col in group.columns else np.nan
        high_std = float(high_row.get(std_col, np.nan)) if std_col in group.columns else np.nan

        if effect_scale == "Percent change from baseline":
            denom = abs(baseline)
            if denom > 1e-12:
                low_plot = 100.0 * low_effect / denom
                high_plot = 100.0 * high_effect / denom
                low_endpoint = 100.0 * (low_value - baseline) / denom
                high_endpoint = 100.0 * (high_value - baseline) / denom
                low_std_plot = 100.0 * low_std / denom if np.isfinite(low_std) else np.nan
                high_std_plot = 100.0 * high_std / denom if np.isfinite(high_std) else np.nan
            else:
                low_plot = high_plot = low_endpoint = high_endpoint = np.nan
                low_std_plot = high_std_plot = np.nan
        else:
            low_plot = low_effect
            high_plot = high_effect
            low_endpoint = low_value
            high_endpoint = high_value
            low_std_plot = low_std
            high_std_plot = high_std

        if ranking_basis == "Outcome range":
            influence = abs(high_value - low_value)
            if effect_scale == "Percent change from baseline" and abs(baseline) > 1e-12:
                influence = 100.0 * influence / abs(baseline)
        elif ranking_basis == "Positive upside":
            influence = max(high_plot, low_plot, 0.0)
        elif ranking_basis == "Negative downside":
            influence = abs(min(high_plot, low_plot, 0.0))
        else:
            influence = max(abs(low_plot), abs(high_plot))

        if not include_zero_effects and (not np.isfinite(influence) or influence <= 1e-12):
            continue

        rows.append(
            {
                "parameter": parameter,
                "mode": mode,
                "baseline": baseline,
                "baseline_label": baseline_row.get("factor_label", ""),
                "baseline_param_value": baseline_row.get("param_value"),
                "low_effect": low_plot,
                "high_effect": high_plot,
                "low_endpoint": low_endpoint,
                "high_endpoint": high_endpoint,
                "low_std": low_std_plot,
                "high_std": high_std_plot,
                "low_label": low_row.get("factor_label", ""),
                "high_label": high_row.get("factor_label", ""),
                "low_param_value": low_row.get("param_value"),
                "high_param_value": high_row.get("param_value"),
                "influence": influence,
                "replicates": int(group["replicates"].max()) if "replicates" in group.columns and group["replicates"].notna().any() else np.nan,
            }
        )

    return pd.DataFrame(rows)


def _format_ofat_export_grid_label(row: pd.Series) -> str:
    factor_pct = pd.to_numeric(pd.Series([row.get("factor_pct")]), errors="coerce").iloc[0]
    if pd.notna(factor_pct):
        return f"{factor_pct:.0%}"
    return str(row.get("factor_label", ""))


def _build_ofat_tornado_export_table(
    aggregated: pd.DataFrame,
    selected_rows: pd.DataFrame,
    metric_col: str,
    effect_scale: str,
    group_modes: bool,
) -> pd.DataFrame:
    preferred_columns = ["-50%", "-25%", "0%", "25%", "50%"]
    table_rows = []
    extra_labels = []

    for _, selected in selected_rows.iterrows():
        parameter = selected["parameter"]
        mode = selected["mode"]
        group = aggregated[
            (aggregated["parameter"] == parameter)
            & (aggregated["mode"] == mode)
        ].copy()
        if group.empty:
            continue

        group[metric_col] = pd.to_numeric(group[metric_col], errors="coerce")
        group = group[np.isfinite(group[metric_col])].sort_values("grid_index")
        baseline_row = _find_ofat_baseline_row(group, metric_col)
        if baseline_row is None:
            continue

        baseline = float(baseline_row[metric_col])
        variable_label = _format_parameter_label(parameter, True)
        if group_modes:
            variable_label = f"{variable_label} | {MODE_LABELS.get(mode, mode)}"

        row = {"Variable": variable_label}
        for _, grid_row in group.iterrows():
            label = _format_ofat_export_grid_label(grid_row)
            value = float(grid_row[metric_col])
            if effect_scale == "Percent change from baseline":
                effect = 100.0 * (value - baseline) / abs(baseline) if abs(baseline) > 1e-12 else np.nan
            else:
                effect = value - baseline
            row[label] = effect
            if label not in preferred_columns and label not in extra_labels:
                extra_labels.append(label)
        table_rows.append(row)

    output = pd.DataFrame(table_rows)
    if output.empty:
        return output

    ordered_columns = ["Variable"] + preferred_columns + extra_labels
    for column in ordered_columns:
        if column not in output.columns:
            output[column] = np.nan
    return output[ordered_columns]


def _build_ofat_tornado_export_workbook(
    export_table: pd.DataFrame,
    settings: dict,
) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_table.to_excel(writer, sheet_name="Tornado effects", index=False)
        pd.DataFrame(
            [{"Setting": key, "Value": value} for key, value in settings.items()]
        ).to_excel(writer, sheet_name="Settings", index=False)
    return output.getvalue()


def _render_ofat_tornado_panel(
    aggregated: pd.DataFrame,
    final_metric_columns: list[str],
    selected_metric_col: str,
    selected_mode: str,
) -> None:
    st.markdown("**Tornado plot**")
    st.caption("Rank parameters by how much their OFAT grid changes the selected final metric.")

    all_modes = sorted(
        aggregated["mode"].dropna().unique().tolist(),
        key=lambda x: MODES.index(x) if x in MODES else x,
    )
    available_parameters = sorted(
        aggregated["parameter"].dropna().unique().tolist(),
        key=lambda x: (0, OFAT_PARAMS.index(x)) if x in OFAT_PARAMS else (1, str(x)),
    )
    default_modes = [selected_mode] if selected_mode in all_modes else all_modes[:1]

    metric_col = st.selectbox(
        "Tornado metric",
        options=final_metric_columns,
        index=final_metric_columns.index(selected_metric_col) if selected_metric_col in final_metric_columns else 0,
        format_func=lambda c: METRIC_LABELS.get(c.replace("final_mean__", ""), c.replace("final_mean__", "")),
        key="ofat_tornado_metric",
    )
    metric_name = metric_col.replace("final_mean__", "")
    metric_label = METRIC_LABELS.get(metric_name, metric_name)

    config_col1, config_col2, config_col3, config_col4 = st.columns(4)
    with config_col1:
        tornado_modes = st.multiselect(
            "Modes",
            options=all_modes,
            default=default_modes,
            format_func=lambda m: MODE_LABELS.get(m, m),
            key="ofat_tornado_modes",
        )
        effect_scale = st.selectbox(
            "Effect scale",
            options=["Metric units", "Percent change from baseline"],
            key="ofat_tornado_effect_scale",
        )
    with config_col2:
        ranking_basis = st.selectbox(
            "Influentiality",
            options=["Max absolute deviation", "Outcome range", "Positive upside", "Negative downside"],
            key="ofat_tornado_rank_basis",
        )
        sort_order = st.selectbox(
            "Sort order",
            options=["Most influential first", "Least influential first"],
            key="ofat_tornado_sort_order",
        )
    with config_col3:
        include_bool = st.toggle("Include boolean toggles", value=True, key="ofat_tornado_include_bool")
        include_zero_effects = st.toggle("Include zero effects", value=False, key="ofat_tornado_include_zero")
    with config_col4:
        group_modes = st.toggle("Separate bars per mode", value=len(tornado_modes) > 1, key="ofat_tornado_group_modes")
        max_top_n = max(1, len(available_parameters) * (max(1, len(tornado_modes)) if group_modes else 1))
        if "ofat_tornado_top_n" in st.session_state:
            st.session_state["ofat_tornado_top_n"] = min(max(int(st.session_state["ofat_tornado_top_n"]), 1), max_top_n)
        top_n = st.slider("Top variables", min_value=1, max_value=max_top_n, value=min(10, max_top_n), key="ofat_tornado_top_n")

    with st.expander("Customize appearance", expanded=False):
        app_col1, app_col2, app_col3, app_col4 = st.columns(4)
        with app_col1:
            positive_color = st.color_picker("Positive bar", value="#2563eb", key="ofat_tornado_positive_color")
            negative_color = st.color_picker("Negative bar", value="#f97316", key="ofat_tornado_negative_color")
        with app_col2:
            figure_height = st.slider("Figure height", 4.0, 14.0, 7.0, 0.5, key="ofat_tornado_height")
            label_raw_name = st.toggle("Show raw parameter names", value=True, key="ofat_tornado_raw_names")
        with app_col3:
            show_values = st.toggle("Annotate bars", value=True, key="ofat_tornado_show_values")
            show_uncertainty = st.toggle("Show replicate std", value=False, key="ofat_tornado_show_uncertainty")
        with app_col4:
            x_symmetric = st.toggle("Symmetric x-axis", value=True, key="ofat_tornado_symmetric")
            show_table = st.toggle("Show ranking table", value=True, key="ofat_tornado_show_table")
            grid_alpha = st.slider("Grid strength", 0.0, 1.0, 0.35, 0.05, key="ofat_tornado_grid_alpha")

    if not tornado_modes:
        st.info("Choose at least one mode for the tornado plot.")
        return

    tornado_df = _build_ofat_tornado_data(
        aggregated=aggregated,
        metric_col=metric_col,
        modes=tornado_modes,
        effect_scale=effect_scale,
        ranking_basis=ranking_basis,
        include_bool=include_bool,
        include_zero_effects=include_zero_effects,
    )
    if tornado_df.empty:
        st.info("No finite OFAT effects found for this metric and mode selection.")
        return

    ascending = sort_order == "Least influential first"
    tornado_df = tornado_df.sort_values(["influence", "parameter", "mode"], ascending=[ascending, True, True])
    if not group_modes and len(tornado_modes) > 1:
        tornado_df = tornado_df.drop_duplicates("parameter", keep="first")
    tornado_df = tornado_df.head(top_n)
    plot_df = tornado_df.copy()
    plot_df["label"] = plot_df["parameter"].map(lambda p: _format_parameter_label(p, label_raw_name))
    if group_modes:
        plot_df["label"] = plot_df.apply(lambda r: f"{r['label']} | {MODE_LABELS.get(r['mode'], r['mode'])}", axis=1)

    plot_df = plot_df.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(11, figure_height))
    for i, row in plot_df.iterrows():
        low = float(row["low_effect"])
        high = float(row["high_effect"])
        if np.isfinite(low):
            ax.barh(i, low, color=negative_color if low < 0 else positive_color, alpha=0.82, height=0.68)
        if np.isfinite(high):
            ax.barh(i, high, color=positive_color if high >= 0 else negative_color, alpha=0.82, height=0.68)
        if show_uncertainty:
            for value, std in ((low, row.get("low_std", np.nan)), (high, row.get("high_std", np.nan))):
                if np.isfinite(value) and np.isfinite(std) and std > 0:
                    ax.errorbar(value, i, xerr=std, fmt="none", ecolor="#334155", elinewidth=1.0, capsize=2.5, alpha=0.75)
        if show_values:
            for value in (low, high):
                if not np.isfinite(value) or abs(value) <= 1e-12:
                    continue
                ha = "left" if value >= 0 else "right"
                offset = 0.012 * max(plot_df[["low_effect", "high_effect"]].abs().max().max(), 1e-9)
                label = f"{value:+.1f}%" if effect_scale == "Percent change from baseline" else f"{value:+.3g}"
                ax.text(value + (offset if value >= 0 else -offset), i, label, va="center", ha=ha, fontsize=8)

    ax.axvline(0, color="#111827", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["label"].tolist(), fontsize=8.5)
    ax.set_xlabel(
        f"Change in {metric_label}" + (" (%)" if effect_scale == "Percent change from baseline" else ""),
        fontsize=9,
    )
    ax.set_title(f"Top {len(plot_df)} OFAT drivers of {metric_label}", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=grid_alpha)
    if x_symmetric:
        max_abs = plot_df[["low_effect", "high_effect"]].abs().max().max()
        if np.isfinite(max_abs) and max_abs > 0:
            ax.set_xlim(-max_abs * 1.18, max_abs * 1.18)
    if effect_scale == "Metric units" and metric_name in _PERCENT_METRICS:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    export_table = _build_ofat_tornado_export_table(
        aggregated=aggregated,
        selected_rows=tornado_df,
        metric_col=metric_col,
        effect_scale=effect_scale,
        group_modes=group_modes,
    )
    if not export_table.empty:
        export_settings = {
            "Metric": metric_label,
            "Metric column": metric_col,
            "Effect scale": effect_scale,
            "Influentiality ranking": ranking_basis,
            "Sort order": sort_order,
            "Top variables": len(tornado_df),
            "Modes selected": ", ".join(MODE_LABELS.get(m, m) for m in tornado_modes),
            "Separate bars per mode": group_modes,
            "Include boolean toggles": include_bool,
            "Include zero effects": include_zero_effects,
            "Cell meaning": (
                "Percent change from that variable's baseline metric value"
                if effect_scale == "Percent change from baseline"
                else "Absolute change from that variable's baseline metric value"
            ),
        }
        export_bytes = _build_ofat_tornado_export_workbook(export_table, export_settings)
        safe_metric = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in metric_name).strip("_")
        st.download_button(
            "Export current tornado selection to Excel",
            data=export_bytes,
            file_name=f"ofat_tornado_{safe_metric}_top{len(tornado_df)}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="ofat_tornado_export_xlsx",
        )

    if show_table:
        table_df = tornado_df.copy()
        table_df["Mode"] = table_df["mode"].map(lambda m: MODE_LABELS.get(m, m))
        table_df["Parameter"] = table_df["parameter"].map(lambda p: _format_parameter_label(p, True))
        table_df = table_df.rename(
            columns={
                "influence": "Influentiality",
                "baseline": "Baseline metric",
                "baseline_label": "Baseline grid",
                "baseline_param_value": "Baseline value",
                "low_effect": "Low effect",
                "high_effect": "High effect",
                "low_std": "Low std",
                "high_std": "High std",
                "low_label": "Low grid",
                "high_label": "High grid",
                "low_param_value": "Low parameter value",
                "high_param_value": "High parameter value",
                "replicates": "Replicates",
            }
        )
        st.dataframe(
            table_df[
                [
                    "Parameter",
                    "Mode",
                    "Influentiality",
                    "Baseline metric",
                    "Baseline grid",
                    "Baseline value",
                    "Low effect",
                    "Low std",
                    "Low grid",
                    "Low parameter value",
                    "High effect",
                    "High std",
                    "High grid",
                    "High parameter value",
                    "Replicates",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


def _render_ofat_explorer():
    st.subheader("OFAT Explorer")
    st.caption(
        "Load a saved one-factor-at-a-time batch, inspect final effects across grid points, "
        "and compare replicate time-series without re-running the model."
    )

    saved_batches = list_ofat_batches(APP_DIR)
    if not saved_batches:
        st.info("No OFAT batches found yet. Run `python model/OFAT_sensitivity.py` first.")
        return

    with st.expander("Load saved OFAT batch", expanded=True):
        batch_options = {
            f"{b.get('batch_name', 'ofat')} ({b.get('created_at', '')})": b
            for b in saved_batches
        }
        selected_batch = st.selectbox(
            "Saved OFAT batch",
            options=[""] + list(batch_options.keys()),
            key="ofat_load_select",
        )
        load_col, del_col = st.columns(2)
        with load_col:
            if selected_batch and st.button("Load OFAT batch", use_container_width=True):
                st.session_state["ofat_batch"] = load_ofat_batch(batch_options[selected_batch]["path"])
                st.success(f"Loaded '{batch_options[selected_batch].get('batch_name', 'ofat')}'.")
                st.rerun()
        with del_col:
            if selected_batch and st.button("Delete OFAT batch", type="secondary", use_container_width=True):
                if delete_ofat_batch(batch_options[selected_batch]["path"]):
                    st.success("Deleted.")
                    if "ofat_batch" in st.session_state:
                        loaded_path = st.session_state["ofat_batch"].get("path")
                        if loaded_path == batch_options[selected_batch]["path"]:
                            st.session_state.pop("ofat_batch", None)
                    st.rerun()

    ofat_state = st.session_state.get("ofat_batch")
    if not ofat_state:
        return

    metadata = ofat_state["metadata"]
    summary = ofat_state["summary"].copy()
    aggregated = ofat_state["aggregated"].copy()
    timeseries_dir = Path(ofat_state["timeseries_dir"])

    if summary.empty or aggregated.empty:
        st.warning("This OFAT batch does not contain summary data.")
        return

    st.markdown("**Batch overview**")
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    info_col1.metric("Parameters", int(metadata.get("parameter_count", summary["parameter"].nunique())))
    info_col2.metric("Modes", int(len(metadata.get("modes", summary["mode"].dropna().unique().tolist()))))
    info_col3.metric("Replicates / point", int(metadata.get("replicates_per_point", 0)))
    info_col4.metric("Total runs", int(metadata.get("total_runs", len(summary))))
    st.caption(
        f"Steps: {metadata.get('n_steps')} | Seeds: {metadata.get('replicate_seeds')} | "
        f"Workers used when created: {metadata.get('workers')}"
    )

    final_metric_columns = sorted(
        col for col in aggregated.columns if col.startswith("final_mean__")
    )
    if not final_metric_columns:
        st.warning("No final metrics found in the aggregated OFAT file.")
        return

    control_col1, control_col2, control_col3 = st.columns(3)
    with control_col1:
        selected_param = st.selectbox(
            "Parameter",
            options=sorted(
                aggregated["parameter"].dropna().unique().tolist(),
                key=lambda x: (0, OFAT_PARAMS.index(x)) if x in OFAT_PARAMS else (1, str(x)),
            ),
            key="ofat_param",
        )
    with control_col2:
        available_modes = sorted(
            aggregated.loc[aggregated["parameter"] == selected_param, "mode"].dropna().unique().tolist(),
            key=lambda x: MODES.index(x) if x in MODES else x,
        )
        selected_mode = st.selectbox(
            "Mode for replicate view",
            options=available_modes,
            format_func=lambda m: MODE_LABELS.get(m, m),
            key="ofat_mode",
        )
    with control_col3:
        selected_metric_col = st.selectbox(
            "Final metric",
            options=final_metric_columns,
            format_func=lambda c: METRIC_LABELS.get(c.replace("final_mean__", ""), c.replace("final_mean__", "")),
            key="ofat_metric",
        )

    selected_metric = selected_metric_col.replace("final_mean__", "")
    selected_std_col = f"final_std__{selected_metric}"
    selected_param_all_modes = aggregated[
        aggregated["parameter"] == selected_param
    ].sort_values(["mode", "grid_index"])
    selected_param_data = selected_param_all_modes[
        selected_param_all_modes["mode"] == selected_mode
    ].sort_values("grid_index")

    if selected_param_all_modes.empty:
        st.warning("No aggregated OFAT rows found for this parameter.")
        return

    overview_tab, tornado_tab, timeseries_tab, summary_tab = st.tabs(
        ["Final effects", "Tornado plot", "Replicate time-series", "Summary table"]
    )

    with overview_tab:
        metric_label = METRIC_LABELS.get(selected_metric, selected_metric)
        fig, ax = plt.subplots(figsize=(10, 4.8))
        is_boolean_param = selected_param in _BOOL_PARAMS
        for mode in available_modes:
            mode_data = selected_param_all_modes[selected_param_all_modes["mode"] == mode].sort_values("grid_index")
            if mode_data.empty:
                continue
            if is_boolean_param:
                x = np.arange(len(mode_data))
                x_labels = mode_data["factor_label"].tolist()
            else:
                numeric_x = pd.to_numeric(mode_data["param_value"], errors="coerce")
                mode_data = mode_data.loc[numeric_x.notna()].copy()
                if mode_data.empty:
                    continue
                x = numeric_x.loc[mode_data.index].to_numpy(dtype=float)
                x_labels = None
            y = mode_data[selected_metric_col].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                marker=MODE_MARKERS.get(mode, "o"),
                linewidth=2.0,
                color=MODE_COLORS.get(mode),
                label=MODE_LABELS.get(mode, mode),
            )
            if selected_std_col in mode_data.columns:
                std = mode_data[selected_std_col].fillna(0.0).to_numpy(dtype=float)
                ax.fill_between(x, y - std, y + std, color=MODE_COLORS.get(mode), alpha=0.12)
        ax.set_title(f"{metric_label} vs {selected_param} - all modes", fontsize=11, fontweight="bold")
        ax.set_xlabel(selected_param, fontsize=9)
        ax.set_ylabel(metric_label, fontsize=9)
        if is_boolean_param:
            ax.set_xticks(np.arange(len(selected_param_all_modes[selected_param_all_modes["mode"] == available_modes[0]])))
            ax.set_xticklabels(selected_param_all_modes[selected_param_all_modes["mode"] == available_modes[0]]["factor_label"].tolist())
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, loc="best")
        if selected_metric in ("ai_adoption_rate", "employment_rate_high", "employment_rate_low", "labour_share"):
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        display_cols = ["mode", "factor_label", "param_value", selected_metric_col]
        if selected_std_col in selected_param_all_modes.columns:
            display_cols.append(selected_std_col)
        display_df = selected_param_all_modes[display_cols].rename(
            columns={
                "mode": "Mode",
                "factor_label": "Grid point",
                "param_value": "Parameter value",
                selected_metric_col: f"Mean {metric_label}",
                selected_std_col: f"Std {metric_label}",
            }
        )
        display_df["Mode"] = display_df["Mode"].map(lambda m: MODE_LABELS.get(m, m))
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tornado_tab:
        _render_ofat_tornado_panel(
            aggregated=aggregated,
            final_metric_columns=final_metric_columns,
            selected_metric_col=selected_metric_col,
            selected_mode=selected_mode,
        )

    with timeseries_tab:
        factor_options = selected_param_data["factor_label"].tolist()
        selected_factor = st.selectbox("Grid point", options=factor_options, key="ofat_factor")
        selected_rows = summary[
            (summary["parameter"] == selected_param)
            & (summary["mode"] == selected_mode)
            & (summary["factor_label"] == selected_factor)
        ].sort_values("replicate_index")
        if selected_rows.empty:
            st.info("No replicate files found for this grid point.")
        else:
            ts_fig, ts_ax = plt.subplots(figsize=(10, 5))
            stacked = []
            for _, row in selected_rows.iterrows():
                csv_path = timeseries_dir / row["timeseries_file"]
                if not csv_path.exists():
                    continue
                df = _load_ofat_timeseries(str(csv_path))
                if selected_metric not in df.columns:
                    continue
                series = pd.to_numeric(df[selected_metric], errors="coerce")
                ts_ax.plot(df.index, series, color="#cbd5e1", linewidth=1.0, alpha=0.9)
                stacked.append(series.rename(int(row["seed"])))
            if stacked:
                mean_df = pd.concat(stacked, axis=1)
                ts_ax.plot(mean_df.index, mean_df.mean(axis=1), color="#1d4ed8", linewidth=2.4, label="Replicate mean")
                ts_ax.set_title(
                    f"{METRIC_LABELS.get(selected_metric, selected_metric)} across 10 replicates",
                    fontsize=11,
                    fontweight="bold",
                )
                ts_ax.set_xlabel("Model step", fontsize=9)
                ts_ax.set_ylabel(METRIC_LABELS.get(selected_metric, selected_metric), fontsize=9)
                ts_ax.grid(True, linestyle="--", alpha=0.5)
                if selected_metric in ("ai_adoption_rate", "employment_rate_high", "employment_rate_low", "labour_share"):
                    ts_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
                ts_ax.legend(fontsize=8, loc="best")
                st.pyplot(ts_fig, use_container_width=True)
                plt.close(ts_fig)
            else:
                plt.close(ts_fig)
                st.warning("The selected metric is not available in the replicate CSV files.")

        compare_all = st.checkbox(
            f"Overlay mean line for all {len(factor_options)} grid points",
            value=False,
            key="ofat_overlay_all",
        )
        if compare_all:
            fig_all, ax_all = plt.subplots(figsize=(10, 5))
            for _, grid_row in selected_param_data.iterrows():
                rows = summary[
                    (summary["parameter"] == selected_param)
                    & (summary["mode"] == selected_mode)
                    & (summary["factor_label"] == grid_row["factor_label"])
                ].sort_values("replicate_index")
                per_rep = []
                for _, row in rows.iterrows():
                    csv_path = timeseries_dir / row["timeseries_file"]
                    if not csv_path.exists():
                        continue
                    df = _load_ofat_timeseries(str(csv_path))
                    if selected_metric not in df.columns:
                        continue
                    per_rep.append(pd.to_numeric(df[selected_metric], errors="coerce"))
                if not per_rep:
                    continue
                overlay_df = pd.concat(per_rep, axis=1)
                ax_all.plot(
                    overlay_df.index,
                    overlay_df.mean(axis=1),
                    linewidth=2.0,
                    label=f"{grid_row['factor_label']} ({grid_row['param_value']})",
                )
            ax_all.set_title(
                f"Mean time-series by grid point: {METRIC_LABELS.get(selected_metric, selected_metric)}",
                fontsize=11,
                fontweight="bold",
            )
            ax_all.set_xlabel("Model step", fontsize=9)
            ax_all.set_ylabel(METRIC_LABELS.get(selected_metric, selected_metric), fontsize=9)
            ax_all.grid(True, linestyle="--", alpha=0.5)
            ax_all.legend(fontsize=8, loc="best")
            if selected_metric in ("ai_adoption_rate", "employment_rate_high", "employment_rate_low", "labour_share"):
                ax_all.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            st.pyplot(fig_all, use_container_width=True)
            plt.close(fig_all)

    with summary_tab:
        rename_map = {
            "parameter": "Parameter",
            "mode": "Mode",
            "factor_label": "Grid point",
            "param_value": "Parameter value",
            "default_value": "Default",
            "seed": "Seed",
            "runtime_seconds": "Runtime (s)",
        }
        st.dataframe(
            summary[
                (summary["parameter"] == selected_param) & (summary["mode"] == selected_mode)
            ].rename(columns=rename_map),
            use_container_width=True,
            hide_index=True,
        )


def main():
    st.set_page_config(page_title="Labour Market ABM Dashboard", layout="wide")
    _inject_styles()
    _apply_pending_sidebar_state()

    st.title("Labour Market ABM Dashboard")
    st.caption("Run local simulations, inspect parameters in plain language, and compare outcomes without unreadable spaghetti plots.")

    run_clicked, run_label, n_steps, selected_modes, params, log_decisions = _build_sidebar()

    latest_run_dir = st.session_state.get("latest_run_dir")
    if run_clicked:
        if not selected_modes:
            st.sidebar.error("Choose at least one mode.")
        else:
            progress_bar = st.sidebar.progress(0.0, text="Simulation starting...")

            def progress(mode: str, completed: int, total: int):
                ratio = completed / max(total, 1)
                progress_bar.progress(ratio, text=f"Running {mode} ({completed}/{total})")

            bundle = run_simulation(
                params=params,
                n_steps=n_steps,
                modes=selected_modes,
                progress_callback=progress,
                run_label=run_label,
                log_decisions=log_decisions,
            )
            st.session_state["latest_bundle"] = bundle
            latest_run_dir = save_run_to_history(bundle, APP_DIR)
            st.session_state["latest_run_dir"] = str(latest_run_dir)
            progress_bar.progress(1.0, text="Run completed")

    page_tab1, page_tab2, page_tab3, page_tab4, page_tab5, page_tab6, page_tab7, page_tab8 = st.tabs([
        "Dashboard", "Saved runs", "Experimenter", "Sensitivity", "OFAT Explorer", "Setup previews", "Metric guide", "Adoption mode info",
    ])
    with page_tab1:
        _render_current_run(latest_run_dir)
        _render_history()
    with page_tab2:
        _render_saved_runs_panel()
    with page_tab3:
        _render_experimenter(params, n_steps, selected_modes)
    with page_tab4:
        _render_sensitivity(params)
    with page_tab5:
        _render_ofat_explorer()
    with page_tab6:
        _render_setup_previews(params)
    with page_tab7:
        _render_metric_guide()
    with page_tab8:
        _render_mode_info()


main()
