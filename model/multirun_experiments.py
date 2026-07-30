"""
multirun_experiments.py - "extended analysis" for the Experimenter tab.

Runs every queued experiment configuration N times (one run per seed, common
random numbers) and reports, for each configuration:

  * the mean trajectory of each metric with a shaded +/- 1 standard-deviation
    ribbon across the N seeds (the figure the thesis shows in Chapter 6);
  * a final-step summary table of the primary outcome variables as mean +/- sd;
  * a complete results export (per-run rows + summary) as an Excel workbook;
  * a paired permutation test between any two configurations that demonstrates
    whether their *whole trajectories* differ, not just their final tick
    (see trajectory_stats.py).

The compute functions take no Streamlit dependency so they can be unit-tested
head-less; only render_multirun_analysis() imports streamlit, and it does so
lazily inside the function.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
import zipfile
import json
from pathlib import Path
import warnings
from typing import Callable, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from base_parameters import BASE_PARAMS
from labour_market_model import LabourMarketModel
from dashboard_utils import (
    ensure_derived_metrics,
    MODES,
    MODE_LABELS,
    METRIC_LABELS,
    PLOT_METRICS,
    KEY_METRICS,
    CUMULATIVE_CHANNEL_METRICS,
)
import trajectory_stats as tstats

# Primary outcome variables of the thesis (section 5.5), in reporting order.
PRIMARY_METRICS = ["gini_income", "skill_wage_premium", "unemployment_rate"]
SUMMARY_METRICS = PRIMARY_METRICS + ["ai_adoption_rate"]

# Curated metric set for diagnosing WHY a protected economy plateaus lower at the
# end of the horizon. The diagnostic runs showed the late divergence is NOT the
# severance/tenure brake (tenure stabilises and severance-per-task falls), but the
# Keynesian expansion gate: protection-off automates slightly more early, which
# through the demand feedback closes the profitability gate and suppresses
# re-hiring. This set therefore traces the whole demand/profitability chain behind
# the gate (price, demand, output, wage bill, profit income, demand contributions,
# labour share, unit labour cost, per-producer profit and labour cost, the
# human-vs-AI crossover), alongside the primary outcomes and the automation
# channels that feed it, and keeps the contract/tenure diagnostics and the AI
# rental cost as controls. Used by the "Emp.-protection set" preselect button next
# to the Metrics multiselect. Only keys present in _METRIC_OPTIONS are kept.
EMP_PROTECTION_ANALYSIS_METRICS = [
    # Primary outcomes
    "gini_income",
    "skill_wage_premium",
    "unemployment_rate",
    "employment_rate_high",
    "employment_rate_low",
    "ai_adoption_rate",
    # Automation channels that feed the demand contraction
    "new_proactive_automations_this_step",
    "new_reactive_automations_this_step",
    "new_displaced_workers_this_step",
    # The Keynesian expansion gate and the demand/profitability chain behind it
    "n_gate_blocks",
    "price",
    "A_demand",
    "total_output",
    "wage_bill",
    "wage_bill_lagged",
    "profit_income_lagged",
    "wage_demand_contribution",
    "profit_demand_contribution",
    "labour_share",
    "ULC",
    "Avg profit per producer",
    "Avg labour cost per producer",
    "n_human_cheaper_tasks",
    # Wage drivers of average cost
    "wage_high",
    "wage_low",
    # Contract/tenure diagnostics, kept as controls (ruled out as the late driver)
    "avg_severance_per_task",
    "share_vast_high",
    "share_vast_low",
    "avg_tenure_vast_years",
    "conversions_this_step",
    # AI price control (confirm it is at its floor and identical in both arms)
    "k_ai_current",
]

_PALETTE = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors[10:])
_LINESTYLES = ["-", "--", ":", "-."]
_PCT_METRICS = ("ai_adoption_rate", "employment_rate_high", "employment_rate_low",
                "labour_share", "unemployment_rate")
_METRIC_LABEL_OVERRIDES = {
    "unemployment_rate": "Unemployment rate",
}
_METRIC_UNITS = {
    "ai_adoption_rate": "%",
    "employment_rate_high": "%",
    "employment_rate_low": "%",
    "labour_share": "%",
    "unemployment_rate": "%",
    "gini_income": "0-1",
    "skill_wage_premium": "ratio",
    "avg_tenure_years": "years",
    "avg_tenure_vast_years": "years",
    "avg_tenure_flex_years": "years",
}


def _mode_label(mode: str) -> str:
    """Extended-analysis label; ULC is not intrinsically the baseline."""
    return "ULC" if mode == "ulc" else MODE_LABELS.get(mode, mode)


def _mode_key(label: str) -> str:
    """Recover an internal mode key from a saved display label where possible."""
    raw = str(label).strip()
    if raw in MODES:
        return raw
    clean = raw.replace(" (baseline)", "").replace("(baseline)", "").strip().lower()
    lookup = {"ulc": "ulc"}
    for mode in MODES:
        lookup[str(mode).lower()] = mode
        lookup[_mode_label(mode).lower()] = mode
        lookup[MODE_LABELS.get(mode, mode).replace(" (baseline)", "").replace("(baseline)", "").strip().lower()] = mode
    return lookup.get(clean, clean.replace(" ", "_"))


def _experiment_label(item: dict, mode: Optional[str] = None) -> str:
    label = str(item["name"])
    if mode == "ulc":
        label = label.replace(" (baseline)", "").replace("(baseline)", "").strip()
    return label


def _metric_label(metric: str) -> str:
    return _METRIC_LABEL_OVERRIDES.get(metric, METRIC_LABELS.get(metric, metric))


def _metric_axis_label(metric: str, cumulative: bool = False, diff: bool = False) -> str:
    label = _metric_label(metric)
    unit = _METRIC_UNITS.get(metric)
    if diff and metric in _PCT_METRICS:
        return f"Difference in {label} (percentage points)"
    if unit:
        label = f"{label} ({unit})"
    if cumulative:
        label = f"Cumulative {label}"
    if diff:
        label = f"Difference in {label}"
    return label


def _set_step_xlim(ax, x: np.ndarray) -> None:
    if x.size:
        ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))


def _drop_excel_index_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed:")]]


def _safe_name(name: str, fallback: str = "setup") -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name)).strip("_")
    return safe or fallback


def _configs_to_dataframe(configs: Optional[Sequence[dict]]) -> pd.DataFrame:
    rows = []
    for cfg in configs or []:
        rows.append({
            "name": cfg.get("name", "Experiment"),
            "n_steps": int(cfg.get("n_steps", 0)),
            "modes_json": json.dumps(list(cfg.get("modes", [])), default=str),
            "params_json": json.dumps(cfg.get("params", {}), default=str),
            "overrides_json": json.dumps(cfg.get("overrides", {}), default=str),
        })
    return pd.DataFrame(rows)


def _configs_from_dataframe(df: pd.DataFrame) -> list[dict]:
    configs = []
    if df is None or df.empty:
        return configs
    df = _drop_excel_index_cols(df)
    for _, row in df.iterrows():
        try:
            modes = json.loads(row.get("modes_json", "[]"))
        except Exception:
            modes = []
        try:
            params = json.loads(row.get("params_json", "{}"))
        except Exception:
            params = {}
        try:
            overrides = json.loads(row.get("overrides_json", "{}"))
        except Exception:
            overrides = {}
        configs.append({
            "name": str(row.get("name", "Experiment")),
            "params": {**BASE_PARAMS, **params},
            "modes": modes or MODES,
            "n_steps": int(row.get("n_steps", 0) or 0),
            "overrides": overrides,
        })
    return configs


def _derive_configs_from_results(results: Sequence[dict], sidebar_params: Optional[dict]) -> list[dict]:
    """Fallback config skeleton for old workbooks without saved parameters."""
    base = {**BASE_PARAMS, **(sidebar_params or {})}
    configs = []
    for item in results:
        modes = list(item.get("modes", []) or MODES)
        n_steps = int(item.get("n_steps", 0) or 0)
        configs.append({
            "name": str(item.get("name", "Experiment")),
            "params": dict(base),
            "modes": modes,
            "n_steps": n_steps,
            "overrides": {},
        })
    return configs


def save_multirun_rerun_setup(configs: Sequence[dict], base_dir: Path | str,
                              name: str, source: str = "") -> Path:
    """Save config-only Experimenter batch so it appears under Load saved batch."""
    root = Path(base_dir) / "dashboard_experimenter"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_dir = root / f"{_safe_name(name, 'loaded_multirun')}_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "batch_name": name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "configs": list(configs),
        "fingerprints": {},
        "config_only": True,
        "source": source,
    }
    (batch_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return batch_dir


# ---------------------------------------------------------------------------
# Derived metric: unemployment rate (not collected directly by the model)
# ---------------------------------------------------------------------------
def add_unemployment_rate(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Add the employment-weighted unemployment rate as a time series.

    Definition
    ----------
    A worker is *unemployed* when they are not assigned to any task
    (Worker.employed == False); there is no job-search or participation
    margin, so the labour force is the full, fixed worker population
    (n_high_skilled + n_low_skilled = 450 in the baseline).  The model
    collects the per-skill employment rates each step
    (labour_market_model._compute_all_stats):

        employment_rate_high = L_h / n_high      # employed high-skilled share
        employment_rate_low  = L_l / n_low       # employed low-skilled share

    The aggregate unemployment rate reported in the thesis is the
    population-weighted complement of those two rates:

        u = 1 - (n_high*emp_high + n_low*emp_low) / (n_high + n_low)

    i.e. the share of ALL workers not currently matched to a task.  Because
    the baseline has 450 workers but only 400 tasks (20 firms x 20 tasks),
    u has a mechanical floor of roughly 11%: the warm start fills every task,
    so u(t=0) ~ 50/450, and any rise above that floor over the horizon is
    driven by the endogenous mechanisms (automation, separations, demand
    contraction), not by the initialisation.

    Matches analysis/experiment_batch.py.
    """
    if "unemployment_rate" in df.columns:
        return df
    if {"employment_rate_high", "employment_rate_low"}.issubset(df.columns):
        n_high = float(params.get("n_high_skilled", BASE_PARAMS.get("n_high_skilled", 1)))
        n_low = float(params.get("n_low_skilled", BASE_PARAMS.get("n_low_skilled", 1)))
        tot = max(n_high + n_low, 1e-9)
        eh = pd.to_numeric(df["employment_rate_high"], errors="coerce")
        el = pd.to_numeric(df["employment_rate_low"], errors="coerce")
        df = df.copy()
        df["unemployment_rate"] = 1.0 - (n_high * eh + n_low * el) / tot
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _aggregate(dfs: Sequence[pd.DataFrame]):
    """Mean and sample-sd (ddof=1) across a list of equally-indexed DataFrames."""
    if not dfs:
        return pd.DataFrame(), pd.DataFrame()
    t_min = min(len(d) for d in dfs)
    common = [c for c in dfs[0].columns if all(c in d.columns for d in dfs)]
    num = [c for c in common if pd.api.types.is_numeric_dtype(dfs[0][c])]
    cube = np.stack([d[num].iloc[:t_min].to_numpy(dtype=float) for d in dfs], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_arr = np.nanmean(cube, axis=0)
        std_arr = (np.nanstd(cube, axis=0, ddof=1) if cube.shape[0] > 1
                   else np.zeros((t_min, len(num))))
    mean = pd.DataFrame(mean_arr, columns=num, index=range(t_min))
    std = pd.DataFrame(std_arr, columns=num, index=range(t_min))
    return mean, std


def run_config_multirun(cfg: dict, seeds: Sequence[int],
                        modes: Optional[Sequence[str]] = None,
                        progress_callback: Optional[Callable] = None,
                        log_decisions: bool = False,
                        _completed: list = None, _total: int = 1) -> dict:
    """Run a single configuration once per seed and aggregate over seeds."""
    name = cfg["name"]
    n_steps = int(cfg["n_steps"])
    use_modes = list(modes) if modes else list(cfg.get("modes") or MODES)
    resolved = {**BASE_PARAMS, **cfg["params"]}
    runs: dict[str, list[pd.DataFrame]] = {m: [] for m in use_modes}
    counter = _completed if _completed is not None else [0]

    for mode in use_modes:
        for seed in seeds:
            mp = {**resolved, "adoption_mode": mode, "seed": int(seed),
                  "log_decisions": log_decisions}
            model = LabourMarketModel(**mp)
            for _ in range(n_steps):
                model.step()
                counter[0] += 1
                if progress_callback:
                    progress_callback(f"{name} / {_mode_label(mode)} / seed {seed}",
                                      counter[0], _total)
            df = ensure_derived_metrics(model.datacollector.get_model_vars_dataframe())
            df = add_unemployment_rate(df, resolved)
            runs[mode].append(df.reset_index(drop=True))

    mean = {m: _aggregate(runs[m])[0] for m in use_modes}
    std = {m: _aggregate(runs[m])[1] for m in use_modes}
    return {"name": name, "modes": use_modes, "n_steps": n_steps,
            "seeds": list(seeds), "runs": runs, "mean": mean, "std": std,
            "params": resolved}


def run_experiment_multirun_batch(configs: Sequence[dict], n_runs: int,
                                  base_seed: int = 42,
                                  modes: Optional[Sequence[str]] = None,
                                  progress_callback: Optional[Callable] = None,
                                  log_decisions: bool = False) -> list[dict]:
    """Run every configuration N times (seeds base_seed .. base_seed+N-1)."""
    seeds = list(range(base_seed, base_seed + int(n_runs)))
    total = 0
    for cfg in configs:
        m = list(modes) if modes else list(cfg.get("modes") or MODES)
        total += len(m) * len(seeds) * int(cfg["n_steps"])
    counter = [0]
    out = []
    for cfg in configs:
        out.append(run_config_multirun(
            cfg, seeds, modes=modes, progress_callback=progress_callback,
            log_decisions=log_decisions, _completed=counter, _total=max(total, 1)))
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _series_for(item: dict, mode: str, metric: str):
    """Return (mean_series, sd_series) for a metric/mode, or (None, None)."""
    mean = item["mean"].get(mode)
    std = item["std"].get(mode)
    if mean is None or metric not in mean.columns:
        return None, None
    return mean[metric], (std[metric] if std is not None and metric in std.columns else None)


def build_multirun_figure(multi_results: Sequence[dict], metric: str, mode: str,
                          n_sd: float = 1.0, rolling: int = 1) -> "plt.Figure":
    """Mean trajectory + shaded +/- n_sd ribbon, one band per configuration."""
    fig, ax = plt.subplots(figsize=(12, 5))
    n_plotted = 0
    x_bounds = []
    cumulative = metric in CUMULATIVE_CHANNEL_METRICS
    for idx, item in enumerate(multi_results):
        mean_s, sd_s = _series_for(item, mode, metric)
        if mean_s is None or mean_s.isna().all():
            continue
        x = np.asarray(mean_s.index)
        y = pd.to_numeric(mean_s, errors="coerce").to_numpy(dtype=float)
        sd = (pd.to_numeric(sd_s, errors="coerce").to_numpy(dtype=float)
              if sd_s is not None else np.zeros_like(y))
        if cumulative:
            y = np.nancumsum(np.nan_to_num(y))
            sd = np.zeros_like(y)
        elif rolling > 1:
            y = pd.Series(y).rolling(rolling, min_periods=1, center=True).mean().to_numpy()
            sd = pd.Series(sd).rolling(rolling, min_periods=1, center=True).mean().to_numpy()
        color = mcolors.to_hex(_PALETTE[idx % len(_PALETTE)])
        ls = _LINESTYLES[idx % len(_LINESTYLES)]
        ax.plot(x, y, label=_experiment_label(item, mode), color=color,
                linestyle=ls, linewidth=2.0)
        if not cumulative and n_sd > 0:
            ax.fill_between(x, y - n_sd * sd, y + n_sd * sd, color=color, alpha=0.18, linewidth=0)
        n_plotted += 1
        x_bounds.append((float(np.nanmin(x)), float(np.nanmax(x))))

    label = _metric_label(metric)
    mode_label = _mode_label(mode)
    band_txt = f"  (mean +/- {n_sd:g} sd)" if n_sd > 0 else "  (mean)"
    ax.set_title(f"{'Cumulative ' if cumulative else ''}{label} - {mode_label}{'' if cumulative else band_txt}",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel(_metric_axis_label(metric, cumulative=cumulative), fontsize=9)
    ax.set_xlabel("Model step", fontsize=9)
    if x_bounds:
        ax.set_xlim(min(b[0] for b in x_bounds), max(b[1] for b in x_bounds))
    ax.grid(True, linestyle="--", alpha=0.5)
    if metric in _PCT_METRICS:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    if n_plotted == 0:
        ax.text(0.5, 0.5, "No data for this metric in this mode",
                ha="center", va="center", transform=ax.transAxes, color="#94a3b8")
    else:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def _iter_result_modes(multi_results: Sequence[dict]):
    """Yield every available (result, mode) pair in queue order."""
    for item in multi_results:
        for mode in item.get("modes", []):
            mean = item.get("mean", {}).get(mode)
            has_mean = mean is not None and not mean.empty
            if item.get("runs", {}).get(mode) or has_mean:
                yield item, mode


def _result_mode_label(item: dict, mode: str) -> str:
    return f"{_experiment_label(item, mode)} / {_mode_label(mode)}"


def _result_legend_label(item: dict, mode: str) -> str:
    return _experiment_label(item, mode)


def build_multirun_figure_all(multi_results: Sequence[dict], metric: str,
                              n_sd: float = 1.0, rolling: int = 1) -> "plt.Figure":
    """Mean trajectory + shaded +/- n_sd ribbon for every queued config/mode pair."""
    fig, ax = plt.subplots(figsize=(12, 5))
    n_plotted = 0
    x_bounds = []
    cumulative = metric in CUMULATIVE_CHANNEL_METRICS
    for idx, (item, mode) in enumerate(_iter_result_modes(multi_results)):
        mean_s, sd_s = _series_for(item, mode, metric)
        if mean_s is None or mean_s.isna().all():
            continue
        x = np.asarray(mean_s.index)
        y = pd.to_numeric(mean_s, errors="coerce").to_numpy(dtype=float)
        sd = (pd.to_numeric(sd_s, errors="coerce").to_numpy(dtype=float)
              if sd_s is not None else np.zeros_like(y))
        if cumulative:
            y = np.nancumsum(np.nan_to_num(y))
            sd = np.zeros_like(y)
        elif rolling > 1:
            y = pd.Series(y).rolling(rolling, min_periods=1, center=True).mean().to_numpy()
            sd = pd.Series(sd).rolling(rolling, min_periods=1, center=True).mean().to_numpy()
        color = mcolors.to_hex(_PALETTE[idx % len(_PALETTE)])
        ls = _LINESTYLES[idx % len(_LINESTYLES)]
        ax.plot(x, y, label=_result_legend_label(item, mode),
                color=color, linestyle=ls, linewidth=2.0)
        if not cumulative and n_sd > 0:
            ax.fill_between(x, y - n_sd * sd, y + n_sd * sd, color=color, alpha=0.16, linewidth=0)
        n_plotted += 1
        x_bounds.append((float(np.nanmin(x)), float(np.nanmax(x))))

    label = _metric_label(metric)
    band_txt = f"  (mean +/- {n_sd:g} sd)" if n_sd > 0 else "  (mean)"
    ax.set_title(f"{'Cumulative ' if cumulative else ''}{label}{'' if cumulative else band_txt}",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel(_metric_axis_label(metric, cumulative=cumulative), fontsize=9)
    ax.set_xlabel("Model step", fontsize=9)
    if x_bounds:
        ax.set_xlim(min(b[0] for b in x_bounds), max(b[1] for b in x_bounds))
    ax.grid(True, linestyle="--", alpha=0.5)
    if metric in _PCT_METRICS:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    if n_plotted == 0:
        ax.text(0.5, 0.5, "No data for this metric",
                ha="center", va="center", transform=ax.transAxes, color="#94a3b8")
    else:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def build_primary_outcomes_grid_figure(multi_results: Sequence[dict],
                                       n_sd: float = 1.0,
                                       rolling: int = 1) -> "plt.Figure":
    """Three-column figure with the primary thesis outcomes."""
    fig, axes = plt.subplots(1, len(PRIMARY_METRICS), figsize=(15.5, 3.9), sharex=True)
    if len(PRIMARY_METRICS) == 1:
        axes = [axes]

    any_plotted = False
    for ax, metric in zip(axes, PRIMARY_METRICS):
        n_plotted = 0
        x_bounds = []
        cumulative = metric in CUMULATIVE_CHANNEL_METRICS
        for idx, (item, mode) in enumerate(_iter_result_modes(multi_results)):
            mean_s, sd_s = _series_for(item, mode, metric)
            if mean_s is None or mean_s.isna().all():
                continue
            x = np.asarray(mean_s.index)
            y = pd.to_numeric(mean_s, errors="coerce").to_numpy(dtype=float)
            sd = (pd.to_numeric(sd_s, errors="coerce").to_numpy(dtype=float)
                  if sd_s is not None else np.zeros_like(y))
            if cumulative:
                y = np.nancumsum(np.nan_to_num(y))
                sd = np.zeros_like(y)
            elif rolling > 1:
                y = pd.Series(y).rolling(rolling, min_periods=1, center=True).mean().to_numpy()
                sd = pd.Series(sd).rolling(rolling, min_periods=1, center=True).mean().to_numpy()
            color = mcolors.to_hex(_PALETTE[idx % len(_PALETTE)])
            ls = _LINESTYLES[idx % len(_LINESTYLES)]
            ax.plot(x, y, label=_result_legend_label(item, mode),
                    color=color, linestyle=ls, linewidth=2.0)
            if not cumulative and n_sd > 0:
                ax.fill_between(x, y - n_sd * sd, y + n_sd * sd, color=color, alpha=0.16, linewidth=0)
            n_plotted += 1
            any_plotted = True
            x_bounds.append((float(np.nanmin(x)), float(np.nanmax(x))))

        ax.set_title(_metric_label(metric), fontsize=11, fontweight="bold")
        ax.set_ylabel(_metric_axis_label(metric, cumulative=cumulative), fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        if metric in _PCT_METRICS:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        if x_bounds:
            ax.set_xlim(min(b[0] for b in x_bounds), max(b[1] for b in x_bounds))
        if n_plotted == 0:
            ax.text(0.5, 0.5, "No data for this metric",
                    ha="center", va="center", transform=ax.transAxes, color="#94a3b8")

    for ax in axes:
        ax.set_xlabel("Model step", fontsize=9)
    if any_plotted:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
                       fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06 if any_plotted else 0, 1, 1))
    return fig


def build_difference_band_figure(cmp: "tstats.TrajectoryComparison",
                                 label_a: str, label_b: str, metric: str) -> "plt.Figure":
    """Two-panel figure: (top) the two mean trajectories; (bottom) the mean
    difference with its simultaneous confidence band and the significant region."""
    label = _metric_label(metric)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 2]})
    x = cmp.steps
    if not cmp.valid:
        ax1.text(0.5, 0.5, cmp.note or "No data", ha="center", va="center",
                 transform=ax1.transAxes, color="#94a3b8")
        return fig

    ax1.plot(x, cmp.mean_a, color="#1f77b4", lw=2.0, label=label_a)
    ax1.fill_between(x, cmp.mean_a - cmp.sd_a, cmp.mean_a + cmp.sd_a, color="#1f77b4", alpha=0.15)
    ax1.plot(x, cmp.mean_b, color="#d62728", lw=2.0, label=label_b)
    ax1.fill_between(x, cmp.mean_b - cmp.sd_b, cmp.mean_b + cmp.sd_b, color="#d62728", alpha=0.15)
    ax1.set_ylabel(_metric_axis_label(metric), fontsize=9)
    ax1.set_title(f"{label}: {label_a} vs {label_b}  (mean +/- 1 sd over {cmp.n_seeds} seeds)",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.axhline(0, color="#444", lw=1.0)
    ax2.plot(x, cmp.mean_diff, color="#2ca02c", lw=2.0, label="mean difference (A - B)")
    ax2.fill_between(x, cmp.band_low, cmp.band_high, color="#2ca02c", alpha=0.20,
                     label=f"{int((1-cmp.alpha)*100)}% simultaneous band")
    # Shade the steps where the band excludes zero (significant difference).
    if np.any(cmp.sig_mask):
        ymin, ymax = ax2.get_ylim()
        ax2.fill_between(x, ymin, ymax, where=cmp.sig_mask, color="#ffd166", alpha=0.30,
                         step="mid", label="significant (band excludes 0)")
        ax2.set_ylim(ymin, ymax)
    ax2.set_ylabel(_metric_axis_label(metric, diff=True), fontsize=9)
    ax2.set_xlabel("Model step", fontsize=9)
    _set_step_xlim(ax2, x)
    ptxt = "p < 0.001" if cmp.p_value < 0.001 else f"p = {cmp.p_value:.3f}"
    ax2.set_title(f"Paired permutation test: {ptxt}   |   sig. on {cmp.frac_sig:.0%} of horizon"
                  f"   |   d_z = {cmp.cohens_dz:.2f}", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(True, linestyle="--", alpha=0.5)
    if metric in _PCT_METRICS:
        pct_fmt = plt.FuncFormatter(lambda v, _: f"{v:.0%}")
        ax1.yaxis.set_major_formatter(pct_fmt)
        ax2.yaxis.set_major_formatter(pct_fmt)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tables and export
# ---------------------------------------------------------------------------
def _final_per_seed(item: dict, mode: str, metric: str) -> np.ndarray:
    """Final-step value of a metric for each seed (1-D array)."""
    vals = []
    for df in item["runs"].get(mode, []):
        if metric in df.columns and len(df):
            vals.append(float(pd.to_numeric(df[metric], errors="coerce").iloc[-1]))
    return np.asarray(vals, dtype=float)


def build_multirun_final_table(multi_results: Sequence[dict], mode: str,
                               metrics: Sequence[str] = SUMMARY_METRICS) -> pd.DataFrame:
    """One row per experiment; each cell is 'mean +/- sd' of the final-step value."""
    rows = []
    for item in multi_results:
        row = {"experiment": _experiment_label(item, mode),
               "n_runs": len(item["runs"].get(mode, []))}
        for m in metrics:
            v = _final_per_seed(item, mode, m)
            if v.size and np.isfinite(v).any():
                mu = np.nanmean(v)
                sd = np.nanstd(v, ddof=1) if v.size > 1 else 0.0
                row[_metric_axis_label(m)] = f"{mu:.4g} +/- {sd:.2g}"
            else:
                row[_metric_axis_label(m)] = "n/a"
        rows.append(row)
    return pd.DataFrame(rows).set_index("experiment")


def build_multirun_final_table_all(multi_results: Sequence[dict],
                                   metrics: Sequence[str] = SUMMARY_METRICS) -> pd.DataFrame:
    """One row per queued config/mode pair; cells are final-step mean +/- sd."""
    rows = []
    for item, mode in _iter_result_modes(multi_results):
        row = {
            "experiment": _experiment_label(item, mode),
            "mode": _mode_label(mode),
            "n_runs": len(item["runs"].get(mode, [])),
        }
        for m in metrics:
            v = _final_per_seed(item, mode, m)
            if v.size and np.isfinite(v).any():
                mu = np.nanmean(v)
                sd = np.nanstd(v, ddof=1) if v.size > 1 else 0.0
                row[_metric_axis_label(m)] = f"{mu:.4g} +/- {sd:.2g}"
            else:
                row[_metric_axis_label(m)] = "n/a"
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(["experiment", "mode"])


def build_perrun_final_dataframe(multi_results: Sequence[dict], mode: str,
                                 metrics: Sequence[str] = SUMMARY_METRICS) -> pd.DataFrame:
    """Tidy table: one row per (experiment, seed) with final-step metric values."""
    rows = []
    for item in multi_results:
        seeds = item["seeds"]
        for i, df in enumerate(item["runs"].get(mode, [])):
            row = {"experiment": _experiment_label(item, mode), "mode": _mode_label(mode),
                   "seed": seeds[i] if i < len(seeds) else i}
            for m in metrics:
                row[m] = (float(pd.to_numeric(df[m], errors="coerce").iloc[-1])
                          if m in df.columns and len(df) else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def build_perrun_final_dataframe_all(multi_results: Sequence[dict],
                                     metrics: Sequence[str] = SUMMARY_METRICS) -> pd.DataFrame:
    frames = [
        build_perrun_final_dataframe([item], mode, metrics)
        for item, mode in _iter_result_modes(multi_results)
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_perrun_timeseries_dataframe(multi_results: Sequence[dict],
                                      metrics: Sequence[str]) -> pd.DataFrame:
    """Wide table: one row per experiment/mode/seed/step for reloadable exports."""
    frames = []
    for item, mode in _iter_result_modes(multi_results):
        seeds = item.get("seeds", [])
        for i, df in enumerate(item.get("runs", {}).get(mode, [])):
            keep = [m for m in metrics if m in df.columns]
            if not keep:
                continue
            frame = df[keep].reset_index(drop=True).copy()
            frame.insert(0, "step", np.arange(len(frame)))
            frame.insert(0, "seed", seeds[i] if i < len(seeds) else i)
            frame.insert(0, "mode", _mode_label(mode))
            frame.insert(0, "mode_key", mode)
            frame.insert(0, "experiment", _experiment_label(item, mode))
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_summary_numeric_dataframe(multi_results: Sequence[dict], mode: str,
                                    metrics: Sequence[str] = SUMMARY_METRICS) -> pd.DataFrame:
    """Numeric mean/sd of the final-step value per experiment (for plotting/export)."""
    rows = []
    for item in multi_results:
        row = {"experiment": _experiment_label(item, mode), "mode": _mode_label(mode),
               "n_runs": len(item["runs"].get(mode, []))}
        for m in metrics:
            v = _final_per_seed(item, mode, m)
            row[f"{m}_mean"] = float(np.nanmean(v)) if v.size else np.nan
            row[f"{m}_sd"] = float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary_numeric_dataframe_all(multi_results: Sequence[dict],
                                        metrics: Sequence[str] = SUMMARY_METRICS) -> pd.DataFrame:
    frames = [
        build_summary_numeric_dataframe([item], mode, metrics)
        for item, mode in _iter_result_modes(multi_results)
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_mean_timeseries_dataframe(multi_results: Sequence[dict], mode: str,
                                    metrics: Sequence[str]) -> pd.DataFrame:
    """Long mean+/-sd time series per experiment for the selected metrics."""
    frames = []
    for item in multi_results:
        mean = item["mean"].get(mode)
        std = item["std"].get(mode)
        if mean is None:
            continue
        for m in metrics:
            if m not in mean.columns:
                continue
            frames.append(pd.DataFrame({
                "experiment": _experiment_label(item, mode), "mode": _mode_label(mode), "step": mean.index,
                "metric": m, "mean": mean[m].to_numpy(),
                "sd": (std[m].to_numpy() if std is not None and m in std.columns else np.nan),
            }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_mean_timeseries_dataframe_all(multi_results: Sequence[dict],
                                        metrics: Sequence[str]) -> pd.DataFrame:
    frames = [
        build_mean_timeseries_dataframe([item], mode, metrics)
        for item, mode in _iter_result_modes(multi_results)
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_multirun_excel_bytes(multi_results: Sequence[dict], mode: str,
                               metrics: Sequence[str] = SUMMARY_METRICS,
                               ts_metrics: Optional[Sequence[str]] = None,
                               test_rows: Optional[list[dict]] = None,
                               configs: Optional[Sequence[dict]] = None) -> bytes:
    """Excel workbook with per-run, summary, mean time-series and (optional) test sheets."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        build_multirun_final_table(multi_results, mode, metrics).to_excel(xl, sheet_name="summary_mean_sd")
        build_summary_numeric_dataframe(multi_results, mode, metrics).to_excel(
            xl, sheet_name="summary_numeric", index=False)
        build_perrun_final_dataframe(multi_results, mode, metrics).to_excel(
            xl, sheet_name="per_run_final", index=False)
        ts = build_mean_timeseries_dataframe(multi_results, mode, ts_metrics or metrics)
        if not ts.empty:
            ts.to_excel(xl, sheet_name="mean_timeseries", index=False)
        per_ts = build_perrun_timeseries_dataframe(multi_results, ts_metrics or metrics)
        if not per_ts.empty:
            per_ts.to_excel(xl, sheet_name="per_run_timeseries", index=False)
        cfg_df = _configs_to_dataframe(configs)
        if not cfg_df.empty:
            cfg_df.to_excel(xl, sheet_name="configs", index=False)
        if test_rows:
            pd.DataFrame(test_rows).to_excel(xl, sheet_name="statistical_tests", index=False)
    return buf.getvalue()


def build_multirun_excel_bytes_all(multi_results: Sequence[dict],
                                   metrics: Sequence[str] = SUMMARY_METRICS,
                                   ts_metrics: Optional[Sequence[str]] = None,
                                   test_rows: Optional[list[dict]] = None,
                                   configs: Optional[Sequence[dict]] = None) -> bytes:
    """Excel workbook with every queued config/mode pair."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        build_multirun_final_table_all(multi_results, metrics).to_excel(xl, sheet_name="summary_mean_sd")
        build_summary_numeric_dataframe_all(multi_results, metrics).to_excel(
            xl, sheet_name="summary_numeric", index=False)
        build_perrun_final_dataframe_all(multi_results, metrics).to_excel(
            xl, sheet_name="per_run_final", index=False)
        ts = build_mean_timeseries_dataframe_all(multi_results, ts_metrics or metrics)
        if not ts.empty:
            ts.to_excel(xl, sheet_name="mean_timeseries", index=False)
        per_ts = build_perrun_timeseries_dataframe(multi_results, ts_metrics or metrics)
        if not per_ts.empty:
            per_ts.to_excel(xl, sheet_name="per_run_timeseries", index=False)
        cfg_df = _configs_to_dataframe(configs)
        if not cfg_df.empty:
            cfg_df.to_excel(xl, sheet_name="configs", index=False)
        if test_rows:
            pd.DataFrame(test_rows).to_excel(xl, sheet_name="statistical_tests", index=False)
    return buf.getvalue()


def load_multirun_excel_bytes(data: bytes) -> tuple[list[dict], dict]:
    """Load a saved extended-analysis workbook.

    Workbooks exported after this loader was added include `per_run_timeseries`,
    which restores full per-seed trajectories and enables permutation tests.
    Older workbooks fall back to `mean_timeseries`, which is enough for plots
    and saved summary tables but not valid for paired tests.
    """
    xl = pd.ExcelFile(BytesIO(data))
    info = {
        "has_raw_runs": "per_run_timeseries" in xl.sheet_names,
        "summary_table": None,
        "configs": [],
        "has_configs": "configs" in xl.sheet_names,
        "message": "",
    }
    if "summary_mean_sd" in xl.sheet_names:
        info["summary_table"] = _drop_excel_index_cols(pd.read_excel(xl, "summary_mean_sd"))
    if info["has_configs"]:
        info["configs"] = _configs_from_dataframe(pd.read_excel(xl, "configs"))

    items: dict[str, dict] = {}

    def get_item(name: str) -> dict:
        if name not in items:
            items[name] = {
                "name": name,
                "modes": [],
                "n_steps": 0,
                "seeds": [],
                "runs": {},
                "mean": {},
                "std": {},
                "params": {},
            }
        return items[name]

    if info["has_raw_runs"]:
        raw = _drop_excel_index_cols(pd.read_excel(xl, "per_run_timeseries"))
        required = {"experiment", "mode", "seed", "step"}
        if not required.issubset(raw.columns):
            raise ValueError("The per_run_timeseries sheet is missing required columns.")
        meta_cols = {"experiment", "mode", "mode_key", "seed", "step"}
        metric_cols = [
            c for c in raw.columns
            if c not in meta_cols and pd.api.types.is_numeric_dtype(raw[c])
        ]
        if not metric_cols:
            raise ValueError("The per_run_timeseries sheet does not contain metric columns.")
        for (exp_name, mode_label), mode_df in raw.groupby(["experiment", "mode"], sort=False):
            mode_key = _mode_key(
                mode_df["mode_key"].iloc[0] if "mode_key" in mode_df.columns else mode_label
            )
            item = get_item(str(exp_name))
            if mode_key not in item["modes"]:
                item["modes"].append(mode_key)
            runs = []
            seeds = []
            for seed, seed_df in mode_df.groupby("seed", sort=True):
                ordered = seed_df.sort_values("step")
                runs.append(ordered[metric_cols].reset_index(drop=True))
                seeds.append(int(seed) if pd.notna(seed) else len(seeds))
            item["runs"][mode_key] = runs
            item["mean"][mode_key], item["std"][mode_key] = _aggregate(runs)
            step_max = int(pd.to_numeric(mode_df["step"], errors="coerce").max())
            item["n_steps"] = max(item["n_steps"], step_max)
            item["seeds"] = sorted(set(item["seeds"]).union(seeds))
        info["message"] = "Loaded full per-seed trajectories; statistical tests are available."
    elif "mean_timeseries" in xl.sheet_names:
        ts = _drop_excel_index_cols(pd.read_excel(xl, "mean_timeseries"))
        required = {"experiment", "mode", "step", "metric", "mean"}
        if not required.issubset(ts.columns):
            raise ValueError("The mean_timeseries sheet is missing required columns.")
        for (exp_name, mode_label), mode_df in ts.groupby(["experiment", "mode"], sort=False):
            mode_key = _mode_key(mode_label)
            item = get_item(str(exp_name))
            if mode_key not in item["modes"]:
                item["modes"].append(mode_key)
            mean = mode_df.pivot_table(index="step", columns="metric", values="mean", aggfunc="first").sort_index()
            if "sd" in mode_df.columns:
                std = mode_df.pivot_table(index="step", columns="metric", values="sd", aggfunc="first").sort_index()
            else:
                std = pd.DataFrame(0.0, index=mean.index, columns=mean.columns)
            item["runs"][mode_key] = []
            item["mean"][mode_key] = mean
            item["std"][mode_key] = std
            step_max = int(pd.to_numeric(mode_df["step"], errors="coerce").max())
            item["n_steps"] = max(item["n_steps"], step_max)
        info["message"] = (
            "Loaded mean trajectories only. This older workbook can be plotted, "
            "but paired permutation tests need a newer export with per_run_timeseries."
        )
    else:
        raise ValueError("This workbook does not contain per_run_timeseries or mean_timeseries.")

    return list(items.values()), info


# ---------------------------------------------------------------------------
# Statistical comparison wiring
# ---------------------------------------------------------------------------
def compare_two_config_modes(item_a: dict, mode_a: str, item_b: dict, mode_b: str,
                             metric: str, n_perm: int = 5000, n_boot: int = 5000,
                             alpha: float = 0.05, seed: int = 0) -> "tstats.TrajectoryComparison":
    """Paired trajectory comparison of two config/mode result series."""
    runs_a = [pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
              for df in item_a["runs"].get(mode_a, []) if metric in df.columns]
    runs_b = [pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
              for df in item_b["runs"].get(mode_b, []) if metric in df.columns]
    cmp = tstats.compare_trajectories(runs_a, runs_b, metric=metric,
                                      n_perm=n_perm, n_boot=n_boot, alpha=alpha, seed=seed)
    return cmp


def compare_two_configs(item_a: dict, item_b: dict, mode: str, metric: str,
                        n_perm: int = 5000, n_boot: int = 5000,
                        alpha: float = 0.05, seed: int = 0) -> "tstats.TrajectoryComparison":
    """Paired trajectory comparison of two multirun results on one metric/mode."""
    return compare_two_config_modes(item_a, mode, item_b, mode, metric,
                                    n_perm=n_perm, n_boot=n_boot, alpha=alpha, seed=seed)


def build_pairwise_test_table(result_pairs: Sequence[tuple[dict, str]],
                              metrics: Sequence[str],
                              n_perm: int = 5000,
                              alpha: float = 0.05) -> pd.DataFrame:
    """Run paired trajectory tests for every unordered pair of result series."""
    rows = []
    for i in range(len(result_pairs)):
        item_a, mode_a = result_pairs[i]
        for j in range(i + 1, len(result_pairs)):
            item_b, mode_b = result_pairs[j]
            label_a = _experiment_label(item_a, mode_a)
            label_b = _experiment_label(item_b, mode_b)
            for metric in metrics:
                if metric not in item_a["mean"].get(mode_a, pd.DataFrame()).columns:
                    continue
                if metric not in item_b["mean"].get(mode_b, pd.DataFrame()).columns:
                    continue
                cmp = compare_two_config_modes(
                    item_a, mode_a, item_b, mode_b, metric,
                    n_perm=n_perm, alpha=alpha,
                    seed=1000 + i * 101 + j * 17 + len(rows),
                )
                rows.append({
                    "A": label_a,
                    "Mode A": _mode_label(mode_a),
                    "B": label_b,
                    "Mode B": _mode_label(mode_b),
                    "Metric": _metric_axis_label(metric),
                    "n seeds": cmp.n_seeds,
                    "p-value": cmp.p_value,
                    "Significant": bool(cmp.p_value < alpha) if np.isfinite(cmp.p_value) else False,
                    "Mean diff (A-B)": cmp.mean_signed_diff,
                    "CI low": cmp.signed_diff_ci[0],
                    "CI high": cmp.signed_diff_ci[1],
                    "Effect d_z": cmp.cohens_dz,
                    "Significant horizon": cmp.frac_sig,
                    "Note": cmp.note,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Streamlit UI  (imports streamlit lazily so the compute layer stays headless)
# ---------------------------------------------------------------------------
_METRIC_OPTIONS = list(dict.fromkeys(list(PLOT_METRICS) + PRIMARY_METRICS))


def render_multirun_analysis(configs, sidebar_params: Optional[dict] = None,
                             base_dir: Optional[Path | str] = None) -> None:
    """Render the 'extended analysis' panel at the bottom of the Experimenter.

    `configs` is the queued experiment list (st.session_state['exp_configs']).
    Each config is run N times (one run per seed); the panel then shows the
    mean +/- 1 sd ribbons, a mean+/-sd final-step table, a full-results Excel
    download, and a paired permutation test between any two configurations.
    """
    import json
    import streamlit as st

    st.subheader("Extended analysis - multiple runs per experiment")
    st.caption(
        "Runs each configuration N times (one run per seed, using common random numbers) "
        "and shows the mean trajectory with a +/-1 sd band, a final-value table "
        "(mean +/- sd), a full-results download, and a paired permutation test "
        "for whether two queued result series differ across the whole time series."
    )

    with st.expander("Load previous extended-analysis Excel", expanded=False):
        uploaded = st.file_uploader(
            "Extended analysis workbook (.xlsx)",
            type=["xlsx"],
            key="mr_load_workbook",
        )
        if uploaded is not None and st.button("Load old extended analysis", key="mr_load_button"):
            try:
                loaded_results, load_info = load_multirun_excel_bytes(uploaded.getvalue())
                st.session_state["mr_cache"] = {
                    "sig": f"loaded:{uploaded.name}:{uploaded.size}",
                    "results": loaded_results,
                    "loaded": True,
                    "source": uploaded.name,
                    "summary_table": load_info.get("summary_table"),
                    "has_raw_runs": load_info.get("has_raw_runs", False),
                    "configs": load_info.get("configs", []),
                    "has_configs": load_info.get("has_configs", False),
                    "message": load_info.get("message", ""),
                }
                st.success(f"Loaded {uploaded.name}.")
            except Exception as exc:
                st.error(f"Could not load workbook: {exc}")

        loaded_for_setup = st.session_state.get("mr_cache")
        if loaded_for_setup and loaded_for_setup.get("loaded"):
            loaded_results = loaded_for_setup.get("results", [])
            loaded_configs = loaded_for_setup.get("configs") or _derive_configs_from_results(
                loaded_results, sidebar_params
            )
            if loaded_for_setup.get("has_configs"):
                st.caption("This workbook contains saved experiment configs and can be queued for an exact rerun.")
            else:
                st.warning(
                    "This older workbook does not contain parameter configs. "
                    "I can prepare a rerun queue using the loaded names/modes/steps and your current sidebar parameters. "
                    "Review the queue before rerunning."
                )
            setup_cols = st.columns(2)
            if setup_cols[0].button("Queue loaded experiments for rerun", key="mr_queue_loaded_configs"):
                st.session_state["exp_configs"] = loaded_configs
                st.session_state["exp_cache"] = {}
                st.success(f"Queued {len(loaded_configs)} experiment(s) for rerun.")
                st.rerun()
            setup_name = st.text_input(
                "Save prepared rerun setup as",
                value=f"rerun_{Path(str(loaded_for_setup.get('source', 'extended_analysis'))).stem}",
                key="mr_loaded_setup_name",
            )
            if setup_cols[1].button("Save so I can find it later", key="mr_save_loaded_setup"):
                if base_dir is None:
                    st.error("No dashboard output directory is configured for saving this setup.")
                else:
                    save_multirun_rerun_setup(
                        loaded_configs,
                        base_dir=base_dir,
                        name=setup_name.strip() or "loaded_multirun_setup",
                        source=str(loaded_for_setup.get("source", "")),
                    )
                    st.success("Saved. You can find it later under Experimenter -> Load saved batch -> Load & rerun.")

    loaded_cache = st.session_state.get("mr_cache")
    loaded_active = bool(loaded_cache and loaded_cache.get("loaded"))
    if not configs and not loaded_active:
        st.info("Add experiments to the queue above first, or load a previous extended-analysis Excel workbook.")
        return

    if configs:
        config_mode_count = sum(len(c.get("modes", []) or MODES) for c in configs)
        c1, c2, c3 = st.columns(3)
        n_runs = int(c1.number_input("Runs per experiment", min_value=2, max_value=100,
                                     value=10, step=1, key="mr_n_runs",
                                     help="Number of seeds. The thesis uses 10 (seeds 42-51)."))
        base_seed = int(c2.number_input("First seed", min_value=0, max_value=100000,
                                        value=42, step=1, key="mr_base_seed"))
        n_sd = float(c3.number_input("SD-band (x sd)", min_value=0.0, max_value=3.0,
                                     value=1.0, step=0.5, key="mr_nsd"))
    else:
        n_runs = 0
        base_seed = 42
        config_mode_count = 0
        n_sd = float(st.number_input("SD-band (x sd)", min_value=0.0, max_value=3.0,
                                     value=1.0, step=0.5, key="mr_nsd"))

    # Preselect via a flag set before the widget renders to avoid
    # StreamlitAPIException (cannot modify a widget key after instantiation).
    if st.session_state.pop("_mr_preselect_empprot_pending", False):
        st.session_state["mr_metrics"] = [
            m for m in EMP_PROTECTION_ANALYSIS_METRICS if m in _METRIC_OPTIONS
        ]
    _mcol1, _mcol2 = st.columns([6, 1.6])
    with _mcol1:
        metrics = st.multiselect("Metrics", options=_METRIC_OPTIONS,
                                 default=[m for m in PRIMARY_METRICS if m in _METRIC_OPTIONS],
                                 format_func=lambda m: _metric_axis_label(m), key="mr_metrics")
    with _mcol2:
        st.write("")  # vertical alignment with the multiselect input
        if st.button(
            "Emp.-protection set",
            key="mr_preselect_empprot",
            help=(
                "Preselect the outcomes for diagnosing WHY a protected economy "
                "plateaus lower late, i.e. why the Keynesian expansion gate diverges. "
                "Beyond the primary outcomes and the automation channels, it traces "
                "the demand/profitability chain behind the gate: gate blocks, price, "
                "demand, output, wage bill, profit income, the two demand "
                "contributions, labour share, unit labour cost, per-producer profit "
                "and labour cost, and the human-vs-AI cost crossover, plus wages. The "
                "contract/tenure diagnostics and the AI rental cost are kept as "
                "controls."
            ),
        ):
            st.session_state["_mr_preselect_empprot_pending"] = True
            st.rerun()
    if configs:
        log_dec = st.checkbox("Log investment decisions (slower)", value=False, key="mr_log")
        seeds_preview = f"{base_seed}-{base_seed + n_runs - 1}"
        total_runs = n_runs * config_mode_count
        run_clicked = st.button(
            f"Run extended analysis - {total_runs} runs ({config_mode_count} config-mode series x {n_runs} seeds {seeds_preview})",
            type="primary", use_container_width=True, key="mr_run",
        )
        sig = json.dumps({
            "cfgs": [(c["name"], tuple(c.get("modes", []) or MODES),
                      sorted((c.get("overrides", {}) or {}).items()), c["n_steps"]) for c in configs],
            "n": n_runs, "seed": base_seed, "log": log_dec,
        }, default=str, sort_keys=True)
    else:
        log_dec = False
        run_clicked = False
        sig = None

    if run_clicked:
        bar = st.progress(0.0, text="Starting...")

        def cb(label, done, total):
            bar.progress(min(done / max(total, 1), 1.0), text=f"{label}  ({done}/{total})")

        with st.spinner("Simulating..."):
            results = run_experiment_multirun_batch(
                configs, n_runs, base_seed=base_seed, modes=None,
                progress_callback=cb, log_decisions=log_dec)
        bar.progress(1.0, text="Done.")
        st.session_state["mr_cache"] = {"sig": sig, "results": results, "loaded": False, "has_raw_runs": True}

    # ---- Paired permutation test ------------------------------------------
    st.markdown("### Statistical test - do two result series differ?")
    st.caption(
        "This is a paired sign-flip permutation test on the full difference trajectory "
        "(A - B). Because both configurations use the same seeds, each seed is a matched "
        "pair. Under H0, both configurations produce the same time series, so the signs "
        "of the paired differences are exchangeable. The test repeatedly flips those "
        "signs to build a null distribution; the p-value is the share of permutations "
        "at least as extreme as the observed trajectory. More permutations give a more "
        "stable p-value and critical band, but take longer."
    )

    cache = st.session_state.get("mr_cache")
    if not cache:
        return
    if cache.get("loaded"):
        if cache.get("source"):
            st.info(f"Loaded workbook: {cache['source']}")
        if cache.get("message"):
            st.caption(cache["message"])
    elif cache.get("sig") != sig:
        if cache:
            st.warning("Settings or configurations changed - run the analysis again.")
        return
    results = cache["results"]
    result_pairs = list(_iter_result_modes(results))
    if not result_pairs:
        st.warning("No result series were produced for the queued modes.")
        return

    test_result_pairs = [
        (item, mode) for item, mode in result_pairs
        if item.get("runs", {}).get(mode)
    ]
    pair_options = list(range(len(test_result_pairs)))
    if not cache.get("has_raw_runs", True):
        st.warning(
            "This loaded workbook has only mean trajectories. Pairwise permutation tests "
            "need a newer export with the per_run_timeseries sheet."
        )
    elif len(pair_options) < 2:
        st.caption("Add at least two queued config/mode series to run the test.")
    else:
        tc1, tc2 = st.columns(2)
        a_idx = tc1.selectbox(
            "Result A",
            pair_options,
            index=0,
            format_func=lambda i: _result_mode_label(*test_result_pairs[i]),
            key="mr_result_a",
        )
        b_idx = tc2.selectbox(
            "Result B",
            pair_options,
            index=1 if len(pair_options) > 1 else 0,
            format_func=lambda i: _result_mode_label(*test_result_pairs[i]),
            key="mr_result_b",
        )
        item_a, mode_a = test_result_pairs[a_idx]
        item_b, mode_b = test_result_pairs[b_idx]
        metric_options = [
            m for m in _METRIC_OPTIONS
            if m in item_a["mean"].get(mode_a, pd.DataFrame()).columns
            and m in item_b["mean"].get(mode_b, pd.DataFrame()).columns
        ]
        tc4, tc5 = st.columns(2)
        if not metric_options:
            st.warning("The selected result series do not share any plottable metrics.")
        else:
            t_metric = tc4.selectbox("Metric", options=metric_options,
                                     format_func=lambda m: _metric_axis_label(m), key="mr_tmetric")
            alpha = float(tc5.number_input("alpha", min_value=0.01, max_value=0.20,
                                           value=0.05, step=0.01, key="mr_alpha"))
            n_perm = int(st.number_input(
                "Permutations", min_value=1000, max_value=50000,
                value=5000, step=1000, key="mr_perm",
                help=(
                    "Number of random sign-flip resamples used to approximate the null "
                    "distribution. Higher values are more stable but slower."
                ),
            ))

            if st.button("Run paired permutation test", key="mr_test", type="primary"):
                selector_label_a = _result_mode_label(item_a, mode_a)
                selector_label_b = _result_mode_label(item_b, mode_b)
                plot_label_a = _result_legend_label(item_a, mode_a)
                plot_label_b = _result_legend_label(item_b, mode_b)
                if a_idx == b_idx:
                    st.warning("Choose two different result series.")
                else:
                    cmp = compare_two_config_modes(item_a, mode_a, item_b, mode_b, t_metric,
                                                   n_perm=n_perm, alpha=alpha)
                    if not cmp.valid:
                        st.warning(cmp.note)
                    else:
                        mc = st.columns(4)
                        mc[0].metric("p-value", "< 0.001" if cmp.p_value < 0.001 else f"{cmp.p_value:.4f}")
                        mc[1].metric("Effect size d_z", f"{cmp.cohens_dz:.2f}")
                        mc[2].metric("Mean difference (A-B)", f"{cmp.mean_signed_diff:.4g}")
                        mc[3].metric("Significant horizon share", f"{cmp.frac_sig:.0%}")
                        st.caption(
                            f"95% CI mean difference: [{cmp.signed_diff_ci[0]:.4g}, {cmp.signed_diff_ci[1]:.4g}]  -  "
                            f"sup|t| = {cmp.stat_obs:.2f}, critical value = {cmp.crit_value:.2f}  -  n = {cmp.n_seeds} seeds")
                        fig = build_difference_band_figure(cmp, plot_label_a, plot_label_b, t_metric)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                    st.markdown("**All primary outcome variables (this pair)**")
                    rows = []
                    for mm in PRIMARY_METRICS:
                        c_mm = compare_two_config_modes(item_a, mode_a, item_b, mode_b, mm,
                                                        n_perm=n_perm, alpha=alpha)
                        rows.append(tstats.summary_row(c_mm, selector_label_a, selector_label_b))
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.markdown("**All pairwise comparisons**")
            st.caption(
                "Runs the same paired trajectory test for every unique pair "
                "(AB, AC, BC, etc.) in the current extended-analysis results."
            )
            pairwise_cols = st.columns(2)
            use_primary_metrics = pairwise_cols[0].checkbox(
                "Use all primary outcome variables",
                value=False,
                key="mr_pairwise_primary",
                help="If unchecked, the table uses only the selected metric above.",
            )
            pairwise_metrics = PRIMARY_METRICS if use_primary_metrics else [t_metric]
            n_pairs = len(pair_options) * (len(pair_options) - 1) // 2
            n_tests = n_pairs * len(pairwise_metrics)
            pairwise_cols[1].caption(f"{n_pairs} pairs, {n_tests} test(s)")
            if st.button("Run all pairwise tests", key="mr_pairwise_tests", use_container_width=True):
                with st.spinner("Running all pairwise tests..."):
                    pairwise_df = build_pairwise_test_table(
                        test_result_pairs,
                        pairwise_metrics,
                        n_perm=n_perm,
                        alpha=alpha,
                    )
                if pairwise_df.empty:
                    st.warning("No comparable metrics were found for the pairwise table.")
                else:
                    display_df = pairwise_df.copy()
                    display_df["p-value"] = display_df["p-value"].map(
                        lambda v: "< 0.001" if np.isfinite(v) and v < 0.001
                        else (f"{v:.4f}" if np.isfinite(v) else "n/a")
                    )
                    for col in ["Mean diff (A-B)", "CI low", "CI high"]:
                        display_df[col] = display_df[col].map(
                            lambda v: f"{v:.4g}" if np.isfinite(v) else "n/a"
                        )
                    display_df["Effect d_z"] = display_df["Effect d_z"].map(
                        lambda v: f"{v:.2f}" if np.isfinite(v) else "n/a"
                    )
                    display_df["Significant horizon"] = display_df["Significant horizon"].map(
                        lambda v: f"{v:.0%}" if np.isfinite(v) else "n/a"
                    )
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("**Final-step summary (mean +/- sd across seeds)**")
    if cache.get("loaded") and not cache.get("has_raw_runs", True) and cache.get("summary_table") is not None:
        st.dataframe(cache["summary_table"], use_container_width=True, hide_index=True)
    else:
        st.dataframe(build_multirun_final_table_all(results), use_container_width=True)

    st.markdown("**Mean trajectory with +/-1 sd band**")
    if not metrics:
        st.caption("Select at least one metric above.")
    for m in metrics:
        fig = build_multirun_figure_all(results, m, n_sd=n_sd)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("**Primary outcomes combined (1x3 grid)**")
    fig = build_primary_outcomes_grid_figure(results, n_sd=n_sd)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    export_metrics = list(dict.fromkeys(list(metrics or []) + PRIMARY_METRICS))
    export_configs = configs if configs else cache.get("configs")
    xb = build_multirun_excel_bytes_all(results, ts_metrics=export_metrics, configs=export_configs)
    st.download_button(
        "Download full results (Excel)", data=xb,
        file_name="extended_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True, key="mr_download")

    if metrics and st.button("Download all figures (.zip)", key="mr_download_zip", use_container_width=True):
        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for m in metrics:
                fig = build_multirun_figure_all(results, m, n_sd=n_sd)
                img_buf = BytesIO()
                fig.savefig(img_buf, format="png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                safe_name = m.replace(" ", "_").replace("/", "_").replace("\\", "_")
                zf.writestr(f"{safe_name}.png", img_buf.getvalue())
            fig_grid = build_primary_outcomes_grid_figure(results, n_sd=n_sd)
            img_buf = BytesIO()
            fig_grid.savefig(img_buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig_grid)
            zf.writestr("primary_outcomes_grid.png", img_buf.getvalue())
        st.download_button(
            "Click to save ZIP",
            data=zip_buf.getvalue(),
            file_name="extended_analysis_figures.zip",
            mime="application/zip",
            use_container_width=True,
            key="mr_download_zip_save",
        )
