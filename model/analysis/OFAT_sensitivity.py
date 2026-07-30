from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Allow imports from the parent model/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from dashboard_utils import BASE_PARAMS, BOOL_PARAMS, KEY_METRICS, MODES, OFAT_PARAMS
from labour_market_model import LabourMarketModel


DEFAULT_STEPS = 1000
DEFAULT_MODES = ["ulc", "npv_mean_field"]
# DEFAULT_MODES = list(MODES)
DEFAULT_WORKERS = 8
REPLICATE_SEEDS = list(range(42, 52))
GRID_FACTORS = [-0.5, -0.25, 0.0, 0.25, 0.5]
CUSTOM_PARAM_VALUES = {
    "nr_multiplier": [0.5, 0.75, 1.0, 1.25, 1.5],
}
ZERO_BASELINE_FALLBACKS = {
    "alpha_min": [0.0, 0.125, 0.25, 0.375, 0.5],
}
LOWER_BOUNDS = {
    "n_producers": 1,
    "n_high_skilled": 0,
    "n_low_skilled": 0,
    "delta": 0.0,
    "k_ai": 0.0,
    "p_evaluate": 0.0,
    "A_base": 0.0,
    "seed": 0,
    "B": 0.0,
    "lam": 0.0,
    "b_h": 0.0,
    "b_l": 0.0,
    "a_prod": 0.0,
    "xi_prod": 0.0,
    "gamma_pi": 0.0,
    "phi_base": 0.0,
    "phi_decay": 0.0,
    "nr_multiplier": 0.0,
    "mismatch_theta": 1.0,
    "I_base": 0.0,
    "complexity_scaling": 0.0,
    "r_discount": 0.0,
    "T_horizon": 1,
    "eta_hurdle": 0.0,
    "omega_adapt": 0.0,
    "k_ai_decay": 0.0,
    "k_ai_floor": 0.0,
    "steps_per_year": 1,
    "severance_rate": 0.0,
    "chain_limit": 1,
    "p_convert": 0.0,
    "init_share_vast": 0.0,
    "init_tenure_max_years": 0.0,
    "alpha_min": 0.0,
    "alpha_max": 0.0,
}
UPPER_BOUNDS = {
    "delta": 1.0,
    "p_evaluate": 1.0,
    "lam": 1.0,
    "eta_hurdle": 1.0,
    "omega_adapt": 1.0,
    "severance_rate": 1.0,
    "p_convert": 1.0,
    "init_share_vast": 1.0,
    "alpha_min": 1.0,
    "alpha_max": 1.0,
}


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))
    return cleaned.strip("_") or "item"


def _format_seconds(total_seconds: float) -> str:
    total_seconds = max(0, int(round(total_seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _build_param_grid(param_name: str, default_value):
    if isinstance(default_value, bool):
        alt_value = not default_value
        grid = [
            {
                "grid_index": 0,
                "factor_pct": None,
                "factor_label": "off" if alt_value is False else "on",
                "param_value": alt_value,
            },
            {
                "grid_index": 1,
                "factor_pct": None,
                "factor_label": "default_on" if default_value else "default_off",
                "param_value": default_value,
            },
        ]
        return grid, "boolean_toggle"

    is_int = isinstance(default_value, int) and not isinstance(default_value, bool)

    if param_name in CUSTOM_PARAM_VALUES:
        raw_values = list(CUSTOM_PARAM_VALUES[param_name])
        strategy = "custom_range"
    elif param_name in ZERO_BASELINE_FALLBACKS:
        raw_values = list(ZERO_BASELINE_FALLBACKS[param_name])
        strategy = "fallback_zero_baseline"
    elif param_name in UPPER_BOUNDS and math.isclose(float(default_value), float(UPPER_BOUNDS[param_name])):
        raw_values = np.linspace(float(default_value) * 0.5, float(default_value), 5).tolist()
        strategy = "fallback_upper_bound"
    else:
        raw_values = [float(default_value) * (1.0 + factor) for factor in GRID_FACTORS]
        strategy = "percentage"

    bounded_values = []
    lower = LOWER_BOUNDS.get(param_name)
    upper = UPPER_BOUNDS.get(param_name)
    for value in raw_values:
        if lower is not None:
            value = max(lower, value)
        if upper is not None:
            value = min(upper, value)
        if is_int:
            value = int(round(value))
        else:
            value = float(round(value, 6))
        bounded_values.append(value)

    values = bounded_values
    if param_name == "alpha_min":
        values = [min(v, float(BASE_PARAMS["alpha_max"])) for v in values]
    if param_name == "alpha_max":
        values = [max(v, float(BASE_PARAMS["alpha_min"])) for v in values]

    grid = []
    for idx, value in enumerate(values):
        if strategy == "custom_range":
            factor = (float(value) / float(default_value) - 1.0) if float(default_value) != 0 else None
            label = f"{value:g}"
        else:
            factor = GRID_FACTORS[idx] if idx < len(GRID_FACTORS) else None
            label = f"{factor:+.0%}" if factor is not None else f"grid_{idx}"
        grid.append(
            {
                "grid_index": idx,
                "factor_pct": factor,
                "factor_label": label,
                "param_value": value,
            }
        )
    return grid, strategy


def _build_tasks(params_to_run: list[str], n_steps: int, modes_to_run: list[str]):
    tasks = []
    grids = {}
    strategies = {}
    for param_name in params_to_run:
        default_value = BASE_PARAMS[param_name]
        grid, strategy = _build_param_grid(param_name, default_value)
        grids[param_name] = grid
        strategies[param_name] = strategy
        for mode in modes_to_run:
            for point in grid:
                for replicate_index, seed in enumerate(REPLICATE_SEEDS):
                    tasks.append(
                        {
                            "parameter": param_name,
                            "mode": mode,
                            "grid_index": point["grid_index"],
                            "factor_pct": point["factor_pct"],
                            "factor_label": point["factor_label"],
                            "param_value": point["param_value"],
                            "default_value": default_value,
                            "replicate_index": replicate_index,
                            "seed": seed,
                            "n_steps": n_steps,
                        }
                    )
    return tasks, grids, strategies


def _run_single_task(task: dict) -> dict:
    run_params = {**BASE_PARAMS, task["parameter"]: task["param_value"], "seed": task["seed"], "adoption_mode": task["mode"]}
    start_time = time.perf_counter()
    model = LabourMarketModel(**run_params)
    for _ in range(task["n_steps"]):
        model.step()
    runtime_seconds = time.perf_counter() - start_time
    df = model.datacollector.get_model_vars_dataframe()
    if df.empty:
        final_metrics = {}
    else:
        final_row = df.iloc[-1]
        final_metrics = {
            column: float(final_row[column])
            for column in df.columns
            if pd.api.types.is_numeric_dtype(df[column])
        }
    return {
        "task": task,
        "df": df,
        "final_metrics": final_metrics,
        "runtime_seconds": runtime_seconds,
    }


def _write_result(result: dict, timeseries_dir: Path) -> dict:
    task = result["task"]
    file_name = (
        f"{task['parameter']}__{task['mode']}__g{task['grid_index']:02d}"
        f"__r{task['replicate_index']:02d}__seed{task['seed']}.csv"
    )
    csv_path = timeseries_dir / file_name
    result["df"].to_csv(csv_path, index_label="Step")

    row = {
        "parameter": task["parameter"],
        "mode": task["mode"],
        "grid_index": task["grid_index"],
        "factor_pct": task["factor_pct"],
        "factor_label": task["factor_label"],
        "param_value": task["param_value"],
        "default_value": task["default_value"],
        "replicate_index": task["replicate_index"],
        "seed": task["seed"],
        "runtime_seconds": round(result["runtime_seconds"], 6),
        "timeseries_file": file_name,
    }
    for metric_name, value in result["final_metrics"].items():
        row[f"final__{metric_name}"] = value
    return row


def _build_aggregated(summary_df: pd.DataFrame) -> pd.DataFrame:
    final_cols = [col for col in summary_df.columns if col.startswith("final__")]
    group_cols = ["parameter", "mode", "grid_index", "factor_pct", "factor_label", "param_value", "default_value"]

    grouped = summary_df.groupby(group_cols, dropna=False)
    aggregated = grouped.agg(
        replicates=("seed", "count"),
        runtime_mean_seconds=("runtime_seconds", "mean"),
        runtime_std_seconds=("runtime_seconds", "std"),
    )

    for col in final_cols:
        metric_name = col.replace("final__", "")
        aggregated[f"final_mean__{metric_name}"] = grouped[col].mean()
        aggregated[f"final_std__{metric_name}"] = grouped[col].std()

    return aggregated.reset_index().sort_values(["parameter", "mode", "grid_index"])


def _print_progress(completed_runs: int, total_runs: int, elapsed_seconds: float, mean_runtime_seconds: float):
    ratio = completed_runs / max(total_runs, 1)
    width = 34
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    throughput = completed_runs / max(elapsed_seconds, 1e-9)
    remaining_runs = max(total_runs - completed_runs, 0)
    eta_seconds = remaining_runs / max(throughput, 1e-9)
    print(
        f"\r[{bar}] {completed_runs:>5}/{total_runs}  {ratio:>6.1%}  "
        f"elapsed { _format_seconds(elapsed_seconds) }  ETA { _format_seconds(eta_seconds) }  "
        f"time/run {mean_runtime_seconds:>6.2f}s",
        end="",
        flush=True,
    )


def _make_output_dir(base_dir: Path, batch_name: str) -> Path:
    root = base_dir / "ofat_runs"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = root / f"{_safe_name(batch_name)}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "timeseries").mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Run OFAT sensitivity for all numeric and boolean Labour Market ABM parameters.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Simulation steps per run.")
    parser.add_argument(
        "--modes",
        nargs="*",
        default=DEFAULT_MODES,
        help="Optional subset of adoption modes. Edit DEFAULT_MODES above or pass modes here.",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel worker processes.")
    parser.add_argument("--batch-name", type=str, default="ofat_sensitivity", help="Output folder label.")
    parser.add_argument(
        "--parameters",
        nargs="*",
        default=OFAT_PARAMS,
        help="Optional subset of parameters. Default: all OFAT parameters except seed, including boolean toggles.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent.parent / "outputs"
    params_to_run = [param for param in args.parameters if param in OFAT_PARAMS]
    if not params_to_run:
        raise SystemExit("No valid parameters selected.")
    modes_to_run = [mode for mode in args.modes if mode in MODES]
    if not modes_to_run:
        raise SystemExit(f"No valid modes selected. Choose from: {', '.join(MODES)}")

    tasks, param_grids, grid_strategies = _build_tasks(params_to_run, args.steps, modes_to_run)
    total_runs = len(tasks)
    output_dir = _make_output_dir(script_dir, args.batch_name)
    timeseries_dir = output_dir / "timeseries"

    print("=" * 72)
    print("OFAT sensitivity sweep")
    print("=" * 72)
    print(f"Output folder       : {output_dir}")
    print(f"Parameters          : {len(params_to_run)}")
    print(f"Modes               : {', '.join(modes_to_run)}")
    bool_param_count = sum(1 for p in params_to_run if p in BOOL_PARAMS)
    numeric_param_count = len(params_to_run) - bool_param_count
    print(f"Numeric parameters  : {numeric_param_count}")
    print(f"Boolean parameters  : {bool_param_count}")
    print(f"Grid points/param   : 5 for numeric, 2 for boolean")
    print(f"Replicates/grid     : {len(REPLICATE_SEEDS)}")
    print(f"Seeds               : {REPLICATE_SEEDS}")
    print(f"Steps/run           : {args.steps}")
    print(f"Workers             : {args.workers}")
    print(f"Total runs          : {total_runs}")

    started_at = time.perf_counter()
    summary_rows = []
    runtime_sum = 0.0

    print("\nRunning calibration task to initialize ETA...")
    calibration_result = _run_single_task(tasks[0])
    calibration_row = _write_result(calibration_result, timeseries_dir)
    summary_rows.append(calibration_row)
    runtime_sum += calibration_result["runtime_seconds"]
    calibration_runtime = calibration_result["runtime_seconds"]
    estimated_wallclock = (calibration_runtime * total_runs) / max(args.workers, 1)
    print(f"Calibration run     : {calibration_runtime:.2f}s")
    print(f"Initial ETA         : {_format_seconds(estimated_wallclock)} at {args.workers} workers")

    remaining_tasks = tasks[1:]
    completed_runs = 1
    _print_progress(completed_runs, total_runs, time.perf_counter() - started_at, runtime_sum / completed_runs)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run_single_task, task) for task in remaining_tasks]
        for future in as_completed(futures):
            result = future.result()
            row = _write_result(result, timeseries_dir)
            summary_rows.append(row)
            completed_runs += 1
            runtime_sum += result["runtime_seconds"]
            elapsed_seconds = time.perf_counter() - started_at
            _print_progress(completed_runs, total_runs, elapsed_seconds, runtime_sum / completed_runs)

    print()

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["parameter", "mode", "grid_index", "replicate_index"]
    )
    aggregated_df = _build_aggregated(summary_df)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    aggregated_df.to_csv(output_dir / "aggregated.csv", index=False)

    metadata = {
        "batch_name": args.batch_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_steps": args.steps,
        "workers": args.workers,
        "modes": modes_to_run,
        "parameters": params_to_run,
        "parameter_count": len(params_to_run),
        "grid_points_per_parameter": {"numeric": 5, "boolean": 2},
        "replicates_per_point": len(REPLICATE_SEEDS),
        "replicate_seeds": REPLICATE_SEEDS,
        "total_runs": total_runs,
        "dashboard_default_params": BASE_PARAMS,
        "grid_strategies": grid_strategies,
        "parameter_grids": param_grids,
        "summary_file": "summary.csv",
        "aggregated_file": "aggregated.csv",
        "timeseries_dir": "timeseries",
        "final_metric_names": sorted(
            col.replace("final__", "") for col in summary_df.columns if col.startswith("final__")
        ),
        "key_metrics": KEY_METRICS,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    elapsed_seconds = time.perf_counter() - started_at
    print("\nFinished OFAT sensitivity sweep.")
    print(f"Wallclock runtime   : {_format_seconds(elapsed_seconds)}")
    print(f"Average time/run    : {runtime_sum / max(total_runs, 1):.2f}s")
    print(f"Saved summary       : {output_dir / 'summary.csv'}")
    print(f"Saved aggregated    : {output_dir / 'aggregated.csv'}")
    print(f"Saved metadata      : {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
