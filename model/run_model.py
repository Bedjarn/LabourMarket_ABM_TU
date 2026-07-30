"""
Run script for the Labour Market ABM - four adoption-mode comparison.

Runs the model in all four adoption modes with identical parameters and seed,
then exports the same figures and Excel workbook used by the dashboard.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from base_parameters import BASE_PARAMS
from dashboard_utils import (
    MODES,
    build_combined_dataframe,
    build_other_figures,
    build_summary_dataframe,
    export_bundle,
    run_simulation,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


N_STEPS = 250


def _print_progress(mode: str, completed: int, total: int):
    width = 32
    ratio = completed / max(1, total)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {completed:>4}/{total}  {ratio:>6.1%}  current mode: {mode}", end="", flush=True)


def _make_output_dir() -> Path:
    root = Path(__file__).resolve().parent / "outputs" / "run_model_outputs"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = root / f"{timestamp}_run_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    print("=" * 60)
    print("Labour Market ABM - Adoption Mode Comparison")
    print("=" * 60)

    output_dir = _make_output_dir()
    bundle = run_simulation(
        params=BASE_PARAMS,
        n_steps=N_STEPS,
        modes=MODES,
        run_label="run_model",
        progress_callback=_print_progress,
    )
    print()

    exported = export_bundle(bundle, output_dir)
    build_summary_dataframe(bundle).to_csv(output_dir / "summary.csv")
    build_combined_dataframe(bundle).to_csv(output_dir / "combined.csv")

    # Export all "Other plots" diagnostic figures
    other_figures = build_other_figures(bundle.results)
    other_paths = {}
    for group_name, fig in other_figures.items():
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in group_name).strip("_").lower()
        fig_path = output_dir / f"other_{safe_name}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        other_paths[group_name] = fig_path

    all_artifacts = {**{k: v.name for k, v in exported.items()},
                     **{f"other_{k}": v.name for k, v in other_paths.items()}}

    metadata = {
        "run_label": bundle.run_label,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_steps": bundle.n_steps,
        "modes": bundle.modes,
        "params": bundle.params,
        "artifacts": all_artifacts,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nAll modes complete.\n")
    print(f"Output folder: {output_dir}")
    for name in all_artifacts.values():
        print(f"Saved {name}")
    print("Saved summary.csv")
    print("Saved combined.csv")
    print("Saved metadata.json")


if __name__ == "__main__":
    main()
