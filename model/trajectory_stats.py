"""
trajectory_stats.py - statistical comparison of two simulation configurations
over the *whole* time-series trajectory, not just the final tick.

Why this module exists
----------------------
The model is stochastic: every seed produces a different path. Comparing two
configurations on a single final-tick value throws away the entire trajectory
and ignores run-to-run noise. The question the thesis actually asks is "do these
two configurations produce *different dynamics*?", a statement about the two
mean curves across the full horizon.

Because each configuration is run with the SAME set of seeds (common random
numbers), the comparison is *paired*: for a given seed, configuration A and
configuration B share the same initial draw, the same separation process, the
same matching order. Subtracting them per seed removes that shared random
component and isolates the effect of the parameter change. This pairing is the
reason the test below is built on the per-seed difference matrix.

The procedure is a paired sign-flip permutation test on the studentised
pointwise mean difference, with a max-statistic (sup-t) over time. The single
resampling delivers three things at once:

  1. a global p-value for H0 "the two configurations have identical
     trajectories" (family-wise across all time steps);
  2. a *simultaneous* confidence band for the mean difference curve, whose
     critical width c* is the (1-alpha) quantile of the permutation null of
     max_t |t(t)|; and
  3. the set of time steps at which the band excludes zero - i.e. *where* along
     the horizon the two lines differ, with multiplicity already controlled.

Distribution-free (no normality assumption) and exact up to the number of
permutations, which suits an agent-based model whose output is neither normal
nor independent across time. Pure NumPy, reusable by the dashboard and the
head-less batch runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class TrajectoryComparison:
    """Outcome of comparing configuration A against configuration B on one metric."""

    metric: str
    n_seeds: int
    n_steps: int
    steps: np.ndarray

    mean_diff: np.ndarray
    se_diff: np.ndarray
    band_low: np.ndarray
    band_high: np.ndarray
    sig_mask: np.ndarray

    mean_a: np.ndarray
    mean_b: np.ndarray
    sd_a: np.ndarray
    sd_b: np.ndarray

    p_value: float
    stat_obs: float
    crit_value: float
    alpha: float
    n_perm: int

    mean_abs_diff: float
    mean_signed_diff: float
    cohens_dz: float
    signed_diff_ci: tuple
    n_sig: int = 0
    frac_sig: float = 0.0
    valid: bool = True
    note: str = ""


def stack_runs(runs: Sequence[np.ndarray]) -> np.ndarray:
    """Stack per-seed 1-D series into an (S, T) matrix, truncating to common T."""
    arrs = [np.asarray(r, dtype=float).ravel() for r in runs]
    if not arrs:
        return np.empty((0, 0))
    t_min = min(a.shape[0] for a in arrs)
    return np.vstack([a[:t_min] for a in arrs])


def paired_difference(runs_a: Sequence[np.ndarray],
                      runs_b: Sequence[np.ndarray]) -> np.ndarray:
    """Per-seed difference matrix D = A - B, shape (S, T) (common random numbers)."""
    A = stack_runs(runs_a)
    B = stack_runs(runs_b)
    if A.size == 0 or B.size == 0:
        return np.empty((0, 0))
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            f"Paired test needs equal seed counts: A has {A.shape[0]}, B has {B.shape[0]}."
        )
    t_min = min(A.shape[1], B.shape[1])
    return A[:, :t_min] - B[:, :t_min]


def _pointwise(D: np.ndarray):
    """Return (mean_diff, se_diff, tstat) over seeds for each time step."""
    S = D.shape[0]
    mbar = np.nanmean(D, axis=0)
    sd = np.nanstd(D, axis=0, ddof=1) if S > 1 else np.zeros(D.shape[1])
    se = sd / np.sqrt(S)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, mbar / se, 0.0)
    return mbar, se, tstat


def permutation_test(D: np.ndarray, n_perm: int = 5000, alpha: float = 0.05, seed: int = 0):
    """Paired sign-flip permutation test on D (S, T) with statistic max_t |t(t)|.

    Both numerator and denominator are recomputed from the sign-flipped data on
    every permutation, exactly as for the observed statistic; freezing the
    denominator makes the test badly anti-conservative.
    """
    S, T = D.shape
    mbar, se, tstat = _pointwise(D)
    stat_obs = float(np.nanmax(np.abs(tstat))) if T else 0.0

    rng = np.random.default_rng(seed)
    perm_max = np.empty(n_perm)
    sqrtS = np.sqrt(S)
    for i in range(n_perm):
        signs = rng.choice((-1.0, 1.0), size=S)
        X = signs[:, None] * D
        m = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=1) if S > 1 else np.zeros(T)
        se_p = sd / sqrtS
        with np.errstate(divide="ignore", invalid="ignore"):
            t_perm = np.where(se_p > 0, m / se_p, 0.0)
        perm_max[i] = np.nanmax(np.abs(t_perm))

    p_value = (1.0 + np.sum(perm_max >= stat_obs)) / (1.0 + n_perm)
    crit = float(np.quantile(perm_max, 1.0 - alpha))
    sig_mask = np.abs(tstat) > crit
    band_low = mbar - crit * se
    band_high = mbar + crit * se
    return {
        "mean_diff": mbar, "se_diff": se, "tstat": tstat,
        "stat_obs": stat_obs, "p_value": float(p_value),
        "crit_value": crit, "alpha": alpha, "n_perm": n_perm,
        "sig_mask": sig_mask, "band_low": band_low, "band_high": band_high,
    }


def _bootstrap_signed_ci(D: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 1):
    """Percentile bootstrap CI (over seeds) for the time-averaged signed diff."""
    S = D.shape[0]
    if S < 2:
        v = float(np.nanmean(D))
        return (v, v)
    per_seed_summary = np.nanmean(D, axis=1)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, S, size=S)
        boot[i] = np.mean(per_seed_summary[idx])
    return (float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2)))


def _effect_sizes(D: np.ndarray):
    """Interpretable effect magnitudes from the difference matrix."""
    mbar = np.nanmean(D, axis=0)
    mean_abs = float(np.nanmean(np.abs(mbar)))
    mean_signed = float(np.nanmean(mbar))
    per_seed_summary = np.nanmean(D, axis=1)
    if D.shape[0] > 1:
        sd_summary = float(np.nanstd(per_seed_summary, ddof=1))
        dz = float(np.nanmean(per_seed_summary) / sd_summary) if sd_summary > 0 else 0.0
    else:
        dz = 0.0
    return mean_abs, mean_signed, dz


def _empty_result(metric, alpha, n_perm, T, note, n_seeds=0):
    z = np.full(T, np.nan) if T else np.array([])
    return TrajectoryComparison(
        metric=metric, n_seeds=n_seeds, n_steps=T, steps=(np.arange(T) if T else np.array([])),
        mean_diff=z.copy(), se_diff=z.copy(), band_low=z.copy(), band_high=z.copy(),
        sig_mask=(np.zeros(T, dtype=bool) if T else np.array([], dtype=bool)),
        mean_a=z.copy(), mean_b=z.copy(), sd_a=z.copy(), sd_b=z.copy(),
        p_value=float("nan"), stat_obs=float("nan"), crit_value=float("nan"),
        alpha=alpha, n_perm=n_perm, mean_abs_diff=float("nan"),
        mean_signed_diff=float("nan"), cohens_dz=float("nan"),
        signed_diff_ci=(float("nan"), float("nan")), valid=False, note=note,
    )


def compare_trajectories(runs_a: Sequence[np.ndarray], runs_b: Sequence[np.ndarray],
                         metric: str = "", steps: Optional[np.ndarray] = None,
                         n_perm: int = 5000, n_boot: int = 5000,
                         alpha: float = 0.05, seed: int = 0) -> TrajectoryComparison:
    """Full paired comparison of two configurations on one metric."""
    A = stack_runs(runs_a)
    B = stack_runs(runs_b)
    if A.size == 0 or B.size == 0:
        return _empty_result(metric, alpha, n_perm, 0, "No data for this metric/mode.")

    D = paired_difference(runs_a, runs_b)
    T = D.shape[1]
    if not np.isfinite(D).any():
        return _empty_result(metric, alpha, n_perm, T,
                             "Metric is undefined (all NaN) in at least one configuration.",
                             n_seeds=D.shape[0])

    perm = permutation_test(D, n_perm=n_perm, alpha=alpha, seed=seed)
    ci = _bootstrap_signed_ci(D, n_boot=n_boot, alpha=alpha, seed=seed + 1)
    mean_abs, mean_signed, dz = _effect_sizes(D)

    t_min = min(A.shape[1], B.shape[1])
    x = np.arange(t_min) if steps is None else np.asarray(steps)[:t_min]
    sig = perm["sig_mask"]
    return TrajectoryComparison(
        metric=metric, n_seeds=D.shape[0], n_steps=T, steps=x,
        mean_diff=perm["mean_diff"], se_diff=perm["se_diff"],
        band_low=perm["band_low"], band_high=perm["band_high"], sig_mask=sig,
        mean_a=np.nanmean(A[:, :t_min], axis=0), mean_b=np.nanmean(B[:, :t_min], axis=0),
        sd_a=(np.nanstd(A[:, :t_min], axis=0, ddof=1) if A.shape[0] > 1 else np.zeros(t_min)),
        sd_b=(np.nanstd(B[:, :t_min], axis=0, ddof=1) if B.shape[0] > 1 else np.zeros(t_min)),
        p_value=perm["p_value"], stat_obs=perm["stat_obs"], crit_value=perm["crit_value"],
        alpha=alpha, n_perm=n_perm,
        mean_abs_diff=mean_abs, mean_signed_diff=mean_signed, cohens_dz=dz,
        signed_diff_ci=ci,
        n_sig=int(np.sum(sig)), frac_sig=float(np.mean(sig)) if T else 0.0, valid=True,
    )


def summary_row(cmp: "TrajectoryComparison", label_a: str, label_b: str) -> dict:
    """Flat dict suitable for a results table / CSV row."""
    sig_flag = (cmp.p_value < cmp.alpha) if np.isfinite(cmp.p_value) else False
    return {
        "comparison": f"{label_a}  vs  {label_b}", "metric": cmp.metric,
        "n_seeds": cmp.n_seeds, "p_value": cmp.p_value, "significant": sig_flag,
        "alpha": cmp.alpha, "mean_signed_diff": cmp.mean_signed_diff,
        "signed_diff_CI_low": cmp.signed_diff_ci[0], "signed_diff_CI_high": cmp.signed_diff_ci[1],
        "mean_abs_diff": cmp.mean_abs_diff, "cohens_dz": cmp.cohens_dz,
        "frac_horizon_significant": cmp.frac_sig, "sup_abs_t": cmp.stat_obs,
        "crit_value": cmp.crit_value, "note": cmp.note,
    }
