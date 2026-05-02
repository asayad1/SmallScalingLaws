import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .planner import approximate_training_flops
from .utils import ensure_dir, read_json

matplotlib.use("Agg")

FIT_MIN_TRAIN_STEPS = 300
FIT_MIN_D_OVER_N = 10.0
FIT_GRID_POINTS = 400


def collect_run_summaries(out_root: str | Path) -> pd.DataFrame:
    run_root = Path(out_root) / "runs"
    rows: List[Dict[str, Any]] = []
    if not run_root.exists():
        return pd.DataFrame()
    for summary_path in sorted(run_root.glob("*/summary.json")):
        rows.append(read_json(summary_path))
    return pd.DataFrame(rows)


def _ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "actual_d_over_n" not in df.columns:
        if "actual_train_tokens" in df.columns and "num_params" in df.columns:
            df["actual_d_over_n"] = df["actual_train_tokens"] / df["num_params"]

    if "token_parameter_ratio" not in df.columns and "actual_d_over_n" in df.columns:
        df["token_parameter_ratio"] = df["actual_d_over_n"]

    return df


def aggregate_results(seed_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if seed_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    seed_df = _ensure_derived_columns(seed_df)

    group_cols = [
        "budget_id",
        "budget_flops",
        "schedule_regime",
        "model_name",
        "n_layer",
        "n_head",
        "n_embd",
        "num_params",
        "target_train_tokens",
        "actual_train_tokens",
        "actual_compute_proxy",
        "fixed_schedule_horizon_steps",
    ]

    agg_df = (
        seed_df.groupby(group_cols, dropna=False)
        .agg(
            runs=("seed", "nunique"),
            mean_train_steps=("train_steps", "mean"),
            mean_final_val_loss=("final_val_loss", "mean"),
            std_final_val_loss=("final_val_loss", "std"),
            mean_best_val_loss=("best_val_loss", "mean"),
            mean_final_test_loss=("final_test_loss", "mean"),
            mean_runtime_sec=("runtime_sec", "mean"),
            mean_tokens_per_second=("tokens_per_second", "mean"),
        )
        .reset_index()
    )

    agg_df["actual_d_over_n"] = agg_df["actual_train_tokens"] / agg_df["num_params"]
    agg_df["token_parameter_ratio"] = agg_df["actual_d_over_n"]

    frontier_df = (
        agg_df.sort_values("mean_final_val_loss")
        .groupby(["budget_id", "budget_flops", "schedule_regime"], as_index=False)
        .first()
        .sort_values(["schedule_regime", "budget_flops"])
        .reset_index(drop=True)
    )
    frontier_df["frontier_token_parameter_ratio"] = (
        frontier_df["actual_train_tokens"] / frontier_df["num_params"]
    )

    return agg_df, frontier_df


def build_frontier_edge_diagnostics(agg_df: pd.DataFrame) -> pd.DataFrame:
    if agg_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for (budget, schedule), group in agg_df.groupby(["budget_flops", "schedule_regime"]):
        g = group.sort_values("num_params").reset_index(drop=True)
        best_idx = g["mean_final_val_loss"].idxmin()
        best = g.loc[best_idx]

        min_n = float(g["num_params"].min())
        max_n = float(g["num_params"].max())

        rows.append(
            {
                "budget_flops": budget,
                "schedule_regime": schedule,
                "best_model_name": best["model_name"],
                "best_num_params": best["num_params"],
                "best_actual_train_tokens": best["actual_train_tokens"],
                "best_actual_d_over_n": best["actual_d_over_n"],
                "best_mean_val_loss": best["mean_final_val_loss"],
                "is_left_edge": bool(best["num_params"] == min_n),
                "is_right_edge": bool(best["num_params"] == max_n),
                "min_num_params_in_group": min_n,
                "max_num_params_in_group": max_n,
                "num_models_in_group": int(g.shape[0]),
            }
        )

    return pd.DataFrame(rows).sort_values(["schedule_regime", "budget_flops"]).reset_index(drop=True)


def filter_points_for_surface_fit(agg_df: pd.DataFrame) -> pd.DataFrame:
    if agg_df.empty:
        return pd.DataFrame()

    fit_df = agg_df.copy()
    fit_df = fit_df[
        (fit_df["mean_train_steps"] >= FIT_MIN_TRAIN_STEPS) &
        (fit_df["actual_d_over_n"] >= FIT_MIN_D_OVER_N)
    ].copy()

    return fit_df.reset_index(drop=True)


def _fit_power_law_surface(df: pd.DataFrame) -> Dict[str, float]:
    if df.shape[0] < 4:
        return {
            "status": "insufficient_points",
            "points_used": int(df.shape[0]),
            "E": math.nan,
            "A": math.nan,
            "alpha": math.nan,
            "B": math.nan,
            "beta": math.nan,
            "rmse": math.nan,
        }

    n = df["num_params"].to_numpy(dtype=float)
    d = df["actual_train_tokens"].to_numpy(dtype=float)
    y = df["mean_final_val_loss"].to_numpy(dtype=float)
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    loss_range = max(1e-6, y_max - y_min)

    bounds = [
        (max(0.0, y_min - loss_range), y_min),
        (-20.0, 20.0),
        (1e-3, 2.5),
        (-20.0, 20.0),
        (1e-3, 2.5),
    ]

    def unpack(theta: np.ndarray) -> Tuple[float, float, float, float, float]:
        e = float(theta[0])
        a = math.exp(float(theta[1]))
        alpha = float(theta[2])
        b = math.exp(float(theta[3]))
        beta = float(theta[4])
        return e, a, alpha, b, beta

    def predict(theta: np.ndarray) -> np.ndarray:
        e, a, alpha, b, beta = unpack(theta)
        return e + a * np.power(n, -alpha) + b * np.power(d, -beta)

    def objective(theta: np.ndarray) -> float:
        pred = predict(theta)
        return float(np.mean(np.square(pred - y)))

    init_templates = [
        np.array([max(0.0, y_min - 0.20 * loss_range), math.log(loss_range), 0.20, math.log(loss_range), 0.20]),
        np.array([max(0.0, y_min - 0.10 * loss_range), math.log(loss_range), 0.35, math.log(loss_range), 0.35]),
        np.array([max(0.0, y_min - 0.05 * loss_range), math.log(loss_range), 0.50, math.log(loss_range), 0.50]),
    ]

    best = None
    for init in init_templates:
        res = minimize(objective, init, method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res

    if best is None or not best.success:
        return {
            "status": "fit_failed",
            "points_used": int(df.shape[0]),
            "E": math.nan,
            "A": math.nan,
            "alpha": math.nan,
            "B": math.nan,
            "beta": math.nan,
            "rmse": math.nan,
        }

    e, a, alpha, b, beta = unpack(best.x)
    rmse = float(np.sqrt(best.fun))
    return {
        "status": "ok",
        "points_used": int(df.shape[0]),
        "E": e,
        "A": a,
        "alpha": alpha,
        "B": b,
        "beta": beta,
        "rmse": rmse,
    }


def _surface_predict(n: np.ndarray, d: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    return (
        params["E"]
        + params["A"] * np.power(n, -params["alpha"])
        + params["B"] * np.power(d, -params["beta"])
    )


def _predict_optimum_from_surface_bounded(
    budget_flops: float,
    params: Dict[str, float],
    support_df: pd.DataFrame,
) -> Tuple[float, float, float, bool, bool]:
    if support_df.empty:
        return math.nan, math.nan, math.nan, False, False

    e = params["E"]
    a = params["A"]
    alpha = params["alpha"]
    b = params["B"]
    beta = params["beta"]

    if min(a, alpha, b, beta) <= 0 or any(math.isnan(x) for x in [e, a, alpha, b, beta]):
        return math.nan, math.nan, math.nan, False, False

    n_min = float(support_df["num_params"].min())
    n_max = float(support_df["num_params"].max())
    d_min = float(support_df["actual_train_tokens"].min())
    d_max = float(support_df["actual_train_tokens"].max())

    candidate_ns = np.logspace(np.log10(n_min), np.log10(n_max), FIT_GRID_POINTS)
    candidate_ds = budget_flops / (6.0 * candidate_ns)

    support_mask = (candidate_ds >= d_min) & (candidate_ds <= d_max)
    if not np.any(support_mask):
        return math.nan, math.nan, math.nan, False, False

    candidate_ns = candidate_ns[support_mask]
    candidate_ds = candidate_ds[support_mask]

    pred_losses = _surface_predict(candidate_ns, candidate_ds, params)
    best_idx = int(np.argmin(pred_losses))
    n_opt = float(candidate_ns[best_idx])
    d_opt = float(candidate_ds[best_idx])
    pred_loss = float(pred_losses[best_idx])

    within_n_support = bool(n_min <= n_opt <= n_max)
    within_d_support = bool(d_min <= d_opt <= d_max)
    return n_opt, d_opt, pred_loss, within_n_support, within_d_support


def fit_surfaces(agg_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if agg_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    fit_df = filter_points_for_surface_fit(agg_df)

    surface_rows: List[Dict[str, Any]] = []
    frontier_rows: List[Dict[str, Any]] = []
    budgets = sorted(agg_df["budget_flops"].unique())

    for schedule, group in agg_df.groupby("schedule_regime"):
        fit_group = fit_df[fit_df["schedule_regime"] == schedule].copy()

        params = _fit_power_law_surface(fit_group)
        surface_row = {
            "schedule_regime": schedule,
            "fit_min_train_steps": FIT_MIN_TRAIN_STEPS,
            "fit_min_d_over_n": FIT_MIN_D_OVER_N,
            "observed_points_total": int(group.shape[0]),
            "observed_points_used": int(fit_group.shape[0]),
            **params,
        }
        surface_rows.append(surface_row)

        if params["status"] != "ok":
            continue

        n_support_min = float(fit_group["num_params"].min())
        n_support_max = float(fit_group["num_params"].max())
        d_support_min = float(fit_group["actual_train_tokens"].min())
        d_support_max = float(fit_group["actual_train_tokens"].max())

        for budget in budgets:
            n_opt, d_opt, pred_loss, within_n_support, within_d_support = _predict_optimum_from_surface_bounded(
                budget,
                params,
                fit_group,
            )
            frontier_rows.append(
                {
                    "schedule_regime": schedule,
                    "budget_flops": budget,
                    "predicted_num_params": n_opt,
                    "predicted_train_tokens": d_opt,
                    "predicted_val_loss": pred_loss,
                    "predicted_compute_proxy": approximate_training_flops(n_opt, d_opt) if not math.isnan(n_opt) else math.nan,
                    "predicted_token_parameter_ratio": (d_opt / n_opt) if not math.isnan(n_opt) else math.nan,
                    "within_n_support": within_n_support,
                    "within_d_support": within_d_support,
                    "fit_n_support_min": n_support_min,
                    "fit_n_support_max": n_support_max,
                    "fit_d_support_min": d_support_min,
                    "fit_d_support_max": d_support_max,
                }
            )

    return (
        pd.DataFrame(surface_rows),
        pd.DataFrame(frontier_rows),
        fit_df,
    )


def _save_plot(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_iso_compute_curves(
    agg_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    plot_dir: Path,
) -> None:
    if agg_df.empty:
        return

    for (budget, schedule), group in agg_df.groupby(["budget_flops", "schedule_regime"]):
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(
            group["num_params"],
            group["actual_train_tokens"],
            c=group["mean_final_val_loss"],
            s=80,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Model parameters N")
        ax.set_ylabel("Training tokens D")
        ax.set_title(f"Iso-compute sweep | C≈{budget:.2e} | schedule={schedule}")

        for _, row in group.iterrows():
            ax.annotate(
                row["model_name"],
                (row["num_params"], row["actual_train_tokens"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        frontier_match = frontier_df[
            (frontier_df["budget_flops"] == budget) & (frontier_df["schedule_regime"] == schedule)
        ]
        edge_match = edge_df[
            (edge_df["budget_flops"] == budget) & (edge_df["schedule_regime"] == schedule)
        ]

        if not frontier_match.empty:
            best = frontier_match.iloc[0]
            ax.scatter([best["num_params"]], [best["actual_train_tokens"]], marker="*", s=220)

        if not edge_match.empty:
            ed = edge_match.iloc[0]
            if bool(ed["is_left_edge"]) or bool(ed["is_right_edge"]):
                side = "left" if bool(ed["is_left_edge"]) else "right"
                ax.text(
                    0.02,
                    0.02,
                    f"Frontier hits {side} edge",
                    transform=ax.transAxes,
                    fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )

        fig.colorbar(scatter, ax=ax, label="Mean final val loss")
        filename = f"iso_compute_budget_{budget:.0e}_{schedule}.png".replace("+", "")
        _save_plot(fig, plot_dir / filename)


def plot_frontiers(
    frontier_df: pd.DataFrame,
    fitted_frontier_df: pd.DataFrame,
    plot_dir: Path,
) -> None:
    if frontier_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for schedule, group in frontier_df.groupby("schedule_regime"):
        group = group.sort_values("budget_flops")
        ax.plot(group["budget_flops"], group["num_params"], marker="o", label=f"empirical-{schedule}")
    for schedule, group in fitted_frontier_df.groupby("schedule_regime"):
        group = group.sort_values("budget_flops")
        ax.plot(
            group["budget_flops"],
            group["predicted_num_params"],
            linestyle="--",
            label=f"fit-{schedule}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute proxy C ≈ 6ND")
    ax.set_ylabel("Optimal model parameters N")
    ax.set_title("Empirical and fitted N_opt(C)")
    ax.legend()
    _save_plot(fig, plot_dir / "frontier_num_params_vs_budget.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    for schedule, group in frontier_df.groupby("schedule_regime"):
        group = group.sort_values("budget_flops")
        ax.plot(
            group["budget_flops"],
            group["actual_train_tokens"],
            marker="o",
            label=f"empirical-{schedule}",
        )
    for schedule, group in fitted_frontier_df.groupby("schedule_regime"):
        group = group.sort_values("budget_flops")
        ax.plot(
            group["budget_flops"],
            group["predicted_train_tokens"],
            linestyle="--",
            label=f"fit-{schedule}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute proxy C ≈ 6ND")
    ax.set_ylabel("Optimal train tokens D")
    ax.set_title("Empirical and fitted D_opt(C)")
    ax.legend()
    _save_plot(fig, plot_dir / "frontier_tokens_vs_budget.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    for schedule, group in frontier_df.groupby("schedule_regime"):
        group = group.sort_values("budget_flops")
        ratio = group["frontier_token_parameter_ratio"]
        ax.plot(group["budget_flops"], ratio, marker="o", label=schedule)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute proxy C ≈ 6ND")
    ax.set_ylabel("Optimal D / N")
    ax.set_title("Compute-optimal token-to-parameter ratio")
    ax.legend()
    _save_plot(fig, plot_dir / "frontier_token_param_ratio_vs_budget.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    for schedule, group in frontier_df.groupby("schedule_regime"):
        group = group.sort_values("budget_flops")
        ax.plot(group["budget_flops"], group["mean_final_val_loss"], marker="o", label=schedule)
    ax.set_xscale("log")
    ax.set_xlabel("Compute proxy C ≈ 6ND")
    ax.set_ylabel("Frontier mean final val loss")
    ax.set_title("Best observed loss per budget")
    ax.legend()
    _save_plot(fig, plot_dir / "frontier_loss_vs_budget.png")


def plot_fit_diagnostics(
    agg_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    surface_df: pd.DataFrame,
    plot_dir: Path,
) -> None:
    if agg_df.empty or surface_df.empty or fit_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    observed_x: List[float] = []
    observed_y: List[float] = []

    for schedule, group in fit_df.groupby("schedule_regime"):
        params = surface_df[surface_df["schedule_regime"] == schedule]
        if params.empty or params.iloc[0]["status"] != "ok":
            continue
        p = params.iloc[0]
        pred = _surface_predict(
            group["num_params"].to_numpy(float),
            group["actual_train_tokens"].to_numpy(float),
            {
                "E": float(p["E"]),
                "A": float(p["A"]),
                "alpha": float(p["alpha"]),
                "B": float(p["B"]),
                "beta": float(p["beta"]),
            },
        )
        ax.scatter(group["mean_final_val_loss"], pred, label=f"{schedule} (filtered fit pts)")
        observed_x.extend(group["mean_final_val_loss"].tolist())
        observed_y.extend(pred.tolist())

    if observed_x and observed_y:
        lo = min(observed_x + observed_y)
        hi = max(observed_x + observed_y)
        ax.plot([lo, hi], [lo, hi], linestyle="--")

    ax.set_xlabel("Observed mean final val loss")
    ax.set_ylabel("Fitted val loss")
    ax.set_title("Power-law surface fit diagnostics")
    ax.legend()
    _save_plot(fig, plot_dir / "fit_observed_vs_predicted.png")


def run_analysis(out_root: str | Path) -> Dict[str, pd.DataFrame]:
    out_root = Path(out_root)
    ensure_dir(out_root)
    plot_dir = ensure_dir(out_root / "plots")

    seed_df = collect_run_summaries(out_root)
    if seed_df.empty:
        raise ValueError(f"No run summaries found under {out_root / 'runs'}")

    seed_df = _ensure_derived_columns(seed_df)
    seed_df = seed_df.sort_values(["budget_flops", "num_params", "schedule_regime", "seed"]).reset_index(drop=True)
    seed_df.to_csv(out_root / "seed_level_results.csv", index=False)

    agg_df, frontier_df = aggregate_results(seed_df)
    agg_df.to_csv(out_root / "aggregated_results.csv", index=False)
    frontier_df.to_csv(out_root / "empirical_frontier.csv", index=False)

    edge_df = build_frontier_edge_diagnostics(agg_df)
    edge_df.to_csv(out_root / "frontier_edge_diagnostics.csv", index=False)

    surface_df, fitted_frontier_df, fit_df = fit_surfaces(agg_df)
    surface_df.to_csv(out_root / "fitted_surface_params.csv", index=False)
    fitted_frontier_df.to_csv(out_root / "fitted_frontier.csv", index=False)
    fit_df.to_csv(out_root / "fit_filtered_points.csv", index=False)

    plot_iso_compute_curves(agg_df, frontier_df, edge_df, plot_dir)
    plot_frontiers(frontier_df, fitted_frontier_df, plot_dir)
    plot_fit_diagnostics(agg_df, fit_df, surface_df, plot_dir)

    return {
        "seed_df": seed_df,
        "agg_df": agg_df,
        "frontier_df": frontier_df,
        "edge_df": edge_df,
        "surface_df": surface_df,
        "fitted_frontier_df": fitted_frontier_df,
        "fit_df": fit_df,
    }