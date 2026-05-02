import math
from typing import Any, Dict, List
import pandas as pd
from .model import GPT, GPTConfig, count_parameters


def approximate_training_flops(num_params: float, num_tokens: float) -> float:
    return 6.0 * float(num_params) * float(num_tokens)


def effective_tokens_per_step(training_cfg: Dict[str, Any]) -> int:
    return (
        int(training_cfg["batch_size"])
        * int(training_cfg["block_size"])
        * int(training_cfg.get("grad_accum_steps", 1))
    )


def _catalog_from_model_grid(config: Dict[str, Any], vocab_size: int) -> pd.DataFrame:
    training_cfg = config["training"]
    rows: List[Dict[str, Any]] = []
    for spec in config["model_grid"]:
        gpt_config = GPTConfig(
            vocab_size=vocab_size,
            block_size=int(training_cfg["block_size"]),
            n_layer=int(spec["n_layer"]),
            n_head=int(spec["n_head"]),
            n_embd=int(spec["n_embd"]),
            dropout=float(spec.get("dropout", training_cfg.get("dropout", 0.1))),
            bias=bool(spec.get("bias", True)),
            mlp_ratio=float(spec.get("mlp_ratio", 4.0)),
        )
        model = GPT(gpt_config)
        n_params = count_parameters(model)
        rows.append(
            {
                "model_name": spec["name"],
                "n_layer": gpt_config.n_layer,
                "n_head": gpt_config.n_head,
                "n_embd": gpt_config.n_embd,
                "dropout": gpt_config.dropout,
                "bias": gpt_config.bias,
                "mlp_ratio": gpt_config.mlp_ratio,
                "num_params": int(n_params),
            }
        )
        del model
    return pd.DataFrame(rows).sort_values("num_params").reset_index(drop=True)



def build_sweep_plan(config: Dict[str, Any], vocab_size: int) -> pd.DataFrame:
    training_cfg = config["training"]
    compute_cfg = config["compute"]
    planning_cfg = config.get("planning", {})

    model_catalog = _catalog_from_model_grid(config, vocab_size=vocab_size)
    budgets = [float(x) for x in compute_cfg["budgets"]]
    seeds = [int(x) for x in config["experiment"].get("seeds", [0])]
    schedules = list(config["experiment"].get("schedule_regimes", ["matched", "fixed"]))
    tps = effective_tokens_per_step(training_cfg)
    min_steps = int(planning_cfg.get("min_steps_per_run", 10))
    max_steps = int(planning_cfg.get("max_steps_per_run", 10_000_000))

    base_rows: List[Dict[str, Any]] = []
    budget_to_rows: Dict[int, List[int]] = {}

    for budget_idx, budget in enumerate(budgets):
        budget_rows = []
        for _, model_row in model_catalog.iterrows():
            n_params = float(model_row["num_params"])
    
            target_tokens = budget / (6.0 * n_params)
            unclipped_train_steps = max(1, math.ceil(target_tokens / tps))
    
            if unclipped_train_steps < min_steps:
                continue
            if unclipped_train_steps > max_steps:
                continue
    
            train_steps = unclipped_train_steps
            actual_tokens = train_steps * tps
    
            target_d_over_n = float(target_tokens) / float(n_params)
            actual_d_over_n = float(actual_tokens) / float(n_params)
    
            record = {
                "budget_id": budget_idx,
                "budget_flops": budget,
                "model_name": model_row["model_name"],
                "n_layer": int(model_row["n_layer"]),
                "n_head": int(model_row["n_head"]),
                "n_embd": int(model_row["n_embd"]),
                "dropout": float(model_row["dropout"]),
                "bias": bool(model_row["bias"]),
                "mlp_ratio": float(model_row["mlp_ratio"]),
                "num_params": int(model_row["num_params"]),
                "tokens_per_step": int(tps),
                "target_train_tokens": float(target_tokens),
                "actual_train_tokens": int(actual_tokens),
                "target_d_over_n": target_d_over_n,
                "actual_d_over_n": actual_d_over_n,
                "train_steps": int(train_steps),
                "unclipped_train_steps": int(unclipped_train_steps),
                "actual_compute_proxy": approximate_training_flops(
                    model_row["num_params"], actual_tokens
                ),
            }
            budget_rows.append(record)
        if not budget_rows:
            raise ValueError(
                f"Budget {budget:.3e} produced zero feasible runs. Adjust budgets or model grid."
            )
        budget_to_rows[budget_idx] = budget_rows
        base_rows.extend(budget_rows)

    fixed_horizon_by_budget = {}
    fixed_mode = str(training_cfg.get("fixed_schedule_horizon", "median_per_budget"))
    for budget_idx, rows in budget_to_rows.items():
        step_values = [r["train_steps"] for r in rows]
        if fixed_mode == "max_per_budget":
            fixed_horizon = max(step_values)
        elif fixed_mode == "min_per_budget":
            fixed_horizon = min(step_values)
        elif fixed_mode == "median_per_budget":
            fixed_horizon = int(round(float(pd.Series(step_values).median())))
        else:
            fixed_horizon = int(fixed_mode)
        fixed_horizon_by_budget[budget_idx] = max(1, fixed_horizon)

    expanded_rows: List[Dict[str, Any]] = []
    for row in base_rows:
        for schedule in schedules:
            for seed in seeds:
                out_row = dict(row)
                out_row["schedule_regime"] = schedule
                out_row["seed"] = seed
                out_row["fixed_schedule_horizon_steps"] = fixed_horizon_by_budget[row["budget_id"]]
                budget_tag = f"b{row['budget_id']}"
                model_tag = str(row["model_name"])
                run_id = f"{budget_tag}__{model_tag}__{schedule}__seed{seed}"
                out_row["run_id"] = run_id
                expanded_rows.append(out_row)

    plan = pd.DataFrame(expanded_rows).sort_values(
        ["budget_flops", "num_params", "schedule_regime", "seed"]
    )
    return plan.reset_index(drop=True)
