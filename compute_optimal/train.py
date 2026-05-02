import math
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
from .data import PreparedDataset, RandomChunkBatcher, SequentialChunkBatcher, load_split_array
from .model import GPT, GPTConfig
from .planner import approximate_training_flops
from .utils import detect_device, ensure_dir, save_json, set_seed, torch_runtime_setup


def _lr_multiplier(
    step_idx: int,
    *,
    warmup_steps: int,
    decay_steps: int,
    min_lr_ratio: float,
) -> float:
    if step_idx < warmup_steps:
        return float(step_idx + 1) / float(max(1, warmup_steps))
    if step_idx >= decay_steps:
        return min_lr_ratio
    progress = (step_idx - warmup_steps) / float(max(1, decay_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


@torch.no_grad()
def evaluate_loss(
    model: GPT,
    array: np.ndarray,
    *,
    block_size: int,
    batch_size: int,
    num_batches: int,
    device: torch.device,
    autocast_enabled: bool,
) -> float:
    model.eval()
    batcher = SequentialChunkBatcher(array, block_size, batch_size, device)
    losses: List[float] = []
    device_type = "cuda" if device.type == "cuda" else "cpu"
    for xb, yb in batcher.iter_batches(num_batches):
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=autocast_enabled):
            _, loss = model(xb, yb)
        assert loss is not None
        losses.append(float(loss.item()))
    return float(np.mean(losses))



def _build_model(run_row: Dict[str, Any], training_cfg: Dict[str, Any], vocab_size: int) -> GPT:
    gpt_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=int(training_cfg["block_size"]),
        n_layer=int(run_row["n_layer"]),
        n_head=int(run_row["n_head"]),
        n_embd=int(run_row["n_embd"]),
        dropout=float(run_row.get("dropout", training_cfg.get("dropout", 0.1))),
        bias=bool(run_row.get("bias", True)),
        mlp_ratio=float(run_row.get("mlp_ratio", 4.0)),
    )
    return GPT(gpt_config)



def train_single_run(
    run_row: Dict[str, Any],
    config: Dict[str, Any],
    dataset: PreparedDataset,
    out_root: str | Path,
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    experiment_cfg = config["experiment"]
    training_cfg = config["training"]
    eval_cfg = config["evaluation"]
    run_dir = ensure_dir(Path(out_root) / "runs" / str(run_row["run_id"]))
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.csv"

    if summary_path.exists() and metrics_path.exists() and not overwrite:
        import json

        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    set_seed(int(run_row["seed"]))
    device = detect_device(str(experiment_cfg.get("device", "auto")))
    torch_runtime_setup(device, cpu_num_threads=int(experiment_cfg.get("cpu_num_threads", 1)))

    train_array = load_split_array(dataset.train_path)
    valid_array = load_split_array(dataset.valid_path)
    test_array = load_split_array(dataset.test_path)

    model = _build_model(run_row, training_cfg, vocab_size=dataset.vocab_size).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        betas=tuple(float(x) for x in training_cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(training_cfg.get("weight_decay", 0.1)),
    )

    total_steps = int(run_row["train_steps"])
    warmup_fraction = float(training_cfg.get("warmup_fraction", 0.05))
    min_lr_ratio = float(training_cfg.get("min_lr_ratio", 0.1))
    schedule_regime = str(run_row["schedule_regime"])
    
    if schedule_regime == "matched":
        decay_steps = total_steps
    elif schedule_regime == "fixed":
        decay_steps = int(run_row["fixed_schedule_horizon_steps"])
    else:
        raise ValueError(f"Unknown schedule regime: {schedule_regime}")
    
    warmup_steps = max(1, int(round(warmup_fraction * decay_steps)))
    warmup_steps = min(warmup_steps, max(1, decay_steps - 1))

    warmup_steps = min(warmup_steps, decay_steps)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step_idx: _lr_multiplier(
            step_idx,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            min_lr_ratio=min_lr_ratio,
        ),
    )

    use_amp = bool(training_cfg.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    batcher = RandomChunkBatcher(
        train_array,
        block_size=int(training_cfg["block_size"]),
        batch_size=int(training_cfg["batch_size"]),
        device=device,
    )

    metrics: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_step = -1
    progress = tqdm(range(1, total_steps + 1), desc=str(run_row["run_id"]), leave=False)
    start_time = time.time()
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    eval_interval = int(eval_cfg.get("eval_interval_steps", max(1, total_steps // 10)))
    eval_batches = int(eval_cfg.get("eval_batches", 20))
    device_type = "cuda" if device.type == "cuda" else "cpu"
    tokens_per_step = int(run_row["tokens_per_step"])

    final_val_loss = float("nan")
    final_test_loss = float("nan")
    final_train_loss = float("nan")

    for step in progress:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_accum = 0.0

        for _ in range(grad_accum_steps):
            xb, yb = batcher.get_batch()
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
                _, micro_loss = model(xb, yb)
                assert micro_loss is not None
                loss = micro_loss / grad_accum_steps
            train_loss_accum += float(micro_loss.item())
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if grad_clip > 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        final_train_loss = train_loss_accum / grad_accum_steps
        do_eval = step == 1 or step == total_steps or step % eval_interval == 0
        if do_eval:
            val_loss = evaluate_loss(
                model,
                valid_array,
                block_size=int(training_cfg["block_size"]),
                batch_size=int(eval_cfg.get("eval_batch_size", training_cfg["batch_size"])),
                num_batches=eval_batches,
                device=device,
                autocast_enabled=use_amp,
            )
            final_val_loss = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
            metrics.append(
                {
                    "step": step,
                    "train_tokens_seen": step * tokens_per_step,
                    "train_loss": final_train_loss,
                    "val_loss": val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            progress.set_postfix(train_loss=f"{final_train_loss:.4f}", val_loss=f"{val_loss:.4f}")

    final_test_loss = evaluate_loss(
        model,
        test_array,
        block_size=int(training_cfg["block_size"]),
        batch_size=int(eval_cfg.get("eval_batch_size", training_cfg["batch_size"])),
        num_batches=int(eval_cfg.get("test_eval_batches", eval_batches)),
        device=device,
        autocast_enabled=use_amp,
    )
    runtime_sec = time.time() - start_time

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(metrics_path, index=False)

    summary: Dict[str, Any] = {
        **run_row,
        "device": str(device),
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
        "best_val_loss": float(best_val_loss),
        "best_step": int(best_step),
        "final_test_loss": float(final_test_loss),
        "runtime_sec": float(runtime_sec),
        "tokens_per_second": float(run_row["actual_train_tokens"]) / max(runtime_sec, 1e-8),
        "approx_final_compute_proxy": approximate_training_flops(
            run_row["num_params"], run_row["actual_train_tokens"]
        ),
        "final_lr": float(optimizer.param_groups[0]["lr"]),
        "val_perplexity": float(math.exp(min(20.0, final_val_loss))),
        "test_perplexity": float(math.exp(min(20.0, final_test_loss))),
    }
    save_json(summary, summary_path)

    if bool(training_cfg.get("save_last_checkpoint", False)):
        checkpoint_path = run_dir / "model_last.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "summary": summary,
            },
            checkpoint_path,
        )

    return summary
