"""
scripts/train_model.py

Training script for the transformer world model.

Auto-detects environment:
  - Local (RTX 3060):  uses "small" model config, batch_size=64, AMP FP16
  - Cluster (A100):    uses "large" model config, batch_size=256, AMP FP16

Usage:
    # Local — small model on MiniGrid
    python scripts/train_model.py --env minigrid --scale small

    # Local — small model on Physics
    python scripts/train_model.py --env physics --scale small

    # Cluster — large model
    python scripts/train_model.py --env minigrid --scale large

    # Resume from checkpoint
    python scripts/train_model.py --env minigrid --scale small --resume checkpoints/minigrid_small_step5000.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import TrajectoryDataset, collate_fn
from models.transformer import WorldModelTransformer, TransformerConfig, build_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        print(f"GPU: {props.name} ({vram_gb:.1f}GB VRAM)")
    else:
        device = torch.device("cpu")
        print("No GPU found — training on CPU (will be slow)")
    return device


def detect_scale(device: torch.device, args_scale: str) -> str:
    """
    Auto-select model scale if not explicitly set.
    RTX 3060 (6GB) -> small. A100 (40GB+) -> large.
    """
    if args_scale != "auto":
        return args_scale

    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        scale = "large" if vram_gb >= 20 else "small"
    else:
        scale = "small"

    print(f"Auto-selected scale: {scale}")
    return scale


def get_hdf5_path(env: str) -> str:
    paths = {
        "minigrid": "data/trajectories/minigrid/minigrid.hdf5",
        "physics":  "data/trajectories/physics/physics.hdf5",
    }
    path = paths[env]
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No data found at {path}. "
            f"Run: python scripts/collect_data.py --env {env}"
        )
    return path


def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float) -> float:
    """
    Cosine learning rate schedule with linear warmup.
    Standard for transformer training — avoids early instability.
    """
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return lr * 0.5 * (1.0 + torch.tensor(progress * 3.14159).cos().item())


def save_checkpoint(
    model: WorldModelTransformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    env: str,
    scale: str,
    config: dict,
) -> str:
    Path("checkpoints").mkdir(exist_ok=True)
    path = f"checkpoints/{env}_{scale}_step{step}.pt"
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step":                 step,
        "loss":                 loss,
        "env":                  env,
        "scale":                scale,
        "model_config":         config,
    }, path)
    return path


def train(args) -> None:
    # ------------------------------------------------------------------ setup
    device = get_device()
    scale  = detect_scale(device, args.scale)

    model_config = load_config(args.config_dir + "/model_config.yaml")
    train_cfg    = model_config["training"]["cluster" if scale == "large" else "local"]
    wandb_cfg    = model_config.get("wandb", {})

    # ------------------------------------------------------------------ data
    hdf5_path = get_hdf5_path(args.env)

    train_dataset = TrajectoryDataset(
        hdf5_path=hdf5_path,
        split="train",
        context_length=model_config[scale]["context_length"],
        stride_steps=1,
    )
    val_dataset = TrajectoryDataset(
        hdf5_path=hdf5_path,
        split="val",
        context_length=model_config[scale]["context_length"],
        stride_steps=1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=train_cfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    # ----------------------------------------------------------------- model
    model = build_model(model_config, scale=scale).to(device)

    # ----------------------------------------------------------------- optim
    # Separate weight decay: apply to weights but NOT biases or LayerNorm params
    decay_params     = [p for n, p in model.named_parameters()
                        if p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters()
                        if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": train_cfg["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=train_cfg["learning_rate"], betas=(0.9, 0.95))

    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg["amp"] and device.type == "cuda")

    start_step = 0
    # Resume from checkpoint if provided
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step}")

    # ----------------------------------------------------------------- wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=wandb_cfg.get("project", "world-model-interpretability"),
                entity=wandb_cfg.get("entity"),
                name=f"{args.env}_{scale}",
                config={
                    "env": args.env,
                    "scale": scale,
                    "n_params": model.num_parameters(),
                    **model_config[scale],
                    **train_cfg,
                },
                tags=["track-a", "world-model", args.env, scale],
                resume="allow" if args.resume else None,
            )
        except ImportError:
            print("wandb not installed — skipping logging")
            use_wandb = False

    # --------------------------------------------------------------- training
    print(f"\nTraining {scale} model on {args.env}")
    print(f"  Parameters:  {model.num_parameters():,}")
    print(f"  Max steps:   {train_cfg['max_steps']:,}")
    print(f"  Batch size:  {train_cfg['batch_size']}")
    print(f"  Device:      {device}")
    print(f"  AMP:         {train_cfg['amp']}\n")

    model.train()
    step       = start_step
    best_val   = float("inf")
    train_iter = iter(train_loader)
    t0         = time.time()

    while step < train_cfg["max_steps"]:
        # Reload iterator when exhausted
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        tokens  = batch["tokens"].to(device)   # (B, T) int64
        targets = batch["targets"].to(device)  # (B, T) int64

        # Learning rate schedule
        lr = get_lr(step, train_cfg["warmup_steps"], train_cfg["max_steps"],
                    train_cfg["learning_rate"])
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Forward + backward with AMP
        with torch.cuda.amp.autocast(enabled=train_cfg["amp"] and device.type == "cuda"):
            _, loss = model(tokens, targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()

        step += 1

        # Logging
        if step % train_cfg["log_every"] == 0:
            elapsed = time.time() - t0
            tokens_per_sec = (train_cfg["batch_size"] *
                              model_config[scale]["context_length"] *
                              train_cfg["log_every"]) / elapsed
            print(f"Step {step:6d} | loss {loss.item():.4f} | "
                  f"lr {lr:.2e} | {tokens_per_sec/1e3:.1f}K tok/s")
            if use_wandb:
                wandb.log({"train/loss": loss.item(), "lr": lr,
                           "tokens_per_sec": tokens_per_sec}, step=step)
            t0 = time.time()

        # Validation
        if step % train_cfg["eval_every"] == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_batch in val_loader:
                    vt = val_batch["tokens"].to(device)
                    vg = val_batch["targets"].to(device)
                    with torch.cuda.amp.autocast(
                        enabled=train_cfg["amp"] and device.type == "cuda"
                    ):
                        _, vloss = model(vt, vg)
                    val_losses.append(vloss.item())

            val_loss = sum(val_losses) / len(val_losses)
            print(f"  Val loss: {val_loss:.4f}")
            if use_wandb:
                wandb.log({"val/loss": val_loss}, step=step)

            if val_loss < best_val:
                best_val = val_loss
                path = save_checkpoint(model, optimizer, step, val_loss,
                                       args.env, scale, model_config)
                print(f"  Saved best checkpoint: {path}")
            model.train()

        # Periodic checkpoint
        if step % train_cfg["save_every"] == 0:
            path = save_checkpoint(model, optimizer, step, loss.item(),
                                   args.env, scale, model_config)
            print(f"  Checkpoint saved: {path}")

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train transformer world model")
    parser.add_argument("--env",    choices=["minigrid", "physics"],
                        required=True)
    parser.add_argument("--scale",  choices=["small", "large", "auto"],
                        default="auto")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--config-dir", type=str, default="config")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
