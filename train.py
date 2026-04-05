"""
CAPITA Training Loop
Handles: training, validation, checkpointing, logging, LR scheduling
"""

import os
import json
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import CAPITAConfig
from dataset import build_dataloaders
from model import CAPITAModel

# ─────────────────────────────────────────────────────────────────────────────
# LOGGER SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(output_dir: str, experiment_name: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{experiment_name}.log")

    logger = logging.getLogger("CAPITA")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """Saves/loads checkpoints, tracks best model, prunes old checkpoints."""

    def __init__(self, output_dir: str, keep_last_n: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.best_metric = float("inf")
        self.checkpoint_history: List[Path] = []

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer,
        scheduler,
        scaler,
        metrics: Dict,
        is_best: bool = False,
    ):
        state = {
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict() if scheduler else None,
            "scaler":     scaler.state_dict() if scaler else None,
            "metrics":    metrics,
        }

        # Save regular checkpoint
        ckpt_path = self.output_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        torch.save(state, ckpt_path)
        self.checkpoint_history.append(ckpt_path)

        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            shutil.copyfile(ckpt_path, best_path)

        # Prune old checkpoints
        while len(self.checkpoint_history) > self.keep_last_n:
            old = self.checkpoint_history.pop(0)
            if old.exists():
                old.unlink()

        return ckpt_path

    def load(self, path: str, model: nn.Module, optimizer=None,
             scheduler=None, scaler=None, device="cuda"):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state["model"])
        if optimizer and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        if scaler and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        return state["epoch"], state.get("metrics", {})

    def is_best(self, metric: float, lower_is_better: bool = True) -> bool:
        if lower_is_better:
            if metric < self.best_metric:
                self.best_metric = metric
                return True
        else:
            if metric > self.best_metric:
                self.best_metric = metric
                return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# METRICS TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class MetricsTracker:
    """Accumulates and averages metrics over batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._sums   = {}
        self._counts = {}

    def update(self, metrics: Dict, n: int = 1):
        for k, v in metrics.items():
            val = v.item() if isinstance(v, torch.Tensor) else float(v)
            if k not in self._sums:
                self._sums[k]   = 0.0
                self._counts[k] = 0
            self._sums[k]   += val * n
            self._counts[k] += n

    def average(self) -> Dict:
        return {
            k: self._sums[k] / max(self._counts[k], 1)
            for k in self._sums
        }

    def __str__(self) -> str:
        avg = self.average()
        return " | ".join(f"{k}: {v:.4f}" for k, v in sorted(avg.items()))


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER & SCHEDULER BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: CAPITAModel, cfg: CAPITAConfig):
    """
    Three parameter groups with separate learning rates:

    Group 1 - LoRA adapter params (inside TinyLlama):
        Only injected low-rank matrices are trainable inside the LLM.
        Use llm_learning_rate (2e-5): careful LLM adaptation.

    Group 2 - Intent projection + prefix projection:
        Bridge between CAPITA pipeline and TinyLlama embedding space.
        Use llm_learning_rate: stable LLM conditioning.

    Group 3 - CAPITA backbone (encoders, GNN, CTRM, classif. heads):
        Use main learning_rate (1e-4): faster learning for fresh modules.

    Note: get_peft_model() marks only LoRA weight matrices as
    requires_grad=True inside the LLM. Base weights are frozen automatically.
    """
    # LoRA params: only trainable rank matrices inside TinyLlama
    lora_params = [
        p for p in model.llm_head.llm.parameters() if p.requires_grad
    ]

    # Intent bridge: projection layers connecting CAPITA to TinyLlama
    bridge_params = (
        list(model.llm_head.intent_proj.parameters()) +
        list(model.llm_head.prefix_proj.parameters())
    )
    bridge_ids  = {id(p) for p in bridge_params}
    lora_ids    = {id(p) for p in lora_params}
    exclude_ids = bridge_ids | lora_ids

    # CAPITA backbone: encoders, GNN, CTRM, classification heads
    backbone_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in exclude_ids
    ]

    param_groups = [
        {"params": lora_params,     "lr": cfg.training.llm_learning_rate,  "name": "lora"},
        {"params": bridge_params,   "lr": cfg.training.llm_learning_rate,  "name": "bridge"},
        {"params": backbone_params, "lr": cfg.training.learning_rate,       "name": "backbone"},
    ]

    optimizer = AdamW(
        param_groups,
        weight_decay=cfg.training.weight_decay,
        eps=1e-8,
    )

    n_lora     = sum(p.numel() for p in lora_params)
    n_bridge   = sum(p.numel() for p in bridge_params)
    n_backbone = sum(p.numel() for p in backbone_params)
    total      = n_lora + n_bridge + n_backbone
    print(
        f"[Optimizer] LoRA: {n_lora/1e6:.2f}M | "
        f"Bridge: {n_bridge/1e6:.2f}M | "
        f"Backbone: {n_backbone/1e6:.2f}M | "
        f"Total trainable: {total/1e6:.2f}M"
    )
    return optimizer


def build_scheduler(optimizer, cfg: CAPITAConfig, steps_per_epoch: int):
    """Warmup + Cosine Annealing scheduler."""
    warmup_steps = cfg.training.warmup_epochs * steps_per_epoch
    total_steps  = cfg.training.num_epochs    * steps_per_epoch

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=cfg.training.min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    return scheduler


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: CAPITAModel,
    loader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    cfg: CAPITAConfig,
    epoch: int,
    logger: logging.Logger,
    global_step: int,
) -> (Dict, int):
    """Run one training epoch. Returns averaged metrics and updated global_step."""

    model.train()
    tracker  = MetricsTracker()
    device   = torch.device(cfg.training.device)
    grad_acc = cfg.training.gradient_accumulation_steps
    log_every = cfg.training.log_every_n_steps

    optimizer.zero_grad()
    t0 = time.time()

    for step, batch in enumerate(loader):

        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with autocast(dtype=torch.float16, enabled=True):
            losses = model(batch, generate=False)

        loss = losses["total_loss"] / grad_acc

        scaler.scale(loss).backward()

        # Gradient accumulation
        if (step + 1) % grad_acc == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        # Track metrics
        batch_size = batch["frames"].shape[0]
        detached   = {k: v.detach() for k, v in losses.items()}
        tracker.update(detached, n=batch_size)

        # Logging
        if (step + 1) % log_every == 0:
            avg      = tracker.average()
            elapsed  = time.time() - t0
            lr_main  = optimizer.param_groups[0]["lr"]
            lr_llm   = optimizer.param_groups[1]["lr"]

            logger.info(
                f"Epoch [{epoch}] Step [{step+1}/{len(loader)}] "
                f"Loss: {avg['total_loss']:.4f} | "
                f"QA: {avg.get('loss_qa', 0):.4f} | "
                f"LR_main: {lr_main:.2e} | LR_llm: {lr_llm:.2e} | "
                f"Time: {elapsed:.1f}s"
            )
            t0 = time.time()

    return tracker.average(), global_step


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: CAPITAModel,
    loader,
    cfg: CAPITAConfig,
    epoch: int,
    logger: logging.Logger,
) -> Dict:
    """Run validation. Returns averaged loss metrics."""

    model.eval()
    tracker = MetricsTracker()
    device  = torch.device(cfg.training.device)

    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with autocast(dtype=torch.float16, enabled=cfg.training.use_amp):
            losses = model(batch, generate=False)

        batch_size = batch["frames"].shape[0]
        detached   = {k: v.detach() for k, v in losses.items()}
        tracker.update(detached, n=batch_size)

    avg = tracker.average()
    logger.info(
        f"[VAL] Epoch [{epoch}] "
        f"Total: {avg['total_loss']:.4f} | "
        f"QA: {avg.get('loss_qa', 0):.4f}"
    )
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class CAPITATrainer:
    """
    Full training orchestrator.
    Usage:
        trainer = CAPITATrainer(cfg)
        trainer.train()
    """

    def __init__(self, cfg: CAPITAConfig, resume_from: Optional[str] = None):
        self.cfg = cfg

        # ── Setup ──────────────────────────────────────────────────────────
        torch.manual_seed(cfg.training.seed)
        torch.cuda.manual_seed_all(cfg.training.seed)
        self.device = torch.device(cfg.training.device)

        exp_dir = os.path.join(
            cfg.training.output_dir, cfg.training.experiment_name
        )
        self.logger      = setup_logger(exp_dir, cfg.training.experiment_name)
        self.ckpt_manager = CheckpointManager(
            exp_dir, keep_last_n=cfg.training.keep_last_n_checkpoints
        )

        self.logger.info("=" * 60)
        self.logger.info(f"CAPITA Training | Dataset: {cfg.data.dataset_name}")
        self.logger.info(f"Experiment: {cfg.training.experiment_name}")
        self.logger.info("=" * 60)

        # ── Data ───────────────────────────────────────────────────────────
        self.logger.info("Building DataLoaders...")
        self.train_loader, self.val_loader = build_dataloaders(cfg)

        # ── Model ──────────────────────────────────────────────────────────
        self.logger.info("Building CAPITA model...")
        self.model = CAPITAModel(cfg).to(self.device)

        total_params     = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters()
                               if p.requires_grad)
        self.logger.info(
            f"Model params: {total_params/1e6:.1f}M total, "
            f"{trainable_params/1e6:.1f}M trainable"
        )

        # ── Optimizer & Scheduler ─────────────────────────────────────────
        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler = build_scheduler(
            self.optimizer, cfg, len(self.train_loader)
        )
        self.scaler    = GradScaler(enabled=cfg.training.use_amp)

        # ── early stopping ─────────────────────────────────────────
        self.best_val_loss      = float('inf')
        self.early_stop_patience = 5   # stop after 5 val epochs with no improvement
        self.patience_counter    = 0
        self.early_stopped       = False

        # ── Optional: WandB ────────────────────────────────────────────────
        self.use_wandb = False
        if cfg.training.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=cfg.training.wandb_project,
                    name=cfg.training.experiment_name,
                    config={
                        "dataset":    cfg.data.dataset_name,
                        "num_frames": cfg.data.num_frames,
                        "batch_size": cfg.training.batch_size,
                        "lr":         cfg.training.learning_rate,
                        "epochs":     cfg.training.num_epochs,
                    },
                )
                self.use_wandb = True
                self.logger.info("WandB initialized.")
            except Exception as e:
                self.logger.warning(f"WandB init failed: {e}. Continuing without it.")

        # ── Resume ─────────────────────────────────────────────────────────
        self.start_epoch = 1
        if resume_from:
            self.logger.info(f"Resuming from {resume_from}")
            self.start_epoch, _ = self.ckpt_manager.load(
                resume_from, self.model, self.optimizer,
                self.scheduler, self.scaler, device=cfg.training.device,
            )
            self.start_epoch += 1

        # ── Training history ───────────────────────────────────────────────
        self.history: List[Dict] = []
        self.global_step = 0

    def train(self):
        cfg = self.cfg.training
        self.logger.info(f"Starting training for {cfg.num_epochs} epochs...")

        for epoch in range(self.start_epoch, cfg.num_epochs + 1):
            self.logger.info(f"\n{'='*50}\nEpoch {epoch}/{cfg.num_epochs}\n{'='*50}")

            # ── Train ──────────────────────────────────────────────────────
            train_metrics, self.global_step = train_one_epoch(
                self.model, self.train_loader,
                self.optimizer, self.scheduler, self.scaler,
                self.cfg, epoch, self.logger, self.global_step,
            )

            # ── Validate ───────────────────────────────────────────────────
            val_metrics = {}
            if epoch % cfg.val_every_n_epochs == 0:
                val_metrics = validate(
                    self.model, self.val_loader,
                    self.cfg, epoch, self.logger,
                )

           # ── History ───────────────────────────────────────────────────
            metrics = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            entry = {"epoch": epoch, **metrics}
            self.history.append(entry)

            train_loss = train_metrics.get("total_loss", 0)
            val_loss   = val_metrics.get("total_loss", train_loss)
            gap        = abs(val_loss - train_loss)
            is_best    = val_loss < self.best_val_loss

            # ── Save best model ────────────────────────────────────────────
            if is_best:
                self.best_val_loss    = val_loss
                self.patience_counter = 0
                best_path = os.path.join(
                    self.cfg.training.output_dir,
                    self.cfg.training.experiment_name,
                    "best_model.pth"
                )
                torch.save({
                    "epoch":   epoch,
                    "model":   self.model.state_dict(),
                    "metrics": entry,
                }, best_path)
                self.logger.info(
                    f"★ Best model saved | epoch={epoch} "
                    f"val={val_loss:.4f} train={train_loss:.4f} gap={gap:.4f}"
                )
            else:
                self.patience_counter += 1
                self.logger.info(
                    f"No improvement | val={val_loss:.4f} "
                    f"best={self.best_val_loss:.4f} "
                    f"patience={self.patience_counter}/{self.early_stop_patience}"
                )

            # ── Save last model (always overwrite) ─────────────────────────
            last_path = os.path.join(
                self.cfg.training.output_dir,
                self.cfg.training.experiment_name,
                "last_model.pth"
            )
            torch.save({
                "epoch":     epoch,
                "model":     self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler":    self.scaler.state_dict(),
                "metrics":   entry,
            }, last_path)

            # ── Dynamic early stopping ─────────────────────────────────────
            if self.patience_counter >= self.early_stop_patience:
                self.logger.info(
                    f"\n{'='*60}\n"
                    f"EARLY STOPPING at epoch {epoch}\n"
                    f"Val loss did not improve for {self.early_stop_patience} "
                    f"consecutive validation rounds.\n"
                    f"Best val loss: {self.best_val_loss:.4f} | "
                    f"Current: {val_loss:.4f}\n"
                    f"{'='*60}"
                )
                break

                
            # Save history JSON
            history_path = os.path.join(
                cfg.output_dir, cfg.experiment_name, "training_history.json"
            )
            with open(history_path, "w") as f:
                json.dump(self.history, f, indent=2)

            # ── WandB logging ──────────────────────────────────────────────
            if self.use_wandb:
                import wandb
                wandb.log(entry, step=epoch)

        self.logger.info("\n" + "="*50)
        self.logger.info("Training complete!")
        self.logger.info(f"Best val_loss: {self.ckpt_manager.best_metric:.4f}")
        self.logger.info("="*50)

        if self.use_wandb:
            import wandb
            wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="CAPITA Training")
    parser.add_argument(
        "--dataset", type=str, default="MultiUAV",
        choices=["MultiUAV", "Anti-UAV", "NPS"],
        help="Which dataset to train on"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None,
        help="Experiment name (default: capita_<dataset>)"
    )
    parser.add_argument(
        "--no_wandb", action="store_true",
        help="Disable WandB logging"
    )
    args = parser.parse_args()

    # ── Build config ────────────────────────────────────────────────────────
    from dataclasses import replace
    cfg = CAPITAConfig()
    cfg.data.dataset_name = args.dataset

    if args.epochs:
        cfg.training.num_epochs = args.epochs
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.experiment_name:
        cfg.training.experiment_name = args.experiment_name
    else:
        cfg.training.experiment_name = f"capita_{args.dataset.lower().replace('-', '_')}"
    if args.no_wandb:
        cfg.training.use_wandb = False

    # ── Launch trainer ──────────────────────────────────────────────────────
    trainer = CAPITATrainer(cfg, resume_from=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()