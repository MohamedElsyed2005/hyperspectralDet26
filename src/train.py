"""Train the 16-band hyperspectral detector and track mAP@[0.5:0.95]."""

import argparse
import random
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from . import config
from .dataset import HyperspectralDetDataset, collate_fn
from .model import build_model
from .utils.metrics import evaluate_coco_metrics
from .utils.logging_utils import (
    setup_logger,
    save_config_snapshot,
    MetricsLogger,
    plot_training_curves,
)

# Individual Faster R-CNN loss components we want visibility into every epoch.
LOSS_COMPONENT_KEYS = [
    "loss_classifier",     # ROI head classification loss
    "loss_box_reg",        # ROI head box regression loss
    "loss_objectness",     # RPN objectness loss
    "loss_rpn_box_reg",    # RPN box regression loss
]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


from torch.cuda.amp import GradScaler, autocast


def train_one_epoch(model, optimizer, loader, device, epoch, scaler, accum_steps=4):
    """Runs one training epoch. Returns (avg_total_loss, avg_component_losses dict).

    Per-batch numbers still stream to the tqdm progress bar in the terminal as
    before; only the epoch-level averages get handed back for logging, so the
    saved log file never sees per-batch noise.
    """
    model.train()
    running_loss = 0.0
    component_sums = {k: 0.0 for k in LOSS_COMPONENT_KEYS}
    n_batches = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")

    for i, (images, targets) in enumerate(pbar):
        # Skip batch if any target has 0 boxes
        if any(t["boxes"].numel() == 0 for t in targets):
            continue

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast():
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        batch_loss = loss.item() * accum_steps
        running_loss += batch_loss
        for k in LOSS_COMPONENT_KEYS:
            if k in loss_dict:
                component_sums[k] += loss_dict[k].item()
        n_batches += 1

        pbar.set_postfix(loss=f"{batch_loss:.4f}")

    n_batches = max(1, n_batches)
    avg_loss = running_loss / n_batches
    avg_components = {k: v / n_batches for k, v in component_sums.items()}
    return avg_loss, avg_components


@torch.no_grad()
def compute_val_loss(model, loader, device):
    """Computes validation loss using the same loss heads as training.

    torchvision detection models only return a loss dict while in train()
    mode. We temporarily switch to train() (under no_grad, so no weights are
    updated) but force every BatchNorm layer back into eval mode first, so
    validation data never contaminates the model's running BN statistics.
    """
    was_training = model.training
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()

    running_loss = 0.0
    component_sums = {k: 0.0 for k in LOSS_COMPONENT_KEYS}
    n_batches = 0

    for images, targets in loader:
        if any(t["boxes"].numel() == 0 for t in targets):
            continue
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        running_loss += loss.item()
        for k in LOSS_COMPONENT_KEYS:
            if k in loss_dict:
                component_sums[k] += loss_dict[k].item()
        n_batches += 1

    if not was_training:
        model.eval()
    else:
        # Restore normal train-mode behavior for BatchNorm layers (we forced
        # them into eval() above purely to avoid updating running stats on
        # validation data); everything else was already in train mode.
        model.train()

    n_batches = max(1, n_batches)
    avg_loss = running_loss / n_batches
    avg_components = {k: v / n_batches for k, v in component_sums.items()}
    return avg_loss, avg_components


def assert_run_dir_is_fresh(run_ckpt_dir: Path, logger):
    """Guards against accidentally reusing a directory that already holds
    checkpoints (e.g. a repeated --run-name), which could otherwise silently
    overwrite a previous run's best.pt/last.pt. This is a one-time check at
    the start of a run; within a run, updating this run's own last.pt/best.pt
    every epoch is expected and safe."""
    existing = list(run_ckpt_dir.glob("*.pt"))
    if existing:
        raise FileExistsError(
            f"{run_ckpt_dir} already contains checkpoint(s) {[p.name for p in existing]}. "
            f"Refusing to start a run that could overwrite them -- pick a different "
            f"--run-name or remove that directory first."
        )
    logger.info(f"Confirmed {run_ckpt_dir} is a fresh, empty checkpoint directory for this run.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Optional custom name for this run's log/checkpoint subdirectory."
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Run setup: every run gets its own timestamped directory so nothing
    # from a previous run (including checkpoints) is ever touched.
    # -----------------------------------------------------------------
    run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = config.OUTPUT_DIR / "runs" / run_id
    run_ckpt_dir = config.CHECKPOINT_DIR / "runs" / run_id
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_log_dir / "train.log")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Existing checkpoints under {config.CHECKPOINT_DIR} (e.g. best.pt/last.pt) "
                f"will NOT be touched. New checkpoints go to: {run_ckpt_dir}")
    assert_run_dir_is_fresh(run_ckpt_dir, logger)

    # Save config + system info snapshot for reproducibility.
    snapshot_path = run_log_dir / "config_snapshot.json"
    save_config_snapshot(config, snapshot_path, extra=vars(args))
    logger.info(f"Saved config/system snapshot -> {snapshot_path}")

    metrics_logger = MetricsLogger(run_log_dir)

    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        full_dataset = HyperspectralDetDataset()
        logger.info(f"Total training samples found: {len(full_dataset)}")

        n_val = max(1, int(len(full_dataset) * config.VAL_SPLIT))
        n_train = len(full_dataset) - n_val
        train_ds, val_ds = random_split(
            full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(config.SEED)
        )

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn
        )

        model = build_model().to(device)
        if args.resume:
            model.load_state_dict(torch.load(args.resume, map_location=device))
            logger.info(f"Resumed weights from {args.resume}")

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA
        )

        best_map = 0.0
        scaler = GradScaler()

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            train_loss, train_components = train_one_epoch(
                model, optimizer, train_loader, device, epoch, scaler, accum_steps=4
            )
            scheduler.step()

            val_loss, val_components = compute_val_loss(model, val_loader, device)
            metrics = evaluate_coco_metrics(model, val_loader, device)
            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} "
                f"(cls={train_components['loss_classifier']:.4f}, "
                f"box_reg={train_components['loss_box_reg']:.4f}, "
                f"obj={train_components['loss_objectness']:.4f}, "
                f"rpn_box_reg={train_components['loss_rpn_box_reg']:.4f}) | "
                f"val_loss={val_loss:.4f} | "
                f"mAP@[.5:.95]={metrics['mAP_50_95']:.4f} | AP50={metrics['AP50']:.4f} | "
                f"AP75={metrics['AP75']:.4f} | precision~={metrics['mean_precision_iou50']:.4f} | "
                f"recall(AR@100)={metrics['AR_max100']:.4f} | "
                f"detections={metrics['num_detections']}/{metrics['num_gt_boxes']}_gt | "
                f"time={epoch_time:.1f}s"
            )
            if metrics["num_detections"] == 0:
                logger.warning(
                    f"Epoch {epoch}: zero detections survived the score threshold on the "
                    f"validation set -- mAP is 0 because there is nothing to score against "
                    f"the {metrics['num_gt_boxes']} ground-truth boxes, not because of poor "
                    f"localization. Watch loss_objectness; if it stays flat, the RPN may need "
                    f"more epochs, a different LR, or a warmup before detections appear."
                )

            metrics_logger.log(
                epoch=epoch,
                train_loss=train_loss,
                train_loss_classifier=train_components["loss_classifier"],
                train_loss_box_reg=train_components["loss_box_reg"],
                train_loss_objectness=train_components["loss_objectness"],
                train_loss_rpn_box_reg=train_components["loss_rpn_box_reg"],
                val_loss=val_loss,
                val_loss_classifier=val_components["loss_classifier"],
                val_loss_box_reg=val_components["loss_box_reg"],
                val_loss_objectness=val_components["loss_objectness"],
                val_loss_rpn_box_reg=val_components["loss_rpn_box_reg"],
                mAP_50_95=metrics["mAP_50_95"],
                AP50=metrics["AP50"],
                AP75=metrics["AP75"],
                AR_max100=metrics["AR_max100"],
                mean_precision_iou50=metrics["mean_precision_iou50"],
                num_detections=metrics["num_detections"],
                num_gt_boxes=metrics["num_gt_boxes"],
                lr=optimizer.param_groups[0]["lr"],
                epoch_time_sec=round(epoch_time, 2),
            )

            # Checkpoints are written only inside this run's own timestamped
            # directory (run_ckpt_dir), so the pre-existing top-level
            # checkpoints/best.pt and checkpoints/last.pt are never touched.
            torch.save(model.state_dict(), run_ckpt_dir / "last.pt")
            if metrics["mAP_50_95"] > best_map:
                best_map = metrics["mAP_50_95"]
                torch.save(model.state_dict(), run_ckpt_dir / "best.pt")
                logger.info(f"--> Saved new BEST checkpoint for this run (mAP = {best_map:.4f})")

        logger.info("Training completed successfully.")

    except Exception:
        logger.error("Training crashed. Traceback:\n" + traceback.format_exc())
        raise

    finally:
        # Always attempt to plot whatever metrics were collected, even after
        # a crash -- prior epochs' logs/metrics are already safely on disk.
        plots_dir = run_log_dir / "plots"
        written = plot_training_curves(metrics_logger.rows, plots_dir, logger=logger)
        if written:
            logger.info(f"Saved {len(written)} plot(s) -> {plots_dir}")
        logger.info(f"Run artifacts: log={run_log_dir / 'train.log'}, "
                     f"metrics_csv={metrics_logger.csv_path}, metrics_json={metrics_logger.json_path}, "
                     f"config_snapshot={snapshot_path}, checkpoints={run_ckpt_dir}")


if __name__ == "__main__":
    main()
