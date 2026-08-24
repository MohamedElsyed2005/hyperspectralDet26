"""Run logging, config/system snapshots, metrics tracking, and plotting utilities.

Designed to be crash-safe: metrics are flushed to disk after every epoch (not
just at the end of training), and the saved log file only ever receives
per-epoch summary lines -- tqdm's own progress bars keep printing to the
terminal exactly as before, but never get written into the log file.
"""

import csv
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
def setup_logger(log_file, name="train", console: bool = True) -> logging.Logger:
    """Creates a logger that writes clean per-epoch lines to `log_file`.

    Only explicit logger.info/warning/error calls land in the file -- tqdm
    progress bars write directly to stderr and are never routed through this
    logger, so the saved log stays readable even after thousands of batches.
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers if setup_logger is called twice
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# System / reproducibility snapshot
# ---------------------------------------------------------------------------
def _get_gpu_info():
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    info = {
        "cuda_available": True,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info["devices"].append({
            "index": i,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
            "multi_processor_count": props.multi_processor_count,
        })

    # Optional: driver version via nvidia-smi, best-effort only.
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            info["driver_version"] = out.stdout.strip().splitlines()[0]
    except Exception:
        pass

    return info


def get_system_info() -> dict:
    """Collects Python/torch/OS/GPU info useful for reproducing a training run."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "torchvision_version": _safe_import_version("torchvision"),
        "cuda": _get_gpu_info(),
    }


def _safe_import_version(module_name):
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "unknown"


def save_config_snapshot(config_module, out_path, extra: dict = None):
    """Dumps every upper-case, JSON-serializable constant from a config module,
    plus system info and any extra run args, to a single JSON file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = {}
    for key in dir(config_module):
        if key.startswith("_") or not key.isupper():
            continue
        value = getattr(config_module, key)
        try:
            json.dumps(value)
            cfg[key] = value
        except TypeError:
            cfg[key] = str(value)  # e.g. Path objects

    snapshot = {
        "config": cfg,
        "system_info": get_system_info(),
    }
    if extra:
        snapshot["run_args"] = extra

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    return snapshot


# ---------------------------------------------------------------------------
# Per-epoch metrics tracking (CSV + JSON, flushed every epoch)
# ---------------------------------------------------------------------------
class MetricsLogger:
    """Accumulates one dict of scalars per epoch and writes CSV + JSON after
    every `log()` call, so a crash mid-training never loses prior epochs."""

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "metrics.csv"
        self.json_path = self.out_dir / "metrics.json"
        self.rows = []
        self._fieldnames = []

    def log(self, **metrics):
        """Records one epoch's metrics and immediately flushes to disk."""
        self.rows.append(metrics)
        for k in metrics:
            if k not in self._fieldnames:
                self._fieldnames.append(k)
        self._flush()
        return metrics

    def _flush(self):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.rows, f, indent=2)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_training_curves(metrics_rows, out_dir, logger=None):
    """Best-effort plotting of loss/metric curves. Never raises -- a plotting
    failure should not lose or interrupt training results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        if logger:
            logger.warning(f"Skipping plots: matplotlib unavailable ({e})")
        return []

    if not metrics_rows:
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [r.get("epoch") for r in metrics_rows]
    written = []

    def _series(key):
        return [r.get(key) for r in metrics_rows]

    def _save(fig_name, plot_fn, title):
        try:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            plot_fn(ax)
            ax.set_xlabel("Epoch")
            ax.set_title(title)
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            path = out_dir / fig_name
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)
        except Exception as e:
            if logger:
                logger.warning(f"Could not generate plot '{fig_name}': {e}")

    # 1. Train vs. val loss
    if any(r.get("train_loss") is not None for r in metrics_rows):
        _save(
            "loss_curve.png",
            lambda ax: (
                ax.plot(epochs, _series("train_loss"), label="Train loss", marker="o"),
                ax.plot(epochs, _series("val_loss"), label="Val loss", marker="o")
                if any(v is not None for v in _series("val_loss")) else None,
            ),
            "Training / Validation Loss",
        )

    # 2. Faster R-CNN loss components
    component_keys = [
        "train_loss_classifier", "train_loss_box_reg",
        "train_loss_objectness", "train_loss_rpn_box_reg",
    ]
    if any(any(r.get(k) is not None for r in metrics_rows) for k in component_keys):
        _save(
            "loss_components.png",
            lambda ax: [
                ax.plot(epochs, _series(k), label=k.replace("train_", ""), marker="o")
                for k in component_keys if any(v is not None for v in _series(k))
            ],
            "Faster R-CNN Loss Components (Train)",
        )

    # 3. mAP / AP metrics
    map_keys = ["mAP_50_95", "AP50", "AP75"]
    if any(any(r.get(k) is not None for r in metrics_rows) for k in map_keys):
        _save(
            "map_curve.png",
            lambda ax: [
                ax.plot(epochs, _series(k), label=k, marker="o")
                for k in map_keys if any(v is not None for v in _series(k))
            ],
            "Validation mAP / AP",
        )

    # 4. Precision / recall proxies
    pr_keys = ["mean_precision_iou50", "AR_max100"]
    if any(any(r.get(k) is not None for r in metrics_rows) for k in pr_keys):
        _save(
            "precision_recall_curve.png",
            lambda ax: [
                ax.plot(epochs, _series(k), label=k, marker="o")
                for k in pr_keys if any(v is not None for v in _series(k))
            ],
            "Approx. Precision / Recall",
        )

    return written
