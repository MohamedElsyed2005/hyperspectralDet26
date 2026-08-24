"""Print a per-class AP breakdown on the validation split for a trained checkpoint.

Usage:
    python -m src.eval_per_class --checkpoint checkpoints/runs/exp_001_cont_v2/best.pt

Uses the exact same train/val split as training (same seed, same helper), so
this is scored on genuinely held-out data, not the training set.
"""

import argparse

import torch
from torch.utils.data import DataLoader, Subset

from . import config
from .dataset import HyperspectralDetDataset, collate_fn
from .model import build_model
from .split import get_train_val_indices
from .utils.metrics import evaluate_per_class_ap, format_per_class_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "best.pt"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_base = HyperspectralDetDataset(transforms=None)
    _, val_indices = get_train_val_indices(len(val_base), config.VAL_SPLIT, config.SEED)
    val_ds = Subset(val_base, val_indices)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    print(f"Evaluating on {len(val_ds)} held-out validation images")

    model = build_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    rows = evaluate_per_class_ap(model, val_loader, device)
    print()
    print(format_per_class_table(rows))

    if rows:
        mean_ap = sum(r["AP_50_95"] for r in rows) / len(rows)
        print(f"\nUnweighted mean over {len(rows)} classes present in val: AP@.5:.95 = {mean_ap:.4f}")
        worst = rows[:3]
        print("Lowest-scoring classes (check these against their real/fake counterpart):")
        for r in worst:
            print(f"  - {r['class_name']} (id={r['class_id']}): AP@.5:.95={r['AP_50_95']:.4f}, num_gt={r['num_gt']}")


if __name__ == "__main__":
    main()
