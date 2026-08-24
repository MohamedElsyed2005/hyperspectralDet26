"""Per-class error diagnostic: categorizes detection failure modes into
Missed (Recall), Misclassified (Confusion), Low Confidence, or True Positive.

Usage:
    python -m src.error_analysis --checkpoint checkpoints/runs/exp_002/best.pt
"""

import argparse
from pathlib import Path
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_iou
from tqdm import tqdm

from . import config
from .dataset import HyperspectralDetDataset, collate_fn
from .split import get_train_val_indices
from .model import build_model


@torch.no_grad()
def run_error_analysis(checkpoint_path: Path, score_thresh: float, iou_thresh: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Device: {device} | Score Threshold: {score_thresh} | IoU Match Threshold: {iou_thresh}")

    # Build validation split
    base_ds = HyperspectralDetDataset(transforms=None)
    _, val_indices = get_train_val_indices(len(base_ds), config.VAL_SPLIT, config.SEED)
    val_ds = Subset(base_ds, val_indices)

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )

    # Load Model
    model = build_model().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    class_stats = {
        cls_name: {"total_gt": 0, "tp": 0, "low_conf": 0, "misclassified": 0, "missed": 0}
        for cls_name in config.CLASSES
    }
    confusion_pairs = defaultdict(int)

    for images, targets in tqdm(val_loader, desc="Running Diagnostic"):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            gt_boxes = target["boxes"].to(device)
            gt_labels = target["labels"].to(device)

            pred_boxes = output["boxes"]
            pred_scores = output["scores"]
            pred_labels = output["labels"]

            if len(gt_boxes) == 0:
                continue

            # If the model proposed no boxes in the whole image
            if len(pred_boxes) == 0:
                for l in gt_labels:
                    cls_name = config.CLASSES[l.item() - 1]
                    class_stats[cls_name]["total_gt"] += 1
                    class_stats[cls_name]["missed"] += 1
                continue

            ious = box_iou(gt_boxes, pred_boxes)

            for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                cls_name = config.CLASSES[gt_label.item() - 1]
                class_stats[cls_name]["total_gt"] += 1

                matched_ious = ious[i]
                best_iou_val, best_pred_idx = torch.max(matched_ious, dim=0)

                if best_iou_val.item() < iou_thresh:
                    class_stats[cls_name]["missed"] += 1
                else:
                    pred_lbl = pred_labels[best_pred_idx].item()
                    pred_score = pred_scores[best_pred_idx].item()

                    if pred_lbl == gt_label.item():
                        if pred_score >= score_thresh:
                            class_stats[cls_name]["tp"] += 1
                        else:
                            class_stats[cls_name]["low_conf"] += 1
                    else:
                        class_stats[cls_name]["misclassified"] += 1
                        pred_name = config.CLASSES[pred_lbl - 1] if 0 < pred_lbl <= len(config.CLASSES) else "bg"
                        confusion_pairs[(cls_name, pred_name)] += 1

    # Print Summary Table
    header = f"{'Class':<18} {'GT':>5} {'TP':>12} {'Missed(IoU<.5)':>16} {'Wrong Class':>14} {'Low Conf':>12}"
    print("\n" + "=" * len(header))
    print("DETECTION FAILURE MODE BREAKDOWN (Validation Split)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for cls_name, counts in class_stats.items():
        total = counts["total_gt"]
        if total == 0:
            continue
        tp_str = f"{counts['tp']} ({counts['tp']/total*100:.1f}%)"
        miss_str = f"{counts['missed']} ({counts['missed']/total*100:.1f}%)"
        miscls_str = f"{counts['misclassified']} ({counts['misclassified']/total*100:.1f}%)"
        lowconf_str = f"{counts['low_conf']} ({counts['low_conf']/total*100:.1f}%)"
        
        print(f"{cls_name:<18} {total:>5} {tp_str:>12} {miss_str:>16} {miscls_str:>14} {lowconf_str:>12}")

    if confusion_pairs:
        print("\n" + "=" * 50)
        print("Top Confusion Pairs (Ground Truth -> Predicted):")
        print("=" * 50)
        sorted_confusion = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
        for (gt_c, pred_c), cnt in sorted_confusion:
            print(f"  * {gt_c:<15} -> {pred_c:<15} : {cnt} instances")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(config.CHECKPOINT_DIR / "runs" / "exp_002" / "best.pt"),
        help="Path to checkpoint .pt file"
    )
    parser.add_argument("--score-thresh", type=float, default=config.SCORE_THRESH)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    args = parser.parse_args()

    run_error_analysis(Path(args.checkpoint), args.score_thresh, args.iou_thresh)


if __name__ == "__main__":
    main()