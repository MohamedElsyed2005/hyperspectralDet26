"""COCO mAP@[0.5:0.95] evaluation for validation validation and model checkpointing."""

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .. import config

# Index map into COCOeval.stats (standard 12-value COCO summary), see
# https://github.com/cocodataset/cocoapi for the canonical ordering.
_STAT_KEYS = [
    "mAP_50_95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR_max1", "AR_max10", "AR_max100", "AR_small", "AR_medium", "AR_large",
]


@torch.no_grad()
def _build_coco_structures(model, data_loader, device):
    """Runs the model over `data_loader` and builds COCO-format ground-truth
    and detection structures. Shared by `evaluate_coco_metrics` and
    `evaluate_per_class_ap` so both stay consistent and the (potentially
    slow) forward pass over the val set only happens once per call site.
    """
    model.eval()

    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": str(i)} for i in range(18)]
    }
    coco_dt = []
    ann_id = 1

    for img_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for i, (target, output) in enumerate(zip(targets, outputs)):
            current_img_id = img_idx * len(images) + i + 1
            _, h, w = images[i].shape
            coco_gt["images"].append({
                "id": current_img_id,
                "height": h,
                "width": w
            })

            # Ground truth
            gt_boxes = target["boxes"].cpu().numpy()
            gt_labels = target["labels"].cpu().numpy()
            for b, l in zip(gt_boxes, gt_labels):
                xmin, ymin, xmax, ymax = b
                coco_gt["annotations"].append({
                    "id": ann_id,
                    "image_id": current_img_id,
                    "category_id": int(l),
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "area": (xmax - xmin) * (ymax - ymin),
                    "iscrowd": 0
                })
                ann_id += 1

            # Predictions
            dt_boxes = output["boxes"].cpu().numpy()
            dt_scores = output["scores"].cpu().numpy()
            dt_labels = output["labels"].cpu().numpy()
            for b, s, l in zip(dt_boxes, dt_scores, dt_labels):
                xmin, ymin, xmax, ymax = b
                coco_dt.append({
                    "image_id": current_img_id,
                    "category_id": int(l),
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "score": float(s)
                })

    return coco_gt, coco_dt


def _run_coco_eval(coco_gt: dict, coco_dt: list):
    """Builds COCO/COCOeval objects and runs evaluate+accumulate+summarize.
    Returns None if there are no detections to score."""
    if not coco_dt:
        return None

    gt_api = COCO()
    gt_api.dataset = coco_gt
    gt_api.createIndex()

    dt_api = gt_api.loadRes(coco_dt)
    coco_eval = COCOeval(gt_api, dt_api, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def evaluate_coco_metrics(model, data_loader, device):
    """Computes COCO mAP/AP/AR metrics on a validation dataloader.

    Returns a dict with (at least, when detections exist):
        mAP_50_95, AP50, AP75, AR_max100 (used here as a "recall" proxy),
        mean_precision_iou50 (mean precision at IoU=0.50 across all
        categories/recall thresholds, from the raw COCOeval precision array
        -- an approximate, threshold-averaged precision since COCO detection
        does not define a single scalar "precision" the way classification does),
        num_detections (total predictions kept for eval across the val set),
        and num_gt_boxes (total ground-truth boxes in the val set) -- these
        two counts make it possible to tell "zero detections above the score
        threshold" (mAP=0 because num_detections=0) apart from "detections
        exist but don't match well" (mAP=0 with num_detections>0).
    All AP/AR values default to 0.0 if there are no predictions to evaluate.
    """
    coco_gt, coco_dt = _build_coco_structures(model, data_loader, device)

    empty_metrics = {k: 0.0 for k in _STAT_KEYS}
    empty_metrics["mean_precision_iou50"] = 0.0
    empty_metrics["num_detections"] = 0
    empty_metrics["num_gt_boxes"] = len(coco_gt["annotations"])
    if not coco_dt:
        return empty_metrics

    coco_eval = _run_coco_eval(coco_gt, coco_dt)

    metrics = {k: float(v) for k, v in zip(_STAT_KEYS, coco_eval.stats)}

    # Approximate scalar "precision": mean of the precision array at the
    # IoU=0.50 threshold, across all recall thresholds/categories/areas
    # that have valid (non -1) values. This is the closest COCO analogue
    # to a single "precision" number and is reported alongside AP/AR.
    precision_arr = coco_eval.eval.get("precision") if coco_eval.eval else None
    if precision_arr is not None:
        p_iou50 = precision_arr[0]  # index 0 == IoU threshold 0.50
        valid = p_iou50[p_iou50 > -1]
        metrics["mean_precision_iou50"] = float(valid.mean()) if valid.size > 0 else 0.0
    else:
        metrics["mean_precision_iou50"] = 0.0

    metrics["num_detections"] = len(coco_dt)
    metrics["num_gt_boxes"] = len(coco_gt["annotations"])

    return metrics


def evaluate_coco_map(model, data_loader, device):
    """Backward-compatible wrapper: returns (mAP@[0.5:0.95], mAP@0.5) as a tuple."""
    metrics = evaluate_coco_metrics(model, data_loader, device)
    return metrics["mAP_50_95"], metrics["AP50"]


def evaluate_per_class_ap(model, data_loader, device, class_names=None):
    """Per-category AP@[.5:.95] and AP50, sorted worst-to-best.

    Useful for this competition specifically because several class pairs are
    designed to be visually near-identical (apple/apple_plastic,
    egg/egg_plastic/egg_wood, banana/banana_plastic, orange/orange_plastic,
    car/car_toy) and are only separable using the 16-band spectral signal.
    A class sitting far below the mean AP here -- especially one half of one
    of those pairs -- is a concrete signal that the model is leaning on
    shape/texture and confusing the pair, rather than using spectral cues.

    Returns a list of dicts, one per category that had at least one
    ground-truth box in this dataloader, each with:
        class_id (0-based, matching class.txt / submission.csv),
        class_name, AP_50_95, AP50, num_gt.
    Sorted ascending by AP_50_95 (worst classes first).
    """
    class_names = class_names or config.ID_TO_CLASS

    coco_gt, coco_dt = _build_coco_structures(model, data_loader, device)
    num_gt_per_class = {}
    for ann in coco_gt["annotations"]:
        cat = ann["category_id"]
        num_gt_per_class[cat] = num_gt_per_class.get(cat, 0) + 1

    if not coco_dt:
        rows = [
            {
                "class_id": cat - 1,
                "class_name": class_names.get(cat - 1, str(cat - 1)),
                "AP_50_95": 0.0,
                "AP50": 0.0,
                "num_gt": n,
            }
            for cat, n in num_gt_per_class.items()
        ]
        return sorted(rows, key=lambda r: r["AP_50_95"])

    coco_eval = _run_coco_eval(coco_gt, coco_dt)
    precision = coco_eval.eval["precision"]  # shape (T, R, K, A, M)
    cat_ids = coco_eval.params.catIds  # length K, in the same order as axis 2

    rows = []
    for k, cat_id in enumerate(cat_ids):
        # AP@[.5:.95]: mean over all IoU thresholds (T) and recall levels (R),
        # area range "all" (A=0), max dets = last (M=-1), valid entries only.
        p_all = precision[:, :, k, 0, -1]
        valid_all = p_all[p_all > -1]
        ap_50_95 = float(valid_all.mean()) if valid_all.size > 0 else 0.0

        # AP50: same but IoU threshold fixed at 0.50 (T index 0).
        p_50 = precision[0, :, k, 0, -1]
        valid_50 = p_50[p_50 > -1]
        ap_50 = float(valid_50.mean()) if valid_50.size > 0 else 0.0

        num_gt = num_gt_per_class.get(cat_id, 0)
        if num_gt == 0:
            continue  # nothing to score this class against in this val split -- skip, not a real 0

        class_id = cat_id - 1  # undo the +1 background shift used everywhere else
        rows.append({
            "class_id": class_id,
            "class_name": class_names.get(class_id, str(class_id)),
            "AP_50_95": ap_50_95,
            "AP50": ap_50,
            "num_gt": num_gt,
        })

    return sorted(rows, key=lambda r: r["AP_50_95"])


def format_per_class_table(rows) -> str:
    """Pretty-prints the output of `evaluate_per_class_ap` as a fixed-width table."""
    header = f"{'class_id':>8}  {'class_name':<16} {'AP@.5:.95':>10} {'AP50':>8} {'num_gt':>7}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['class_id']:>8}  {r['class_name']:<16} "
            f"{r['AP_50_95']:>10.4f} {r['AP50']:>8.4f} {r['num_gt']:>7}"
        )
    return "\n".join(lines)
