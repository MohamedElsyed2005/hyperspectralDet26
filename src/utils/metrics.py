"""COCO mAP@[0.5:0.95] evaluation for validation validation and model checkpointing."""

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Index map into COCOeval.stats (standard 12-value COCO summary), see
# https://github.com/cocodataset/cocoapi for the canonical ordering.
_STAT_KEYS = [
    "mAP_50_95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR_max1", "AR_max10", "AR_max100", "AR_small", "AR_medium", "AR_large",
]


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
    model.eval()

    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": str(i)} for i in range(18)]
    }
    coco_dt = []
    ann_id = 1

    with torch.no_grad():
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

    empty_metrics = {k: 0.0 for k in _STAT_KEYS}
    empty_metrics["mean_precision_iou50"] = 0.0
    empty_metrics["num_detections"] = 0
    empty_metrics["num_gt_boxes"] = len(coco_gt["annotations"])
    if not coco_dt:
        return empty_metrics

    gt_api = COCO()
    gt_api.dataset = coco_gt
    gt_api.createIndex()

    dt_api = gt_api.loadRes(coco_dt)
    coco_eval = COCOeval(gt_api, dt_api, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

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