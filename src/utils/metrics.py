"""COCO mAP@[0.5:0.95] evaluation for validation validation and model checkpointing."""

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_coco_map(model, data_loader, device):
    """Computes mAP@[0.5:0.95] and mAP@0.5 on validation dataloader."""
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

    if not coco_dt:
        return 0.0, 0.0

    gt_api = COCO()
    gt_api.dataset = coco_gt
    gt_api.createIndex()

    dt_api = gt_api.loadRes(coco_dt)
    coco_eval = COCOeval(gt_api, dt_api, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map_50_95 = coco_eval.stats[0]
    map_50 = coco_eval.stats[1]
    return map_50_95, map_50