"""Inference script for test (Phase 1) and ranking sets (Phase 2)."""

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms
from tqdm import tqdm

from . import config
from .dataset import HyperspectralInferenceDataset
from .model import build_model
from .submission import append_ranking_predictions, write_submission, validate_submission

SPLIT_DIRS = {
    "test": config.TEST_IMG_DIR,
    "ranking": config.RANKING_IMG_DIR,
}


@torch.no_grad()
def run_inference(model, loader, device, score_thresh, nms_thresh, max_dets):
    model.eval()
    rows = []
    for images, image_ids in tqdm(loader, desc="Inference"):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for image_id, output in zip(image_ids, outputs):
            boxes = output["boxes"]
            scores = output["scores"]
            labels = output["labels"]

            keep = scores >= score_thresh
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            if boxes.numel() > 0:
                keep_idx = nms(boxes, scores, nms_thresh)
                boxes, scores, labels = boxes[keep_idx], scores[keep_idx], labels[keep_idx]

            if boxes.numel() > 0 and len(boxes) > max_dets:
                top = torch.topk(scores, max_dets).indices
                boxes, scores, labels = boxes[top], scores[top], labels[top]

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                rows.append({
                    "image_id": image_id,
                    "class_id": int(label.item()) - 1,  # Undo background shift
                    "confidence": round(float(score.item()), 4),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "best.pt"))
    parser.add_argument("--split", choices=["test", "ranking"], default="test")
    parser.add_argument("--out", type=str, default=str(config.OUTPUT_DIR / "submission.csv"))
    parser.add_argument("--append-to", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--score-thresh", type=float, default=config.SCORE_THRESH)
    parser.add_argument("--nms-thresh", type=float, default=config.NMS_THRESH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    dataset = HyperspectralInferenceDataset(SPLIT_DIRS[args.split])
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=lambda batch: tuple(zip(*batch))
    )
    print(f"Running inference on {len(dataset)} images from {SPLIT_DIRS[args.split]}")

    rows = run_inference(
        model, loader, device, args.score_thresh, args.nms_thresh, config.MAX_DETECTIONS_PER_IMAGE
    )

    if args.split == "ranking" and args.append_to:
        df = append_ranking_predictions(args.append_to, rows, args.out)
    else:
        df = write_submission(rows, args.out)

    validate_submission(args.out)


if __name__ == "__main__":
    main()