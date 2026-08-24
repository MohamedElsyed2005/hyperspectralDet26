"""Geometry- and intensity-safe data augmentation for 16-band hyperspectral detection.

Only spatial transforms (flips, 90-degree rotations, small affine jitter) and a
single *shared* intensity scale across all bands are used. Per-band color/contrast
jitter is deliberately avoided: the whole point of the 16-band signal in this
competition is telling materially-similar look-alikes apart (apple vs.
apple_plastic, egg vs. egg_plastic vs. egg_wood, ...) using the *shape* of each
pixel's reflectance curve across bands. Any transform that perturbs bands
independently (per-channel brightness/contrast/hue jitter) would corrupt exactly
that cue, so this module intentionally sticks to transforms that treat all 16
bands identically.
"""

import random

import numpy as np
import torch

import albumentations as A


def _build_geometric_pipeline(min_visibility: float = 0.2) -> A.Compose:
    """Spatial-only pipeline -- safe to apply identically across all 16 bands."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-10, 10),
                p=0.5,
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=min_visibility,
        ),
    )


class DetectionAugmentation:
    """Callable with the `(image, target) -> (image, target)` signature that
    HyperspectralDetDataset already expects (see `self.transforms(image, target)`
    in dataset.py) -- so wiring this in is a one-line change in train.py, no
    changes to dataset.py needed.

    `image` is a (C=16, H, W) float32 tensor in [0, 1].
    `target` holds `boxes` (N, 4) in pascal_voc pixel coords and `labels` (N,).

    Boxes that get cropped down to near-nothing by the affine jitter are
    dropped (via min_visibility) instead of being kept as degenerate/near-zero
    area boxes that would otherwise poison the loss.
    """

    def __init__(self, min_visibility: float = 0.2, brightness_jitter: float = 0.15):
        self.pipeline = _build_geometric_pipeline(min_visibility=min_visibility)
        self.brightness_jitter = brightness_jitter

    def __call__(self, image: torch.Tensor, target: dict):
        img_np = image.permute(1, 2, 0).numpy()  # (C,H,W) -> (H,W,C) for albumentations
        boxes = target["boxes"].numpy().tolist()
        labels = target["labels"].numpy().tolist()

        if len(boxes) > 0:
            out = self.pipeline(image=img_np, bboxes=boxes, labels=labels)
            img_np = out["image"]
            boxes = out["bboxes"]
            labels = out["labels"]

        if self.brightness_jitter > 0:
            factor = 1.0 + random.uniform(-self.brightness_jitter, self.brightness_jitter)
            img_np = np.clip(img_np * factor, 0.0, 1.0)

        image_out = torch.from_numpy(np.ascontiguousarray(img_np)).permute(2, 0, 1).float()

        if len(boxes) > 0:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            labels_t = torch.as_tensor(labels, dtype=torch.int64).reshape(-1)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        target["boxes"] = boxes_t
        target["labels"] = labels_t
        target["area"] = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        target["iscrowd"] = torch.zeros((len(labels_t),), dtype=torch.int64)

        return image_out, target


def build_train_transforms(config_module=None) -> DetectionAugmentation:
    """Factory so train.py only needs a one-line change to opt in.

    Reads AUG_MIN_BOX_VISIBILITY / AUG_BRIGHTNESS_JITTER from `config_module`
    if provided (falls back to this module's defaults otherwise), so the
    knobs live in one place (config.py) rather than being hardcoded here.
    """
    if config_module is None:
        return DetectionAugmentation()
    return DetectionAugmentation(
        min_visibility=getattr(config_module, "AUG_MIN_BOX_VISIBILITY", 0.2),
        brightness_jitter=getattr(config_module, "AUG_BRIGHTNESS_JITTER", 0.15),
    )
