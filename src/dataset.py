"""PyTorch datasets for 16-band Hyperspectral Object Detection."""

from pathlib import Path
from xml.etree import ElementTree as ET
import torch
from torch.utils.data import Dataset

from . import config
from .x2cube import load_cube, normalize_cube


def parse_voc_xml(xml_path: Path, max_w: float = None, max_h: float = None):
    """Parse VOC XML bounding boxes, enforce x2 > x1, y2 > y1, and clamp to boundaries."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes, labels = [], []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in config.CLASS_TO_ID:
            continue
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        # Clamp to bounds if known
        if max_w is not None and max_h is not None:
            xmin = max(0.0, min(xmin, max_w - 1.0))
            ymin = max(0.0, min(ymin, max_h - 1.0))
            xmax = max(0.0, min(xmax, max_w))
            ymax = max(0.0, min(ymax, max_h))

        # Enforce strictly positive width and height (at least 1 pixel)
        if (xmax - xmin) >= 1.0 and (ymax - ymin) >= 1.0:
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(config.CLASS_TO_ID[name])

    return boxes, labels


class HyperspectralDetDataset(Dataset):
    def __init__(self, img_dir=None, ann_dir=None, transforms=None):
        self.img_dir = Path(img_dir or config.TRAIN_IMG_DIR)
        self.ann_dir = Path(ann_dir or config.TRAIN_ANN_DIR)
        self.transforms = transforms

        raw_samples = sorted(p.stem for p in self.img_dir.glob("*.png"))
        
        # Verify valid annotation and at least 1 valid bounding box
        self.samples = []
        for s in raw_samples:
            ann_file = self.ann_dir / f"{s}.xml"
            if ann_file.exists():
                boxes, _ = parse_voc_xml(ann_file)
                if len(boxes) > 0:
                    self.samples.append(s)

        print(f"Loaded dataset: {len(self.samples)} valid samples with annotations.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem = self.samples[idx]
        img_path = self.img_dir / f"{stem}.png"
        ann_path = self.ann_dir / f"{stem}.xml"

        cube = load_cube(img_path)          # (H, W, 16)
        h, w, _ = cube.shape
        cube = normalize_cube(cube)
        image = torch.from_numpy(cube).permute(2, 0, 1).float()  # (16, H, W)

        boxes, labels = parse_voc_xml(ann_path, max_w=float(w), max_h=float(h))

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor([l + 1 for l in labels], dtype=torch.int64).reshape(-1)

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


class HyperspectralInferenceDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = Path(img_dir)
        self.paths = sorted(self.img_dir.glob("*.png"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        cube = load_cube(path)
        cube = normalize_cube(cube)
        image = torch.from_numpy(cube).permute(2, 0, 1).float()
        image_id = path.stem
        return image, image_id


def collate_fn(batch):
    return tuple(zip(*batch))