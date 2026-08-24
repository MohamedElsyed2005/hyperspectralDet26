"""Cheap, GPU-free diagnostic: per-class bounding-box size statistics from the
training annotations alone.

Run this to check whether classes with low AP (see `eval_per_class.py` output)
are disproportionately large relative to the image -- if so, that points at
the RPN/anchor configuration (region proposal / scale coverage) rather than
at the classifier or the spectral features, which is a very different fix.

Usage:
    python -m src.analyze_box_sizes
"""

from collections import defaultdict
from xml.etree import ElementTree as ET

from . import config
from .dataset import parse_voc_xml


def main():
    img_dir = config.TRAIN_IMG_DIR
    ann_dir = config.TRAIN_ANN_DIR

    per_class_fracs = defaultdict(list)   # box_area / image_area
    per_class_areas = defaultdict(list)   # raw box_area in px^2
    per_class_aspect = defaultdict(list)  # box width / height

    ann_files = sorted(ann_dir.glob("*.xml"))
    print(f"Scanning {len(ann_files)} annotation files under {ann_dir} ...")

    n_missing_size = 0
    for ann_path in ann_files:
        tree = ET.parse(ann_path)
        root = tree.getroot()
        size_el = root.find("size")
        if size_el is None:
            n_missing_size += 1
            continue
        img_w = float(size_el.find("width").text)
        img_h = float(size_el.find("height").text)
        img_area = img_w * img_h

        boxes, labels = parse_voc_xml(ann_path, max_w=img_w, max_h=img_h)
        for (xmin, ymin, xmax, ymax), label_id in zip(boxes, labels):
            w = xmax - xmin
            h = ymax - ymin
            area = w * h
            class_name = config.ID_TO_CLASS[label_id]
            per_class_fracs[class_name].append(area / img_area)
            per_class_areas[class_name].append(area)
            per_class_aspect[class_name].append(w / h if h > 0 else 0.0)

    if n_missing_size:
        print(f"(skipped {n_missing_size} files with no <size> element)")

    def median(vals):
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2

    rows = []
    for class_name, fracs in per_class_fracs.items():
        rows.append({
            "class_name": class_name,
            "n_boxes": len(fracs),
            "median_area_frac": median(fracs),
            "median_area_px2": median(per_class_areas[class_name]),
            "median_aspect_wh": median(per_class_aspect[class_name]),
        })

    rows.sort(key=lambda r: r["median_area_frac"], reverse=True)

    header = f"{'class_name':<16} {'n_boxes':>8} {'med_area_%img':>14} {'med_area_px2':>13} {'med_w/h':>8}"
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['class_name']:<16} {r['n_boxes']:>8} "
            f"{r['median_area_frac']*100:>13.2f}% {r['median_area_px2']:>13.0f} "
            f"{r['median_aspect_wh']:>8.2f}"
        )

    print()
    print("Cross-reference the classes at the TOP of this list (largest median")
    print("area-fraction-of-image) against the LOW-AP classes from eval_per_class.py.")
    print("If they're the same classes, the low AP is a scale/RPN-coverage issue,")
    print("not a spectral-confusion issue -- fix direction is anchor/RPN tuning,")
    print("not more spectral-band engineering.")


if __name__ == "__main__":
    main()
