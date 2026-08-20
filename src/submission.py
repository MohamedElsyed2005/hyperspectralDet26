"""Generate and validate submission.csv."""

from pathlib import Path
import pandas as pd
from . import config

REQUIRED_COLUMNS = ["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]


def rows_to_dataframe(rows):
    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS[1:])
    df.insert(0, "id", range(len(df)))
    return df[REQUIRED_COLUMNS]


def write_submission(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = rows_to_dataframe(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} detections for {df['image_id'].nunique()} images -> {out_path}")
    return df


def append_ranking_predictions(test_csv_path, ranking_rows, out_path):
    existing = pd.read_csv(test_csv_path)
    ranking_df = rows_to_dataframe(ranking_rows)
    ranking_df["id"] = range(len(existing), len(existing) + len(ranking_df))
    combined = pd.concat([existing, ranking_df], ignore_index=True)
    combined.to_csv(out_path, index=False)
    print(f"Combined: {len(existing)} test rows + {len(ranking_df)} ranking rows -> {out_path}")
    return combined


def validate_submission(csv_path):
    df = pd.read_csv(csv_path)
    problems = []

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        problems.append(f"Missing columns: {missing_cols}")

    if df["id"].duplicated().any():
        problems.append("Duplicate 'id' values found")

    if not df["class_id"].between(0, config.NUM_CLASSES - 1).all():
        problems.append(f"class_id out of range [0, {config.NUM_CLASSES - 1}]")

    if not df["confidence"].between(0, 1).all():
        problems.append("confidence out of range [0, 1]")

    bad_boxes = df[(df["x2"] <= df["x1"]) | (df["y2"] <= df["y1"])]
    if len(bad_boxes) > 0:
        problems.append(f"{len(bad_boxes)} rows have non-positive-area boxes")

    if problems:
        print("[VALIDATION FAILED]")
        for p in problems:
            print(f"  - {p}")
        return False
    else:
        print(f"[OK] {csv_path} is valid and ready for Kaggle submission ({len(df)} rows).")
        return True