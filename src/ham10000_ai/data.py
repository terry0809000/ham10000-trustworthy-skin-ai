from __future__ import annotations

from pathlib import Path
from typing import Iterable
import zipfile
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def extract_nested_zip(zip_path: str | Path, out_dir: str | Path) -> Path:
    """Extract an outer zip file and any nested zip files one level deep or recursively."""
    zip_path = Path(zip_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    # recursively extract nested archives
    for nested in list(out_dir.rglob("*.zip")):
        nested_out = nested.with_suffix("")
        nested_out.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(nested) as zf:
                zf.extractall(nested_out)
        except zipfile.BadZipFile:
            continue
    return out_dir


def find_metadata_file(root: str | Path) -> Path:
    root = Path(root)
    candidates = [p for p in root.rglob("*.csv") if "metadata" in p.name.lower() or "groundtruth" in p.name.lower()]
    if not candidates:
        raise FileNotFoundError(f"No metadata CSV found under {root}")
    # Prefer HAM10000 metadata over ISIC test ground truth if both exist.
    candidates = sorted(candidates, key=lambda p: ("HAM10000" not in p.name.upper(), len(str(p))))
    return candidates[0]


def build_image_map(root: str | Path) -> dict[str, str]:
    root = Path(root)
    return {p.stem: str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS}


def load_ham10000_dataframe(root: str | Path, class_to_idx: dict[str, int]) -> pd.DataFrame:
    meta_path = find_metadata_file(root)
    df = pd.read_csv(meta_path)
    required = {"image_id", "dx", "lesion_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata file missing required columns: {missing}")
    df = df.copy()
    df["dx"] = df["dx"].astype(str).str.lower()
    df = df[df["dx"].isin(class_to_idx)].copy()
    df["label"] = df["dx"].map(class_to_idx).astype(int)
    img_map = build_image_map(root)
    df["image_path"] = df["image_id"].map(img_map)
    df = df[df["image_path"].notna()].reset_index(drop=True)
    return df


def audit_dataframe(df: pd.DataFrame) -> dict[str, int]:
    return {
        "n_rows": int(len(df)),
        "n_images": int(df["image_id"].nunique()) if "image_id" in df else int(len(df)),
        "n_lesions": int(df["lesion_id"].nunique()) if "lesion_id" in df else 0,
        "n_lesions_with_multiple_images": int((df.groupby("lesion_id").size() > 1).sum()) if "lesion_id" in df else 0,
        "max_images_per_lesion": int(df.groupby("lesion_id").size().max()) if "lesion_id" in df else 0,
    }


def lesion_level_split(
    df: pd.DataFrame,
    label_col: str = "label",
    group_col: str = "lesion_id",
    test_size: float = 0.20,
    val_size_relative: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation, test dataframes using lesion-level grouped splitting."""
    df = df.reset_index(drop=True).copy()
    try:
        sgkf = StratifiedGroupKFold(n_splits=int(round(1 / test_size)), shuffle=True, random_state=seed)
        train_val_idx, test_idx = next(sgkf.split(df, df[label_col], df[group_col]))
    except Exception:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_idx, test_idx = next(gss.split(df, df[label_col], df[group_col]))

    train_val = df.iloc[train_val_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    try:
        sgkf2 = StratifiedGroupKFold(n_splits=int(round(1 / val_size_relative)), shuffle=True, random_state=seed)
        train_idx, val_idx = next(sgkf2.split(train_val, train_val[label_col], train_val[group_col]))
    except Exception:
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size_relative, random_state=seed)
        train_idx, val_idx = next(gss2.split(train_val, train_val[label_col], train_val[group_col]))

    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)
    return train, val, test
