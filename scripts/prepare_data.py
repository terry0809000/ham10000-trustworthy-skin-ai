#!/usr/bin/env python
from pathlib import Path
import argparse
from ham10000_ai.config import load_config
from ham10000_ai.data import extract_nested_zip, load_ham10000_dataframe, audit_dataframe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Path to dataverse_files.zip or equivalent archive")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    extract_nested_zip(args.zip, cfg.paths.raw_dir)
    df = load_ham10000_dataframe(cfg.paths.raw_dir, cfg.class_to_idx)
    out = cfg.paths.processed_dir / "ham10000_manifest.csv"
    df.to_csv(out, index=False)
    print(audit_dataframe(df))
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
