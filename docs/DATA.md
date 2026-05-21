# Data access and layout

This repository does **not** include HAM10000 or ISIC image files.

Expected structure after extraction:

```text
data/raw/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
  HAM10000_images_part_2/
  ISIC2018_Task3_Test_GroundTruth.csv      # optional external test
  ISIC2018_Task3_Test_Images/              # optional external test
```

Use:

```bash
python scripts/prepare_data.py --zip /path/to/dataverse_files.zip
```

The script extracts nested archives, finds metadata, maps image IDs to paths,
and writes `data/processed/ham10000_manifest.csv`.
