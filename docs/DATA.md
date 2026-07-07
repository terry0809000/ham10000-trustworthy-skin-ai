# Data access and layout

This repository does **not** include HAM10000 or ISIC image files.

## HAM10000

HAM10000 can be obtained from the official Harvard Dataverse record:

- Persistent ID: `doi:10.7910/DVN/DBW86T`
- Default archive name expected by the notebook: `dataverse_files.zip`

The V4 notebook can attempt to download the Dataverse archive when `CONFIG["download_ham10000_dataverse_if_missing"]` is enabled. If you already have the archive, place it in Google Drive as `MyDrive/dataverse_files.zip` for the default Colab path, or pass the archive to `scripts/prepare_data.py` locally.

## ISIC2018 Task 3 related-domain test

The notebook can optionally download the official ISIC2018 Task 3 test input images and ground truth:

- `ISIC2018_Task3_Test_Input.zip`
- `ISIC2018_Task3_Test_GroundTruth.zip`

These files are used only for related-domain robustness checks. They are not used for training, validation, model selection, temperature scaling, threshold selection, or hyperparameter tuning.

Expected structure after extraction:

```text
data/raw/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
  HAM10000_images_part_2/
  ISIC2018_Task3_Test_GroundTruth.csv      # optional related-domain test
  ISIC2018_Task3_Test_Images/              # optional related-domain test
```

Use:

```bash
python scripts/prepare_data.py --zip /path/to/dataverse_files.zip
```

The script extracts nested archives, finds metadata, maps image IDs to paths,
and writes `data/processed/ham10000_manifest.csv`.

## Repository exclusion policy

Raw images, archives, generated manifests, trained models, result tables, figures, and converted deployment files are ignored by git. Commit only code, notebooks, configuration, documentation, and lightweight tests.
