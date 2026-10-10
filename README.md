<p align="center"><img src="assets/project-cover.png" width="100%" alt="Conceptual scientific artwork for Skin lesion AI" /></p>

# Calibrated Selective Referral for Skin Lesion AI on HAM10000

A leakage-controlled, calibrated, selective-referral evaluation workflow for HAM10000 skin-lesion AI.

This repository packages the final notebook Colab workflow and reusable Python helpers for the study:

> **Calibrated Selective Referral for Skin Lesion AI on HAM10000**

The notebook is configured for a CPU-only full publishable rerun in Google Colab or a compatible local Python environment. It includes reviewer-requested exports for split auditing, class-wise metrics, calibration, melanoma-specific selective referral counts, related-domain ISIC2018 checks, TFLite profiling, and a final compliance matrix.

## Core design

The project evaluates a skin-lesion classifier as a **clinical decision-support workflow**, not as a leaderboard classifier:

1. lesion-level train/validation/test splitting to reduce same-lesion leakage;
2. validation-only model selection using macro-F1;
3. long-tail loss comparison: weighted CE, focal loss, and class-balanced CE;
4. validation-fitted temperature scaling;
5. selective referral / abstention using risk-coverage curves;
6. subgroup reliability analysis with denominators and bootstrap uncertainty;
7. optional ISIC2018 Task 3 related-domain robustness analysis;
8. stress testing, TFLite equivalence checks, and deployment-feasibility accounting;
9. reviewer/editor compliance exports that map claims to generated files.

## Repository structure

```text
configs/                 YAML configuration
notebooks/               Full Colab notebook
src/ham10000_ai/         Reusable Python package modules
scripts/                 CLI/helper scripts
results/                 Generated CSV outputs (ignored by git)
models/                  Trained models (ignored by git)
figures/                 Generated figures (ignored by git)
docs/                    Data and reproducibility notes
tests/                   Lightweight unit tests
```

## Quick start

```bash
git clone https://github.com/terry0809000/ham10000-trustworthy-skin-ai.git
cd ham10000-trustworthy-skin-ai
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install -e .
```

Prepare the data from a local archive:

```bash
python scripts/prepare_data.py --zip /path/to/dataverse_files.zip
```

Open the full workflow notebook:

```bash
jupyter notebook notebooks/01_full_colab_workflow.ipynb
```

The notebook remains the authoritative end-to-end executable analysis. The `scripts/full_notebook_pipeline.py` file is auto-extracted from the notebook for auditability, and the `src/` package contains modularised utilities for metrics, calibration, selective referral, subgroup analysis, data handling, and model construction.

## Colab run policy

Use the notebook from the first cell in a fresh Colab CPU runtime. The V4 workflow mounts Google Drive before creating Drive subdirectories, falls back safely to `/content` if Drive is unavailable, and can download the HAM10000 Dataverse archive and official ISIC2018 Task 3 test files when configured to do so.

For the manuscript-ready full run, keep the major switches enabled in `CONFIG`: training, model funnel, long-tail ablation, three final seeds, bootstrap CIs, MC Dropout, stress testing, TFLite export, and reviewer compliance matrix generation.

## Data availability

HAM10000 and ISIC image files are not redistributed in this repository. Download them from their official sources or from your approved archive. See `docs/DATA.md`.

## Clinical interpretation boundary

This project is a retrospective benchmark and decision-support evaluation. It does not claim autonomous clinical diagnosis, prospective deployment, or skin-tone fairness. HAM10000 does not provide reliable skin-tone labels; subgroup analyses are internal reliability analyses across available metadata-defined strata. ISIC2018 Task 3 is treated as related-domain robustness, not independent external validation.

## License

Code is released under the MIT License. Dataset licenses and terms remain governed by their original providers.
