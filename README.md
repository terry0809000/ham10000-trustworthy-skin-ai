# When Should Skin-Lesion AI Stay Silent?

A leakage-controlled, calibrated, and selective-referral evaluation workflow for HAM10000 skin-lesion AI.

This repository packages the code from the Colab notebook into a GitHub-ready research repository. It supports the manuscript:

> **When Should Skin-Lesion AI Stay Silent? A Calibrated HAM10000 Study of Selective Referral, Subgroup Reliability, and External Robustness**

## Core design

The project evaluates a skin-lesion classifier as a **clinical decision-support workflow**, not as a leaderboard classifier:

1. lesion-level train/validation/test splitting to reduce same-lesion leakage;
2. validation-only model selection using macro-F1;
3. long-tail loss comparison: weighted CE, focal loss, and class-balanced CE;
4. validation-fitted temperature scaling;
5. selective referral / abstention using risk-coverage curves;
6. subgroup reliability analysis with denominators and bootstrap uncertainty;
7. optional ISIC2018 external evaluation;
8. stress testing and deployment-feasibility accounting.

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
manuscript/              Optional article files, if included
tests/                   Lightweight unit tests
```

## Quick start

```bash
git clone <repo-url>
cd ham10000-trustworthy-skin-ai
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
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

The notebook remains the authoritative end-to-end executable analysis. The `src/` package contains modularised utilities for metrics, calibration, selective referral, subgroup analysis, data handling, and model construction.

## Data availability

HAM10000 and ISIC image files are not redistributed in this repository. Download them from their official sources or from your approved archive. See `docs/DATA.md`.

## Clinical interpretation boundary

This project is a retrospective benchmark and decision-support evaluation. It does not claim autonomous clinical diagnosis, prospective deployment, or skin-tone fairness. HAM10000 does not provide reliable skin-tone labels; subgroup analyses are internal reliability analyses across available metadata-defined strata.

## License

Code is released under the MIT License. Dataset licenses and terms remain governed by their original providers.
