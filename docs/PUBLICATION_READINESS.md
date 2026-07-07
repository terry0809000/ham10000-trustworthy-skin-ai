# Publication Readiness

This repository is prepared as a public, code-availability package for the V4 reviewer-patched notebook.

## Included

- Authoritative Colab notebook: `notebooks/01_full_colab_workflow.ipynb`
- Auto-extracted audit script: `scripts/full_notebook_pipeline.py`
- Reusable package helpers in `src/ham10000_ai/`
- Lightweight tests and GitHub Actions workflow
- Data access, reproducibility, citation, license, and security notes

## Excluded

- HAM10000 and ISIC images or archives
- Trained model files and converted deployment binaries
- Generated result tables, figures, caches, logs, and row-level processed data

## Interpretation boundary

The repository supports retrospective reproducibility and manuscript review. It does not provide autonomous clinical diagnosis, prospective validation, skin-tone fairness claims, or deployment clearance.

## Before manuscript submission

1. Run the notebook end to end in a fresh CPU Colab runtime or equivalent environment.
2. Deposit the exact generated `splits/`, `results/`, `figures/`, `config.json`, `manifest.json`, and reviewer compliance files in the manuscript DOI archive if required by the target journal.
3. Insert the final DOI/archive link into the notebook-generated code availability statement and manuscript files.
4. Verify that manuscript title, cover letter, and response letter use the same revised title.
