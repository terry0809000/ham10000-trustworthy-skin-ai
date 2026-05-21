# Reproducibility notes

- Primary split is lesion-level via `lesion_id`.
- Main model selection uses validation macro-F1 only.
- Temperature scaling is fitted only on validation logits.
- `dx_type` is retained for subgroup analysis but excluded from predictive metadata.
- External ISIC2018 evaluation uses the HAM10000 validation-fitted temperature.
- TFLite export, if used, should be accompanied by equivalence, latency, and memory checks.

The full executed workflow remains in `notebooks/01_full_colab_workflow.ipynb`.
