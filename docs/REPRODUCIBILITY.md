# Reproducibility notes

- Primary split is lesion-level via `lesion_id`.
- Main model selection uses validation macro-F1 only.
- Temperature scaling is fitted only on validation logits.
- `dx_type` is retained for subgroup analysis but excluded from predictive metadata.
- ISIC2018 Task 3 is used only as related-domain robustness and uses the HAM10000 validation-fitted temperature.
- The V4 notebook enables model funnel comparison, long-tail loss ablation, repeated final seeds, bootstrap confidence intervals, MC Dropout, stress tests, TFLite export, and reviewer/editor compliance exports.
- TFLite export is a technical conversion step. It should be accompanied by numerical equivalence, latency, memory, and post-quantization performance checks before any deployment claim.
- No universal clinical ECE threshold is assumed. ECE is reported as top-label ECE with equal-width bins, and calibration findings should be interpreted alongside class-wise and subgroup results.

The full executed workflow remains in `notebooks/01_full_colab_workflow.ipynb`.

## Reviewer-facing outputs

After a complete run, the notebook writes a manifest and reviewer exports under the configured results, figures, splits, and model directories. Key publication audit outputs include:

- split protocol and class distribution summaries;
- training reproducibility details and class weights;
- global and class-wise calibration tables, ECE bins, MCE, and calibration CIs;
- melanoma-specific selective-referral counts;
- cross-dataset overlap checks for HAM10000 versus ISIC2018;
- TFLite numerical equivalence, latency, memory, and post-quantization performance summaries;
- `reviewer_editor_compliance_matrix.csv`.
