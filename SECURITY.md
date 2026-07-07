# Security and Clinical-Safety Policy

This repository is a retrospective research workflow. It is not a deployed medical device and must not be used for autonomous diagnosis or patient management.

## Reporting issues

Please open a GitHub issue for reproducibility, data-leakage, dependency, or documentation problems. Do not include patient-identifiable information, raw image archives, private access tokens, or restricted dataset files in an issue or pull request.

## Data handling

HAM10000 and ISIC files are not redistributed here. Keep raw datasets, derived row-level manifests, trained model artifacts, generated result tables, figures, and deployment binaries out of git unless a release process explicitly approves them.

## Clinical boundary

Any clinical interpretation should be limited to retrospective decision-support evaluation. Prospective validation, target-hardware testing, workflow integration, and clinical governance review are required before deployment claims.
