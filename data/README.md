# Data directory

This directory stores data used by the regime-aware portfolio risk and allocation pipeline.

Recommended structure:

- `raw/`: downloaded or externally sourced raw data.
- `processed/`: cleaned returns, regime labels, and intermediate outputs.
- `sample/`: small sample files that may be tracked for demonstration and testing.

Large or reproducible CSV files are intentionally ignored by Git. They should be regenerated through the data pipeline notebooks or scripts.