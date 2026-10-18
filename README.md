# FOG thermal-compensation metric audit

This is a limited, evaluation-only release candidate for the measurements reported in the associated IEEE TIM submission. It is **not** the full AG-CAG implementation, a training package, or a reproduction of the protected hardware experiment. The repository is not public yet.

The verifier recomputes error against the paper's **sensor-derived operational bias proxy**, not against an independently calibrated absolute bias. A favorable proxy score must not be interpreted as proof of absolute calibration or performance under nonzero rotation.

## What an independent reader can check

If institutionally approved, held-out per-sample proxy targets and predictions are added as `approved_data/data1.csv` and `approved_data/data2.csv`. The same test rows are used for all six estimators in the manuscript's within-unit comparison: SVR, Random Forest, Standard LSTM, Patch Transformer, TSMixer, and AG-CAG. A reader can recompute RMSE, MAE, and R² for each estimator and check the relative comparison. The files can have one common constant offset removed from all target and prediction columns in each dataset; these three metrics are invariant to that operation.

This does **not** independently verify data collection, training, preprocessing, uncertainty propagation, Allan-deviation results, or physical generalization. Those claims require additional authorized evidence. The synthetic example below only tests the verifier; it is not paper evidence.

## Run

Python 3.9 or later is sufficient; no third-party packages are needed.

```text
python verify_metrics.py examples/synthetic_demo.csv
python -m unittest discover -s tests
```

Once the two real CSVs have been individually cleared for public release:

```text
python verify_metrics.py approved_data/data1.csv
python verify_metrics.py approved_data/data2.csv
```

Each CSV has columns `sample_index,target_proxy,svr,random_forest,standard_lstm,patch_transformer,tsmixer,agcag`. Indices are consecutive test-window indices, not source-file row numbers or timestamps. Targets and predictions have units of degrees per hour, after one shared constant offset per dataset. The verifier rejects missing/nonfinite values, duplicate or unordered indices, and fewer than two observations. It uses population mean-squared error for RMSE and the usual coefficient of determination, matching the comparison script's metric definitions.

## Release boundary

No raw gyroscope or temperature time series, hardware identifiers, device parameters, location, dates, model checkpoints, or full training code are included. **Do not copy files from the private project or validation bundle into this repository.** The candidate's `.gitignore` intentionally excludes `approved_data/` until a named owner approves the exact sanitized CSVs and the public-release decision is documented.

No software reuse license is granted by this candidate. Before making the repository public, the rights holder must choose the license and confirm that publishing even evaluation code and derived test values is permitted. A public GitHub repository cannot technically prevent others from copying what it exposes.
