# Evaluation protocol and pointwise data schemas

This document describes only the paper-disclosed numerical checks supported by the scripts. Data 1, Data 2, and other experimental CSVs are not distributed in this repository. Access requests go to the corresponding author and are considered individually under the applicable confidentiality and approval conditions.

## Within-unit comparison

Data 1 is the gradual temperature-change record; Data 2 is the severe temperature-change record from the same original device. Each uses a chronological 70/15/15 training/validation/test partition. Sequential estimators use 200-sample windows, formed separately within each partition; the reported test comparison uses 4,301 held-out windows for each record. The six reported methods share the causal feature construction, target definition, split, and test rows. Static estimators use the endpoint feature vector; sequence estimators use the full window.

An **individually provided and authorized** pointwise CSV for each record can be checked locally with `verify_metrics.py`. Its exact header is:

```text
sample_index,target_proxy,svr,random_forest,standard_lstm,patch_transformer,timemixer,agcag
```

Each row is one held-out 200-sample window. `sample_index` is consecutive from zero within the provided test set and is not an original file row number or timestamp. All target and prediction values are in degrees per hour. The same constant may be subtracted from the target and every prediction in a record without changing RMSE, MAE, or R²; no method-specific recentering is allowed.

## Cross-unit transfer

The manuscript reports transfer between two independent physical units in both directions. Training and fitted scalers come from the source unit. Model weights are frozen on the target unit; only a constant offset estimated from the first 600 target samples is permitted, and those samples are excluded from evaluation. Results are reported by target axis and by transfer direction. This is **offset-aligned transfer**, not zero-shot, target-independent generalization.

If separately authorized and provided, a pointwise transfer CSV can be checked locally with `verify_transfer.py`. Its exact header is:

```text
direction,axis,thermal_segment,sample_index,target_proxy,agcag
```

`direction` is `A_to_B` or `B_to_A`; `axis` is `X`, `Y`, or `Z`; and `thermal_segment` is `dynamic`, `steady`, or `intermediate`. `sample_index` is consecutive from zero within each direction-axis test sequence. `target_proxy` and `agcag` use degrees per hour. The directional score is the unweighted arithmetic mean of the three axis scores. Dynamic and steady subsets use the prespecified temperature-rate rules in the manuscript's Supplementary Material; intermediate samples contribute to the overall score but not either stratified score. Segment labels allow a reader to recompute the strata but do not independently verify how those labels were constructed from restricted temperature records.

## Interpretation

The scripts compute RMSE as the square root of mean squared error, MAE as mean absolute error, and R² as one minus residual sum of squares divided by target total sum of squares. They reject missing or nonfinite values, malformed headers, nonconsecutive indices, and groups with insufficient or constant target data.

These files audit the numerical agreement of disclosed predictions with a declared operational proxy. They do not recreate the protected acquisition chain, train a model, establish an independent physical bias reference, or prove performance outside the reported stationary, temperature-varying conditions.
