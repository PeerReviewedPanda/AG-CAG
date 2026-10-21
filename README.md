# AG-CAG for causal FOG thermal-error compensation

This repository is the implementation-oriented companion to the associated IEEE TIM manuscript. It presents the core Adaptive Gated Convolutional Attention-GRU (AG-CAG) architecture, executable examples, metric-verification utilities, and schema-checked evaluation tools for causal thermal-error compensation in fiber-optic gyroscopes (FOGs).

<p align="center">
  <img src="assets/fig1_agcag_architecture.png" alt="Figure 1: causal preprocessing and AG-CAG network architecture" width="660">
</p>

<p align="center"><em>Fig. 1. Causal preprocessing and AG-CAG architecture, reproduced from the revised manuscript.</em></p>

## Algorithm overview

AG-CAG is designed as a causal estimator for temperature-driven bias variation. Its processing path combines:

1. **Backward-only preprocessing**, so each estimate uses only measurements available at or before the current time.
2. **A 1-D convolutional encoder**, which extracts local temperature and gyro-history patterns from each sliding window.
3. **Compact channel attention**, where shared average- and max-pooled descriptors adaptively emphasize informative causal features.
4. **GRU temporal modeling**, which summarizes longer thermal-history effects associated with heat accumulation and hysteresis.
5. **Learned gated fusion**, which balances the terminal GRU representation with a local CNN shortcut before regression.

The resulting architecture jointly represents recent thermal change and accumulated thermal history while retaining a deployment-compatible causal signal path. The repository exposes this tensor flow in `agcag_example.py` and provides independent metric-audit utilities for the evaluation interfaces reported in the manuscript.

## What can be checked

- `agcag_example.py` shows the complete CNN, shared average/max channel attention, GRU, learned gate, and regression-head tensor flow on executable history windows.
- `verify_metrics.py` recalculates RMSE, MAE, and R² for the six within-unit estimators on the same held-out rows of Data 1 or Data 2.
- `verify_transfer.py` recalculates per-axis, per-direction, and temperature-segment metrics from an approved pointwise cross-unit file. Directional summaries are arithmetic means of the three axis scores, not scores after pooling axes.
- The included schema-compatible examples provide immediate software checks for metric calculation, grouping rules, and input validation.

The file schemas and the boundary between numerical audit and full experimental reproduction are described in [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md).

## Run the examples

Python 3.9 or later is sufficient for the metric scripts. The architecture example additionally requires PyTorch; see `requirements-model.txt`.

```text
python verify_metrics.py examples/synthetic_demo.csv
python verify_transfer.py examples/synthetic_transfer.csv
python -m unittest discover -s tests
```

After installing PyTorch, the network can be exercised without any experimental input:

```text
python agcag_example.py
```

## Verification and research contact

The metric scripts accept schema-compatible pointwise files supplied through an approved verification channel. Researchers and reviewers interested in reproducing the reported checks, discussing model adaptation, or extending AG-CAG to related inertial-sensor applications may contact the corresponding author at **qiao_huiting@hotmail.com**.

The repository is intended for scholarly inspection, numerical verification, and research discussion. Questions about reuse or methodological extensions are welcome through the same contact address.
