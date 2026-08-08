# Regression for Large Matrices with Joint Column and Row Reduction

Sampling-variance study for sequential reduction procedures targeting both
sample size and dimensionality, applied to OLS, logistic regression, and
Lasso.

## Structure

```
pipeline.py              Experiment runners (one function per pipeline)
reduction.py             Column & row reduction primitives
scoring.py               Score computation (LS, CLS, RS, CS)
modeling.py              Model fitting (OLS, Logistic, Lasso)
evaluation.py            Screening metrics (TPR, FDR, MCC)
visualizations.py        Plotnine plots

Masterthesis_Reduction   Runs all pipelines
Masterthesis_Plots       Loads pickles, produces all figures
Simulation               Generates synthetic data

Results/                 Per-seed pickle files
plots/                   Saved PDFs
```

## Pipelines

| # | Name | Reduction | Model |
|---|------|-----------|-------|
| 1 | Column→Row | Column selection → Leverage row sampling | OLS |
| 2 | Column Only | Column selection (no row reduction) | OLS |
| 3 | Row→Column | Rademacher sketch → Column selection on sketch | OLS |
| 4 | Logistic | Column selection → Coreset row sampling | LogReg (L2) |
| 5 | Lasso | Sketch → Lasso in 3 variants (see below) | Lasso |

**Lasso variants:**
- *Theoretical*: λ = 1/√(2k), applied to full sketch
- *Practical*: Binary-search λ, applied to full sketch
- *Hybrid*: CLS scores on sketch → column reduction → Lasso with binary-search λ

## Scoring Methods

| Key | Name | Idea |
|-----|------|------|
| LS | Leverage Scores | Column leverage from top-k right singular vectors |
| CLS | Cross-Leverage Scores | Column–response alignment via augmented SVD |
| RS | Random Scores | Uniform random (baseline) |
| CS | Combined Scores | 0.8·|CLS| + 0.2·|LS| |

## Experimental Setup

10 pre-generated datasets × 10 seeds = 100 runs per (k, method).
Each seed defines a fresh 80/20 train/test split.

Scores are computed on the training data (Pipelines 1, 2, 4) or on the
sketch (Pipelines 3, 5). All random operations are seeded for full
reproducibility — see `numpy_train_test_split` and `np.random.seed(seed + 42)`.

## Soft Abort (OLS Only)

If a CUR reduction leaves more columns than rows (underdetermined), OLS has
no unique solution. In that case the OLS pipelines mark the rep as aborted
and write NaN for *all* methods at that rep (keeps sample sizes fair).

Not needed for Logistic/Lasso — regularization handles d > n.

## Configuration

```python
BASE = "/path/to/Masterthesis/"
DATA_FOLDER = "Simulation Data/d1000_n1000"
RESULTS_FOLDER = "Results/d1000"

k_vector = [10, 20, 25, 50, 100, 200, 300, 400, 500]
REPS = 10
OUTER_REPS = 10
TEST_SIZE = 0.2
SEED_FROM = 1
```

## Output

Per-seed pickles: `Results/d1000/results_ols_col_row_seed_1.pkl`, etc.

Structure: `{k: {method: {metric: [values_per_rep]}}}`

## Plots

`Masterthesis_Plots` generates:
- Loss (RMSE, Brier, Cross-Entropy)
- Time (total + decomposition into scoring, reduction, model fit)
- Screening (TPR, FDR, MCC)
- Tuned hyperparameters (μ, λ)
- Score distributions, number of selected covariates

Aggregation: median across the 10 reps per seed (one point per seed per boxplot).

## Dependencies

numpy, pandas, scikit-learn, plotnine.
