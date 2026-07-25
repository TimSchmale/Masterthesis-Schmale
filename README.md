# Regression for Large Matrices with Joint Column and Row Reduction

Sampling-variance study for CUR-based dimensionality reduction applied to
linear regression, logistic regression, and Lasso.

## Project Structure

```
refactored/
├── pipeline.py              # Experiment runners (5 pipelines)
├── reduction.py             # Column & row reduction primitives
├── scoring.py               # Column/row score computation (LS, CLS, RS, CS)
├── modeling.py              # Model fitting (OLS, Logistic, Lasso)
├── evaluation.py            # Screening metrics (TPR, FDR, MCC)
├── visualizations.py        # Plotnine-based boxplots & histograms
├── Masterthesis_Reduction   # Main notebook: runs all pipelines
├── Masterthesis_Plots       # Plotting notebook: loads pickles & visualizes
└── README.md
```

## Pipelines

| # | Pipeline | Function | Order | Model |
|---|----------|----------|-------|-------|
| 1 | OLS Column→Row | `run_ols_experiment(..., order="column_first")` | Column reduction → Row reduction (leverage) → OLS | OLS |
| 2 | OLS Column Only | `run_ols_experiment(..., order="column_only")` | Column reduction → OLS (no row reduction) | OLS |
| 3 | OLS Row→Column | `run_ols_experiment(..., order="row_first")` | Rademacher sketch → Column reduction → OLS | OLS |
| 4 | Logistic | `run_logistic_experiment(...)` | Column reduction → Coreset row reduction → Logistic | Logistic Reg. |
| 5 | Lasso (3 variants) | `run_lasso_experiment(...)` | Sketch → Lasso (theo/binary/CLS+Lasso) | Lasso |

## Experimental Design

- **10 pre-generated datasets** (X1.csv…X10.csv) per simulation setting (p, n)
- **10 outer seeds** (SEED_FROM=1…10): each seed produces a different 80/20 train/test split
- **Per seed × dataset**:
  - Pipelines 1, 2, 4: Scores on X_train → Reduction → Model → Evaluate on test set
  - Pipeline 3 (Row→Col): Sketch X_train → Scores on Sketch → Column reduction on Sketch → Model → Evaluate on test set
  - Pipeline 5 (Lasso): Sketch X_train → Lasso / (CLS Scores on X_train → Column reduction on Sketch → Lasso)
- **Total**: 100 pipeline runs per (k, method) combination

### Scoring Methods

| Key | Method | Description |
|-----|--------|-------------|
| LS | Leverage Scores | Column leverage from top-k right singular vectors |
| CLS | Cross-Leverage Scores | Alignment between columns and response via augmented SVD |
| RS | Random Scores | Uniform baseline (non-informative) |
| CS | Combined Scores | Convex mix: 0.8·‖CLS‖ + 0.2·‖LS‖ |

## Reproducibility

All random operations are seeded deterministically:

- `numpy_train_test_split(seed=seed)` — own RNG via `default_rng(seed)`
- `np.random.seed(seed + 42)` — global state controls:
  - Bernoulli sampling in `column_reduction` and `row_reduction_leverage`
  - Rademacher sketch in `row_reduction_sketch`
  - Coreset sampling in `row_reduction_coreset`
  - `randomized_svd` (via `random_state=None` → global state)
  - Random scores (`np.random.uniform`)

Running the same configuration multiple times produces **identical results**.

## Soft Abort Logic

When column+row reduction produces an underdetermined system (more columns
than rows), the pipeline uses a **two-phase approach**:

1. **Phase 1**: Compute reductions for ALL methods × ALL reps. If any
   method produces an underdetermined system for rep `i`, mark `i` as aborted.
2. **Phase 2**: Model fitting. Aborted reps get `NaN` for **all methods**
   (ensures fair comparison with equal sample sizes).

## Configuration

```python
# Paths
BASE = "/path/to/Masterthesis/"
DATA_FOLDER = "Simulation Data/p1000_n1000"  # p5000_n5000, p10000_n10000
RESULTS_FOLDER = "Results/p1000"

# Parameters
k_vector = [10, 20, 25, 50, 100, 200, 300, 400, 500]
REPS = 10          # number of datasets
OUTER_REPS = 10    # number of seeds
TEST_SIZE = 0.2
SEED_FROM = 1
```

## Output

Results are saved as per-seed pickles:
```
Results/p1000/results_ols_col_row_seed_1.pkl
Results/p1000/results_ols_col_row_seed_2.pkl
...
```

Structure: `{k: {method: {metric: [values_per_rep]}}}`

## Visualization

The `Masterthesis_Plots` notebook loads all pickles and generates:
- Loss boxplots (RMSE, Brier, Cross-Entropy, Total Time)
- Screening boxplots (TPR, FDR, MCC)
- Score distribution histograms
- Number of selected features

Default aggregation: **median per seed** (10 points per boxplot).
Override with `aggregate="raw"` for all 100 individual values.

## Dependencies

- numpy, pandas, scikit-learn
- plotnine (visualization)
- No Spark required — runs on single-node Python
