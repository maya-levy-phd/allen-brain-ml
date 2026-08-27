# Predicting Mouse Neuron Dendrite Type from Electrophysiology

This project develops a reproducible machine-learning workflow for predicting whether mouse cortical neurons are **spiny** or **non-spiny** from electrophysiological measurements in the [Allen Cell Types Database](https://celltypes.brain-map.org/). To reduce information leakage, cells from the same donor are kept together during validation and are never split between the development and held-out test sets.

A regularized decision tree achieved a macro-F1 score of **0.940** on the previously untouched, donor-disjoint test set, compared with **0.885** for logistic regression. The result suggests that nonlinear thresholds and feature interactions contain useful information about broad dendritic phenotype.

## Key results

| Model | Development CV macro-F1 | Held-out macro-F1 |
|---|---:|---:|
| Most-frequent dummy | 0.348 ± 0.001 | — |
| Logistic regression | 0.846 ± 0.035 | 0.885 |
| Unrestricted decision tree | 0.877 ± 0.021 | — |
| Regularized decision tree | **0.891 ± 0.012** | **0.940** |

The final decision tree used:

```text
ccp_alpha=0.002
min_samples_leaf=5
```

On the held-out set, logistic regression made 27 classification errors and the regularized tree made 14. The tree improved recall for both target classes, with its largest improvement occurring for spiny cells.

## Scientific question

Neuronal cell types are commonly characterized using several complementary modalities, including morphology, electrophysiology, and gene expression. This analysis asks whether a cell's intrinsic electrophysiological properties are sufficient to predict a broad morphological distinction: whether its dendrites are spiny or non-spiny.

The original target contained three Allen dendrite labels:

- `aspiny`
- `sparsely spiny`
- `spiny`

Initial three-class experiments showed that the small sparsely-spiny class was substantially more difficult to distinguish. The primary analysis therefore uses the broader binary target:

| Original label | Binary target |
|---|---|
| `aspiny` | `non-spiny` |
| `sparsely spiny` | `non-spiny` |
| `spiny` | `spiny` |

The binary target was created only after the permanent development/test split had been established from the original three-class labels. This preserved the previously frozen test donors and avoided redefining the test set after inspecting the data.

## Data

The project retrieves cell records from the Allen Brain Map API and caches the raw response locally. Raw data are not committed to the repository.

The modeling cohort contains mouse cells with complete values for the selected electrophysiological features:

| Cohort property | Value |
|---|---:|
| Modeled cells | 1,186 |
| Unique donors | 697 |
| Development cells | 950 |
| Held-out test cells | 236 |
| Development non-spiny cells | 506 |
| Development spiny cells | 444 |
| Test non-spiny cells | 126 |
| Test spiny cells | 110 |

## Leakage prevention and validation design

The validation design reflects the grouped structure of the data: one donor can contribute multiple cells, so treating every cell as statistically independent could produce overly large estimates.

The workflow therefore uses the following safeguards:

- A permanent held-out test set is created before model selection.
- Development and test sets contain different donors.
- Five-fold cross-validation uses `StratifiedGroupKFold`.
- Cross-validation folds are fixed and shared across model comparisons.
- Models are compared using paired fold-level macro-F1 scores.
- Hyperparameters are selected using development data only.
- The held-out test set is evaluated once after the model and preprocessing choices are frozen.
- `tag__apical` is excluded because it directly reveals information about the dendrite label.
- `ef__avg_isi` is excluded because it duplicates information encoded by firing rate.

Macro-F1 is the primary metric because it gives equal weight to each target class rather than allowing the more common class to contribute more to the evaluation.

## Modeling approach

The analysis progresses from simple baselines to a regularized nonlinear model:

1. **Most-frequent dummy classifier** — establishes performance without predictive signal.
2. **Logistic regression** — provides an interpretable linear baseline with standardized features.
3. **Unrestricted decision tree** — tests whether nonlinear thresholds and feature interactions improve classification.
4. **Tree-complexity audit** — measures training/validation gaps, depth, leaf count, and the number of cells in the smallest leaf.
5. **Regularized decision tree** — controls complexity using minimum leaf size and cost-complexity pruning.
6. **Grid search** — evaluates all combinations of selected `ccp_alpha` and `min_samples_leaf` values using the fixed grouped CV folds.
7. **Held-out evaluation** — compares the frozen logistic and tree models on donor-disjoint test data.

The regularization grid was:

```python
{
    "ccp_alpha": [0.0, 0.002, 0.005, 0.01],
    "min_samples_leaf": [1, 5, 10, 20],
}
```

The selected combination, `ccp_alpha=0.002` and `min_samples_leaf=5`, achieved mean development macro-F1 of 0.891 with a standard deviation of 0.012 across folds. It substantially reduced the unrestricted trees' perfect training fit and single leaves while preserving strong validation performance.

## Held-out results

| Model | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Logistic regression | 0.886 | 0.884 | 0.885 |
| Regularized decision tree | **0.941** | **0.941** | **0.940** |

### Logistic regression

| Actual class | Predicted non-spiny | Predicted spiny | Recall |
|---|---:|---:|---:|
| Non-spiny | 115 | 11 | 0.913 |
| Spiny | 16 | 94 | 0.855 |

### Regularized decision tree

| Actual class | Predicted non-spiny | Predicted spiny | Recall |
|---|---:|---:|---:|
| Non-spiny | 118 | 8 | 0.937 |
| Spiny | 6 | 104 | 0.945 |

Both models performed better on the held-out set than their mean cross-validation scores. Each final model was trained on all 950 development cells, whereas an individual cross-validation model was trained on approximately 80% of the development cohort. The particular held-out donors may also constitute an easier sample than some validation folds.

The strong tree performance supports the presence of useful nonlinear structure, but individual tree splits and feature-importance values should not automatically be interpreted as causal biological mechanisms.

## Repository structure

```text
allen-brain-ml/
├── src/allen_brain_ml/
│   ├── allen_api.py       # Allen API client and pagination
│   ├── data.py            # Local caching and data preparation
│   ├── datasets.py        # Cohort, target, and grouped-split utilities
│   ├── evaluation.py      # Cross-validation and holdout evaluation
│   ├── features.py        # Feature selection utilities
│   └── models.py          # Model and pipeline factories
├── scripts/
│   ├── explore_cells.py   # Exploratory data analysis
│   └── train_baseline.py  # Reproducible experiment orchestration
├── tests/                 # Unit tests
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

The project was developed with Python 3.12 on Ubuntu.

```bash
git clone https://github.com/maya-levy-phd/allen-brain-ml.git
cd allen-brain-ml

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Usage

Explore the retrieved cell records and cohort characteristics:

```bash
python scripts/explore_cells.py
```

Run the complete classification analysis from the repository root:

```bash
python scripts/train_baseline.py
```

On the first run, the script retrieves records from the Allen API and saves a local cache under `data/raw/`. Subsequent runs reuse that cache. The raw cache is excluded from version control.

## Tests

Run the complete test suite from the repository root:

```bash
pytest
```

Tests cover API behavior, caching, cohort construction, target validation, grouped splitting, model factories, cross-validation, tree-complexity auditing, and held-out evaluation.

## Limitations

- The final evaluation uses one fixed held-out split rather than an independent external dataset.
- Only mouse cells with complete values for the selected electrophysiological features are included; complete-case filtering may introduce selection bias.
- Several cells can come from one donor. Group-aware splitting prevents leakage but does not eliminate all donor-level sources of variation.
- The binary target combines aspiny and sparsely-spiny cells and therefore does not represent the full diversity of neuronal morphology.
- The available labels are broad morphological categories rather than direct quantitative measurements of dendritic structure.

## Next steps: reconstructed morphology

The next phase will use the subset of cells with digital neuronal reconstructions. Planned work includes:

- downloading and validating reconstruction files;
- extracting quantitative morphology features such as total dendritic length, branch count, maximum path length, surface area, and spatial extent;
- comparing morphology-derived measurements across dendrite classes;
- testing whether reconstruction features explain or improve electrophysiology-based predictions;
- evaluating morphology models with the same donor-aware validation principles used here.

This phase shifts the project from predicting broad database labels toward modeling richer, directly measured neuronal structure.

## Data attribution

Data are provided by the [Allen Institute for Brain Science](https://alleninstitute.org/) through the [Allen Cell Types Database](https://celltypes.brain-map.org/) and Allen Brain Map API. Users of the data should consult the Allen Institute's [citation policy](https://alleninstitute.org/legal/citation-policy) and the relevant Allen Brain Map documentation for the requested dataset citation.

## License

This project’s source code is available under the [MIT License](LICENSE).
Data from the Allen Institute remain subject to the applicable Allen
Institute terms and citation requirements.
