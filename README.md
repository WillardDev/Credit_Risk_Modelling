# Credit Risk Modelling

Predict whether a loan applicant will default, using a heavily imbalanced consumer-loan dataset. The project is a beginner-friendly Jupyter notebook that walks through the full credit-risk workflow: data cleaning, EDA, feature engineering, model selection, evaluation, and a live "score a new applicant" example.

## Project structure

```
credit_risk_modelling/
├── Week 3 project.ipynb          # Main notebook (Sections 1-7 + appendix)
├── 0305 Starter Notebook_copy.ipynb  # Starter template (reference)
├── data/
│   └── raw-data.csv              # Raw dataset, 119,528 rows x 32 columns (V1-V32)
└── README.md
```

## Requirements

- Python 3.9+
- Jupyter (Notebook or Lab)

Core libraries:

| Package        | Used for                              |
|----------------|---------------------------------------|
| pandas         | Data loading and manipulation         |
| numpy          | Numeric operations                    |
| matplotlib     | Plots                                 |
| seaborn        | Statistical plots, heatmaps           |
| scikit-learn   | Preprocessing, models, evaluation     |

Optional (used only in the appendix; auto-installed via `pip` when run):

| Package            | Used for                    |
|--------------------|-----------------------------|
| missingno          | Missingness map (A.1)       |
| imbalanced-learn   | SMOTE comparison (A.3)      |
| statsmodels        | VIF analysis (A.4)          |

On macOS (Homebrew) the appendix installs may hit `externally-managed-environment`. Create a virtual environment before running instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn jupyter missingno imbalanced-learn statsmodels
jupyter notebook
```

## Running the notebook

1. Open `Week 3 project.ipynb` in Jupyter.
2. Run all cells (Kernel > Restart & Run All). The appendix cells load their own optional packages.
3. The main pipeline (Sections 1-6) needs only the core libraries; the appendix (A.1-A.5) needs the optional ones.

## What the notebook covers

1. **Libraries & Data** — loads `data/raw-data.csv` (119,528 rows x 32 columns) and renames the generic `V1`-`V32` headers into meaningful business names (`loan_amount`, `employment`, `tier`, `target`, ...).
2. **Missing Values & Data Cleaning** — explains why rows missing the "block" of structural columns (34,480 rows, ~28.8%) are kept rather than deleted, imputes medians/`Unknown`, turns near-empty columns into `has_...` flags, and builds an `age` feature from date of birth.
3. **Exploratory Data Analysis (EDA)** — quantifies the severe class imbalance (~2.2% defaults), and shows how tier, employment type, and the missingness flag relate to default risk.
4. **Feature Engineering** — creates an EMI-to-loan ratio and one-hot encodes categories.
5. **Model Selection & Training** — compares three alternatives with reasoning:
   - Logistic Regression (`class_weight='balanced'`)
   - Random Forest (`class_weight='balanced'`)
   - HistGradientBoosting (chosen over XGBoost because it ships with scikit-learn)
6. **Model Evaluation & Tuning** — evaluates with precision/recall/F1, ROC-AUC, PR-AUC, confusion matrices and ROC curves (accuracy alone is misleading with 2.2% defaults). The best model by ROC-AUC (HistGradientBoosting) gets light tuning with `RandomizedSearchCV`.
7. **Using the Model on a New Applicant** — a `prepare_new_applicant()` helper runs one raw applicant through the exact same cleaning/feature/scaling steps, then `predict_applicant()` returns the default probability, prediction, and a 300-850 credit score. Two worked examples score a low-risk and a higher-risk borrower.

### Appendix (extension work)

- **A.1 Missingness map** — visual missing-value pattern using `missingno`.
- **A.2 Deletion vs. imputation** — trade-off between rows lost and data kept; imputation wins.
- **A.3 SMOTE vs. class-weighting** — resampling vs. reweighting; `class_weight='balanced'` wins on simplicity and stability.
- **A.4 Multicollinearity & linear separability** — VIF analysis, then PCA to show the classes overlap heavily (not linearly separable).
- **A.5 Generating a credit score** — maps default probability to a 300-850 score and sanity-checks that higher score bands have lower default rates.

## Key results

- Default rate: ~2.2% (2614 of 119,528 rows) → severe class imbalance.
- Row deletion (as in the starter notebook) would discard ~28.8% of the data; keeping + imputing is preferred.
- Best model by ROC-AUC: **HistGradientBoosting**; tuned with `RandomizedSearchCV` (10 iterations, 3-fold, ROC-AUC scoring).
- Evaluation uses precision/recall/F1 + ROC-AUC + PR-AUC because accuracy (~98%) looks great but hides the model's weak recall on actual defaulters.
- Live example: a low-risk applicant scores ~1.1% default probability (~844) vs. a higher-risk applicant at ~7.8% (~807).