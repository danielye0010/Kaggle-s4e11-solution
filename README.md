# Kaggle S4E11 — Mental Health Classification

A feature-engineered ensemble solution for Kaggle Playground Series S4E11, **Exploring Mental Health Data**, combining domain-specific feature construction, gradient-boosted trees, Optuna tuning, stacking, and AutoML benchmarking.

The recorded manual competition workflow achieved **0.94360 accuracy and rank 294 / Top 10%**. Follow-up experiments reached a best recorded score of **0.94488**.

## Results

| Approach | Recorded accuracy | Result |
|---|---:|---|
| **Manual ML / competition workflow** | **0.94360** | **Rank 294 / Top 10%** |
| KANE AutoML benchmark | 0.93847 | Fast 5-minute AutoML baseline |
| **Custom ensemble experiment** | **0.94488** | Best recorded experimental score |
| Preprocessing + AutoGluon | 0.94477 | Strong AutoML experiment |

The manual workflow combines targeted preprocessing and feature engineering with XGBoost, CatBoost, LightGBM, Optuna, and a logistic-regression stacking layer.

## Competition task

The task is binary classification of the `Depression` target from demographic, academic, professional, lifestyle, and mental-health survey variables.

The project focuses on extracting useful structure from mixed numerical/categorical survey data rather than relying on a single off-the-shelf model.

## Feature engineering

Several task-specific features are constructed before model training:

- **Pressure** — combines `Academic Pressure` for students and `Work Pressure` for working professionals.
- **Satisfaction** — combines `Study Satisfaction` and `Job Satisfaction` into one role-aware feature.
- **PS ratio** — pressure divided by satisfaction.
- **PF factor** — pressure multiplied by financial stress.
- **Age × Work Pressure** — interaction capturing age-dependent work-pressure effects.
- **Rare-category filtering** — compresses low-frequency values in high-cardinality fields such as profession, city, and degree.
- **Target encoding** — adds compact representations for city and profession.

## Ensemble pipeline

`best-kaggle.py` implements the full manual workflow:

1. clean and combine student/professional features;
2. standardize numerical variables and encode categorical variables;
3. tune **XGBoost**, **CatBoost**, and **LightGBM** with Optuna;
4. use stratified 5-fold cross-validation during model selection;
5. combine the tuned models with a **logistic-regression stacking classifier**;
6. train the final ensemble and generate `submission.csv`.

The script supports either CUDA or CPU execution:

```bash
# GPU (default)
python best-kaggle.py

# CPU
KAGGLE_DEVICE=cpu python best-kaggle.py
```

On Windows PowerShell:

```powershell
$env:KAGGLE_DEVICE="cpu"
python best-kaggle.py
```

## AutoGluon benchmark

`automlkaggle.py` provides a compact AutoML comparison using AutoGluon Tabular with bagging and stacking. It trains for up to one hour and writes:

```text
submission_autogluon.csv
```

This provides a useful comparison between a manually engineered ensemble and a high-quality AutoML pipeline.

## Installation

```bash
pip install -r requirements.txt
```

Place the Kaggle competition files in the repository root:

```text
train.csv
test.csv
```

Competition data and generated submissions are intentionally excluded from Git.

## Run

Manual tuned ensemble:

```bash
python best-kaggle.py
```

AutoGluon benchmark:

```bash
python automlkaggle.py
```

## What this project demonstrates

- feature engineering on heterogeneous tabular data
- high-cardinality categorical handling
- gradient-boosting model selection
- Optuna hyperparameter optimization
- stacking ensembles
- GPU-accelerated tabular ML
- AutoML vs manually engineered workflow comparison
- end-to-end Kaggle submission generation

## Evaluation note

The **rank 294 / Top 10%** result is the recorded competition result for the manual workflow. The 0.94488 and 0.94477 values are retained as later experimental scores and are not presented as official 1st- or 4th-place leaderboard finishes.
