import os
import warnings

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from category_encoders import TargetEncoder
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
rs = 42
DEVICE = os.getenv("KAGGLE_DEVICE", "cuda").lower()
USE_GPU = DEVICE == "cuda"

# Load datasets
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# Distinguish train and test data
df_train["is_train"] = 1
df_test["is_train"] = 0

# Combine train and test data for consistent non-target preprocessing
df_combined = pd.concat([df_train, df_test], ignore_index=True)


def keep_frequent_categories(df, column_name, min_count=50):
    """Keep common categories and map rare values to missing."""
    freq = df[column_name].value_counts()
    frequent_categories = freq[freq > min_count].index
    df[column_name] = df[column_name].where(
        df[column_name].isin(frequent_categories), other=np.nan
    )
    return df


# Process high-cardinality categorical features
for col in ["Profession", "City", "Degree"]:
    if col == "Degree":
        df_combined[col] = df_combined[col].str.replace(".", "", regex=False)
    df_combined = keep_frequent_categories(df_combined, col)

# Combine student/professional pressure into one feature
df_combined["Pressure"] = df_combined.apply(
    lambda row: row["Work Pressure"]
    if row["Working Professional or Student"] == "Working Professional"
    else row["Academic Pressure"]
    if row["Working Professional or Student"] == "Student"
    else np.nan,
    axis=1,
)
df_combined["Pressure"].replace("None", np.nan, inplace=True)
df_combined["Pressure"].fillna(df_combined["Pressure"].median(), inplace=True)

# Combine student/professional satisfaction into one feature
df_combined["Satisfaction"] = df_combined.apply(
    lambda row: row["Job Satisfaction"]
    if row["Working Professional or Student"] == "Working Professional"
    else row["Study Satisfaction"]
    if row["Working Professional or Student"] == "Student"
    else np.nan,
    axis=1,
)
df_combined["Satisfaction"].replace("None", np.nan, inplace=True)
df_combined["Satisfaction"].fillna(
    df_combined["Satisfaction"].median(), inplace=True
)

# Interaction features. Treat zero satisfaction as missing for the ratio so the
# downstream median imputer handles it instead of producing infinity.
satisfaction_denominator = df_combined["Satisfaction"].replace(0, np.nan)
df_combined["PS_ratio"] = df_combined["Pressure"] / satisfaction_denominator
df_combined["PF_factor"] = (
    df_combined["Pressure"] * df_combined["Financial Stress"]
)
df_combined["Age_WorkPressure"] = (
    df_combined["Age"] * df_combined["Work Pressure"]
)

# Re-separate train and test
df_train = df_combined[df_combined["is_train"] == 1].drop("is_train", axis=1)
df_test = df_combined[df_combined["is_train"] == 0].drop(
    ["is_train", "Depression"], axis=1
)

# Target encoding for high-cardinality categories, fitted on training data only
encoder = TargetEncoder(cols=["City", "Profession"])
df_train[["City_encoded", "Profession_encoded"]] = encoder.fit_transform(
    df_train[["City", "Profession"]], df_train["Depression"]
)
df_test[["City_encoded", "Profession_encoded"]] = encoder.transform(
    df_test[["City", "Profession"]]
)

X_train = df_train.drop("Depression", axis=1)
y_train = df_train["Depression"]
X_test = df_test.copy()

numerical_columns = X_train.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_columns = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

columns_to_remove = [
    "id",
    "Name",
    "Degree",
    "City",
    "Profession",
    "Working Professional or Student",
]
for col in columns_to_remove:
    if col in numerical_columns:
        numerical_columns.remove(col)
    if col in categorical_columns:
        categorical_columns.remove(col)

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("convert_to_float32", FunctionTransformer(lambda x: x.astype(np.float32))),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        (
            "ordinal",
            OrdinalEncoder(
                dtype=np.int32,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns),
    ]
)

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

scoring = make_scorer(accuracy_score)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)


def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": rs,
        "tree_method": "hist",
        "device": "cuda" if USE_GPU else "cpu",
    }
    xgb_model = XGBClassifier(**params)
    scores = cross_val_score(
        xgb_model,
        X_train_preprocessed,
        y_train,
        cv=skf,
        scoring=scoring,
        n_jobs=1 if USE_GPU else -1,
    )
    return scores.mean()


xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(objective_xgb, n_trials=5)


def objective_catboost(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 1000),
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 128),
        "task_type": "CPU",
        "verbose": 0,
        "random_state": rs,
    }
    cat_model = CatBoostClassifier(**params)
    scores = cross_val_score(
        cat_model,
        X_train_preprocessed,
        y_train,
        cv=skf,
        scoring=scoring,
        n_jobs=1,
    )
    return scores.mean()


catboost_study = optuna.create_study(direction="maximize")
catboost_study.optimize(objective_catboost, n_trials=5)


def objective_lgbm(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
        "num_leaves": trial.suggest_int("num_leaves", 31, 128),
        "max_depth": trial.suggest_int("max_depth", -1, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": rs,
        "n_jobs": -1,
    }
    lgbm_model = LGBMClassifier(**params)
    scores = cross_val_score(
        lgbm_model,
        X_train_preprocessed,
        y_train,
        cv=skf,
        scoring=scoring,
        n_jobs=-1,
    )
    return scores.mean()


lgbm_study = optuna.create_study(direction="maximize")
lgbm_study.optimize(objective_lgbm, n_trials=5)

best_xgb_params = xgb_study.best_params
best_xgb_params.update(
    {
        "random_state": rs,
        "tree_method": "hist",
        "device": "cuda" if USE_GPU else "cpu",
    }
)
best_xgb_model = XGBClassifier(**best_xgb_params)

best_catboost_params = catboost_study.best_params
best_catboost_params.update(
    {
        "task_type": "GPU" if USE_GPU else "CPU",
        "verbose": 0,
        "random_state": rs,
    }
)
best_catboost_model = CatBoostClassifier(**best_catboost_params)

best_lgbm_params = lgbm_study.best_params
best_lgbm_params["random_state"] = rs
best_lgbm_model = LGBMClassifier(**best_lgbm_params)

stacking_ensemble = StackingClassifier(
    estimators=[
        ("xgb", best_xgb_model),
        ("catboost", best_catboost_model),
        ("lgbm", best_lgbm_model),
    ],
    final_estimator=LogisticRegression(),
    cv=skf,
    passthrough=False,
    n_jobs=1,
)

stacking_ensemble.fit(X_train_preprocessed, y_train)
test_preds = stacking_ensemble.predict(X_test_preprocessed)

submission = pd.DataFrame(
    {"id": df_test["id"], "Depression": test_preds.astype(int)}
)
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
