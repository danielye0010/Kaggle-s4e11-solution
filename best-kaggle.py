import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import optuna
import warnings

warnings.filterwarnings('ignore')
rs = 42

# Load datasets
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# distinguish train and test data
df_train['is_train'] = 1
df_test['is_train'] = 0

# Combine train and test data for consistent preprocessing
df_combined = pd.concat([df_train, df_test], ignore_index=True)

# Function to keep categories with more than a specified number of occurrences
def keep_frequent_categories(df, column_name, min_count=50):
    freq = df[column_name].value_counts()
    frequent_categories = freq[freq > min_count].index
    df[column_name] = df[column_name].where(df[column_name].isin(frequent_categories), other=np.nan)
    return df


# Process 'Profession', 'City', and 'Degree'
for col in ['Profession', 'City', 'Degree']:
    if col == 'Degree':
        df_combined[col] = df_combined[col].str.replace('.', '', regex=False)
    df_combined = keep_frequent_categories(df_combined, col)

# Create 'Pressure' feature based on 'Work Pressure' or 'Academic Pressure'
df_combined['Pressure'] = df_combined.apply(
    lambda row: row['Work Pressure'] if row['Working Professional or Student'] == 'working professional'
    else row['Academic Pressure'] if row['Working Professional or Student'] == 'student' else np.nan, axis=1
)

# Replace 'None' with NaN in 'Pressure'
df_combined['Pressure'].replace('None', np.nan, inplace=True)

# Impute missing values in 'Pressure' with the median
pressure_median = df_combined['Pressure'].median()
df_combined['Pressure'].fillna(pressure_median, inplace=True)

# Create 'Satisfaction' feature based on 'JobSatisfaction' or 'StudySatisfaction'
df_combined['Satisfaction'] = df_combined.apply(
    lambda row: row['JobSatisfaction'] if row['Working Professional or Student'] == 'working professional'
    else row['StudySatisfaction'] if row['Working Professional or Student'] == 'student' else np.nan, axis=1
)

# Replace 'None' with NaN
df_combined['Satisfaction'].replace('None', np.nan, inplace=True)

# Impute missing values in 'Satisfaction' with the median
satisfaction_median = df_combined['Satisfaction'].median()
df_combined['Satisfaction'].fillna(satisfaction_median, inplace=True)

# Create interaction features
df_combined['PS_ratio'] = df_combined['Pressure'] / df_combined['Satisfaction']
df_combined['PF_factor'] = df_combined['Pressure'] * df_combined['Financial Stress']

# Interaction between 'Age' and 'Work Pressure'
df_combined['Age_WorkPressure'] = df_combined['Age'] * df_combined['Work Pressure']

# Re-separate the data
df_train = df_combined[df_combined['is_train'] == 1].drop('is_train', axis=1)
df_test = df_combined[df_combined['is_train'] == 0].drop(['is_train', 'Depression'], axis=1)

# Target encoding for 'City' and 'Profession' using training data
encoder = TargetEncoder(cols=['City', 'Profession'])
df_train[['City_encoded', 'Profession_encoded']] = encoder.fit_transform(
    df_train[['City', 'Profession']], df_train['Depression']
)
df_test[['City_encoded', 'Profession_encoded']] = encoder.transform(df_test[['City', 'Profession']])

X_train = df_train.drop('Depression', axis=1)
y_train = df_train['Depression']
X_test = df_test.copy()

# Identify numerical and categorical columns
numerical_columns = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove columns not needed for modeling
columns_to_remove = ['id', 'Name', 'Degree', 'City', 'Profession', 'Working Professional or Student']
for col in columns_to_remove:
    if col in numerical_columns:
        numerical_columns.remove(col)
    if col in categorical_columns:
        categorical_columns.remove(col)

# Define preprocessing pipelines for numerical and categorical data
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('convert_to_float32', FunctionTransformer(lambda x: x.astype(np.float32)))
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('ordinal', OrdinalEncoder(dtype=np.int32, handle_unknown='use_encoded_value', unknown_value=-1))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_columns),
        ('cat', categorical_pipeline, categorical_columns)
    ]
)

# Apply preprocessing to training and test data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Define scoring metric
scoring = make_scorer(accuracy_score)

# Define cross-validation strategy
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
# Optimize XGBoost hyperparameters

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'use_label_encoder': False,
        'random_state': rs,
        'tree_method': 'gpu_hist'  # Use GPU if available
    }
    xgb_model = XGBClassifier(**params)
    # Set n_jobs=1 to prevent parallel processing conflicts with GPU
    scores = cross_val_score(xgb_model, X_train_preprocessed, y_train, cv=skf, scoring=scoring, n_jobs=-1)
    return scores.mean()

xgb_study = optuna.create_study(direction='maximize')
xgb_study.optimize(objective_xgb, n_trials=5)

# Optimize CatBoost hyperparameters
def objective_catboost(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 1000),  # Reduced upper limit to save memory
        'depth': trial.suggest_int('depth', 4, 8),  # Reduced range
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),  # Smaller range
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 128),  # Reduced upper limit
        'task_type': 'CPU',  # Use CPU to avoid GPU memory issues during tuning
        'verbose': 0,
        'random_state': rs,
    }
    cat_model = CatBoostClassifier(**params)
    # Use n_jobs=1 and CPU for CatBoost during tuning
    scores = cross_val_score(cat_model, X_train_preprocessed, y_train, cv=skf, scoring=scoring, n_jobs=1)
    return scores.mean()

catboost_study = optuna.create_study(direction='maximize')
catboost_study.optimize(objective_catboost, n_trials=5)

# Optimize LightGBM hyperparameters
def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'num_leaves': trial.suggest_int('num_leaves', 31, 128),  # Reduced upper limit
        'max_depth': trial.suggest_int('max_depth', -1, 15),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),  # Smaller range
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'random_state': rs,
        'n_jobs': -1
    }
    lgbm_model = LGBMClassifier(**params)
    scores = cross_val_score(lgbm_model, X_train_preprocessed, y_train, cv=skf, scoring=scoring, n_jobs=-1)
    return scores.mean()

lgbm_study = optuna.create_study(direction='maximize')
lgbm_study.optimize(objective_lgbm, n_trials=5)

# Get the best XGBoost model
best_xgb_params = xgb_study.best_params
best_xgb_params['use_label_encoder'] = False
best_xgb_params['random_state'] = rs
best_xgb_params['tree_method'] = 'gpu_hist'  # Use GPU if available
best_xgb_model = XGBClassifier(**best_xgb_params)

# Get the best CatBoost model
best_catboost_params = catboost_study.best_params
best_catboost_params['task_type'] = 'GPU'  # Use GPU for final training
best_catboost_params['verbose'] = 0
best_catboost_params['random_state'] = rs
best_catboost_model = CatBoostClassifier(**best_catboost_params)

# Get the best LightGBM model
best_lgbm_params = lgbm_study.best_params
best_lgbm_params['random_state'] = rs
best_lgbm_model = LGBMClassifier(**best_lgbm_params)

# Create stacking ensemble model
stacking_ensemble = StackingClassifier(
    estimators=[
        ('xgb', best_xgb_model),
        ('catboost', best_catboost_model),
        ('lgbm', best_lgbm_model)
    ],
    final_estimator=LogisticRegression(),
    cv=skf,
    passthrough=False,
    n_jobs=1
)

# Fit the stacking ensemble on the entire training data
stacking_ensemble.fit(X_train_preprocessed, y_train)

# Make predictions on the test data
test_preds = stacking_ensemble.predict(X_test_preprocessed)

# Prepare submission DataFrame
submission = pd.DataFrame({'id': df_test['id'], 'class': test_preds})

# Save predictions to CSV file
submission.to_csv('submission.csv', index=False)