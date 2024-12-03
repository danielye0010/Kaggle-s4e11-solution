# Kaggle S4E11 solution

The competition, Exploring Mental Health Data comes from Kaggle Playground S4E11, contains a synthetic dataset designed to identify factors contributing to the risk of depression. This competition is active from Nov 1, 2024 to Nov 30, 2024, with 2,891 Participants, 23,174 Submissions. 

---

## Workflow Components

### 1. Tailored Data Preprocessing
Custom preprocessing was designed to address dataset-specific characteristics while minimizing information loss:

- **Missing Numerical Values**: Imputed using the median to ensure robustness against outliers.
- **Missing Categorical Values**: Imputed with the label `'missing'` to retain the absence of information as a separate category. For variables like `Academic Pressure`, `Work Pressure`, `Job Satisfaction`, and `Study Satisfaction`, missing values were left unaltered to preserve implicit patterns.
- **Frequent Category Retention**: Categories with fewer than 50 occurrences were replaced with `'Na'` for variables such as `Profession`, `City`, and `Degree`, reducing dimensionality.
- **Data Cleaning**: Text entries in categorical variables were standardized to ensure consistency (e.g., unifying `BS` and `B.S.`).

---

### 2. Advanced Feature Engineering
Feature engineering was implemented to enhance the dataset's predictive power by creating meaningful, domain-specific features:

- **Unified Features**:
  - `Pressure`: Combined `Academic Pressure` (students) and `Work Pressure` (professionals) into a single variable.
  - `Satisfaction`: Merged `Study Satisfaction` and `Job Satisfaction` into one variable to capture overall contentment.
- **Feature Removal**:
  - Removed variables like `Name` and `ID` as they do not contribute to the prediction task.
- **New Interaction Features**:
  - `PS Ratio`: \(\text{Pressure} / \text{Satisfaction}\), capturing the balance between stress and satisfaction.
  - `PF Factor`: \(\text{Pressure} \times \text{Financial Stress}\), reflecting the compounded impact of stress and financial pressure.
  - `Age_WorkPressure`: \(\text{Age} \times \text{Work Pressure}\), exploring age-related differences in stress response.
- **Target Encoding**:
  - Applied to variables with high cardinality, such as `City` and `Profession`, replacing categories with the mean target (`Depression`) value, reducing feature dimensionality.

---

### 3. Model Training
Gradient-boosted decision tree models were selected for their performance and ability to handle heterogeneous data:

- **Models Used**:
  - **XGBoost**: Optimized for scalability and structured datasets.
  - **CatBoost**: Effective at handling categorical data with minimal preprocessing.
  - **LightGBM**: Memory-efficient and fast for large datasets.
- **Stacking**:
  - Predictions from individual models were combined using a logistic regression meta-model to leverage the strengths of each base model.
- **Cross-Validation**:
  - A 5-fold cross-validation strategy was used to ensure robust evaluation and reduce overfitting.
- **Hyperparameter Tuning**:
  - The Optuna library was utilized for efficient exploration of hyperparameters like learning rate, tree depth, and estimators to maximize cross-validation accuracy.

---

## Results

The manual ML workflow demonstrated strong performance on the Kaggle leaderboard:

| Approach                    | Training Time | Accuracy  | Leaderboard Rank |
|-----------------------------|---------------|-----------|------------------|
| **Manual ML**               | 60 minutes    | 0.94360   | Top 10% (294/2891) |
| **KANE (AutoML)**           | 5 minutes     | 0.93847   | Top 57% (1645/2891) |
| **Custom Ensemble**         | N/A           | 0.94488   | 1st Place         |
| **Preprocessing + AutoGluon** | N/A         | 0.94477   | 4th Place         |


