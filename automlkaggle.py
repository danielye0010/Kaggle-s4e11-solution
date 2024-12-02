import pandas as pd
from autogluon.tabular import TabularPredictor

warnings.filterwarnings('ignore')
rs = 42

# Load datasets
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


train_data = df_train.copy()
test_data = df_test.copy()

train_data['label'] = train_data['Depression']
train_data = train_data.drop('Depression', axis=1)


if 'Depression' in test_data.columns:
    test_data = test_data.drop('Depression', axis=1)


columns_to_remove = ['id', 'Name', 'Degree', 'City', 'Profession', 'Working Professional or Student']
train_data = train_data.drop(columns=columns_to_remove, errors='ignore')
test_data = test_data.drop(columns=columns_to_remove, errors='ignore')


predictor = TabularPredictor(label='label', eval_metric='accuracy').fit(
    train_data=train_data,
    presets='medium_quality',
    num_bag_folds=5,
    num_stack_levels=1,
    time_limit=3600,
    verbosity=2
)

best_model = predictor.get_model_best()
print({best_model)

stacked_models = predictor.get_model_full_dict()[best_model]
print(stacked_models)
