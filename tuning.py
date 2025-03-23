'''With this tuning, the score on Kaggle went from -79.82 to -81.97'''


import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from feature_engineering import engineer_features
from metrics_k import score

# Load and preprocess the dataset
df = pd.read_csv('dsb-24-german-credit/german_credit_train.csv')
df = engineer_features(df)
X = df.drop(columns=['Risk'])
y = df['Risk'].map({'No Risk': 0, 'Risk': 1})

# One-hot encoding of categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function for hyperparameter optimization
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    # Train the model with current hyperparameters
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)


    # Make predictions on validation set
    y_pred = model.predict(X_val)

    # Map predictions and ground truth back to string labels
    y_val_pred_labels = pd.Series(y_pred).map({0: 'No Risk', 1: 'Risk'})
    y_val_labels = y_val.map({0: 'No Risk', 1: 'Risk'})

    # Prepare dataframes for custom scoring
    df_val_true = pd.DataFrame({'Risk': y_val_labels, 'LoanAmount': df.loc[y_val.index, 'LoanAmount']})
    df_val_pred = pd.DataFrame({'Risk': y_val_pred_labels})

    # Compute custom cost metric (lower is better)
    cost = score(df_val_true, df_val_pred, row_id_column_name=None)
    return cost  # Minimizing this cost

# Create the Optuna study and run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

# Print best results
print("Best parameters found:")
print(study.best_params)
print("Best cost (lower is better):")
print(study.best_value)