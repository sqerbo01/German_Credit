import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model and returns it.
    """
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """
    Trains an XGBoost classifier and returns it.
    """
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model


def predict_model(model, X):
    """
    Returns model predictions on X.
    """
    return model.predict(X)


def save_submission(predictions, output_path='submission.csv'):
    """
    Saves predictions in the required Kaggle format with columns: Id, Risk.
    """
    submission = pd.DataFrame({
        'Id': np.arange(len(predictions)),
        'Risk': predictions
    })
    submission.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")
