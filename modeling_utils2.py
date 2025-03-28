
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

def custom_metric(y_true: pd.Series, y_pred: pd.Series, loan_amounts: pd.Series) -> float:
    """
    Computes the competition metric.
    y_true/y_pred: 'Risk'=1, 'No Risk'=0
    """
    costs = {
        'Risk_No Risk': 5.0 + 0.6 * loan_amounts,
        'No Risk_No Risk': 1.0 - 0.05 * loan_amounts,
        'Risk_Risk': 1.0,
        'No Risk_Risk': 1.0
    }
    real_prop = {'Risk': 0.02, 'No Risk': 0.98}
    train_prop = {'Risk': 1/3, 'No Risk': 2/3}
    custom_weight = {
        'Risk': real_prop['Risk']/train_prop['Risk'],
        'No Risk': real_prop['No Risk']/train_prop['No Risk']
    }
    
    loss = (
        (y_true == 1) * custom_weight['Risk'] * (
            (y_pred == 1) * costs['Risk_Risk'] + 
            (y_pred == 0) * costs['Risk_No Risk']
        ) +
        (y_true == 0) * custom_weight['No Risk'] * (
            (y_pred == 1) * costs['No Risk_Risk'] + 
            (y_pred == 0) * costs['No Risk_No Risk']
        )
    )
    return loss.mean()


def find_optimal_threshold(y_probs: np.array, y_true: pd.Series, loan_amounts: pd.Series) -> float:
    """Finds threshold that maximizes custom metric"""
    thresholds = np.linspace(0.1, 0.5, 50)
    scores = [
        custom_metric(y_true, (y_probs > t).astype(int), loan_amounts) 
        for t in thresholds
    ]
    return thresholds[np.argmax(scores)]

def evaluate_model(model, X, y, loan_amounts, n_splits=5):
    """
    Returns dictionary with:
    - 'mean_score': Average CV score
    - 'std_score': Standard deviation of scores  
    - 'optimal_threshold': Mean best threshold
    - 'models': List of trained models (one per fold)
    - 'cv_scores': List of all fold scores  # This was missing before
    """
    skf = StratifiedKFold(n_splits=n_splits)
    cv_scores, models, thresholds = [], [], []
    
    for train_idx, val_idx in skf.split(X, y):
        # Train model
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        models.append(model)
        
        # Predict and optimize threshold
        y_probs = model.predict_proba(X.iloc[val_idx])[:, 1]
        thresh = find_optimal_threshold(y_probs, y.iloc[val_idx], loan_amounts.iloc[val_idx])
        thresholds.append(thresh)
        
        # Score
        score = custom_metric(y.iloc[val_idx], (y_probs > thresh).astype(int), loan_amounts.iloc[val_idx])
        cv_scores.append(score)
    
    return {
        'mean_score': np.mean(cv_scores),
        'std_score': np.std(cv_scores),
        'optimal_threshold': np.mean(thresholds),
        'models': models,
        'cv_scores': cv_scores  # Now explicitly returned
    }