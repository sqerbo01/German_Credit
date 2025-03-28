import pandas as pd
import numpy as np

def compute_costs(LoanAmount):
    return {
        'Risk_No Risk': 5.0 + 0.6 * LoanAmount,
        'No Risk_No Risk': 1.0 - 0.05 * LoanAmount,
        'Risk_Risk': 1.0,
        'No Risk_Risk': 1.0
    }

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = None) -> float:
    '''
    Custom cost-based metric for the German credit dataset.
    '''
    real_prop = {'Risk': .02, 'No Risk': .98}
    train_prop = {'Risk': 1/3, 'No Risk': 2/3}
    custom_weight = {
        'Risk': real_prop['Risk'] / train_prop['Risk'],
        'No Risk': real_prop['No Risk'] / train_prop['No Risk']
    }
    
    costs = compute_costs(solution['LoanAmount'])
    y_true = solution['Risk']
    y_pred = submission['Risk']
    
    loss = (
        (y_true == 'Risk') * custom_weight['Risk'] *
        ((y_pred == 'Risk') * costs['Risk_Risk'] + (y_pred == 'No Risk') * costs['Risk_No Risk']) +
        (y_true == 'No Risk') * custom_weight['No Risk'] *
        ((y_pred == 'Risk') * costs['No Risk_Risk'] + (y_pred == 'No Risk') * costs['No Risk_No Risk'])
    )
    
    return loss.mean()


from modeling_utils import score  # Or whatever your import path is

def evaluate_model_custom_score(model, X_val, y_val, loan_amounts):
    """
    Evaluate the custom competition score for a trained model.

    Parameters:
    - model: trained model with a .predict() method
    - X_val: validation features
    - y_val: true labels (0/1)
    - loan_amounts: corresponding loan amounts (Series)

    Returns:
    - custom_score: float
    """
    # Predict
    y_pred = model.predict(X_val)

    # Format into DataFrames expected by the `score()` function
    solution_df = pd.DataFrame({
        'Risk': y_val.map({1: 'Risk', 0: 'No Risk'}),
        'LoanAmount': loan_amounts
    })

    submission_df = pd.DataFrame({
        'Risk': pd.Series(y_pred).map({1: 'Risk', 0: 'No Risk'}).values,
        'LoanAmount': loan_amounts
    })

    # Compute and return score
    return score(solution_df, submission_df)




import pandas as pd
from feature_engineering import engineer_features, feature_selection

def generate_kaggle_submission(model, test_csv_path, X_train_columns, output_path="german_credit_submission.csv"):
    """
    Generates a Kaggle submission file using a trained model.

    Parameters:
    - model: Trained classifier (e.g., best_xgb)
    - test_csv_path (str): Path to the raw test CSV file
    - X_train_columns (list): Column names used during training (to align test features)
    - output_path (str): Output filename for the CSV submission

    Returns:
    - submission_df (pd.DataFrame): The final submission DataFrame
    """
    # Load test data
    test_df = pd.read_csv(test_csv_path)

    # Add dummy Id column if not already present
    if 'Id' not in test_df.columns:
        test_df['Id'] = range(1, len(test_df) + 1)

    # Feature engineering & selection
    test_fe = engineer_features(test_df)
    test_final = feature_selection(test_fe)

    # Drop LoanTier if it exists
    test_final = test_final.drop(columns=['LoanTier'], errors='ignore')

    # Align test features with training features
    test_final = test_final.reindex(columns=X_train_columns, fill_value=0)

    # Predict class labels
    preds = model.predict(test_final)
    preds_labels = pd.Series(preds).map({1: "Risk", 0: "No Risk"})

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        "Id": test_df["Id"],
        "Risk": preds_labels
    })

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"✅ Submission file saved: {output_path}")

    return submission_df


from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.base import clone

def crossval_custom_score_general(df, FEATURES, TARGET, model, threshold=0.5, k=5, verbose=True):
    """
    Performs stratified K-Fold CV for any model with .fit() and .predict_proba(),
    and returns the average custom competition score using thresholding.

    Parameters:
    - df: full DataFrame (must include 'LoanAmount')
    - FEATURES: list of feature columns
    - TARGET: string, name of the target column
    - model: an untrained model instance (will be cloned and retrained per fold)
    - threshold: float (e.g., 0.5). Applied to predicted probabilities
    - k: number of cross-validation folds
    - verbose: print per-fold results

    Returns:
    - scores: list of per-fold scores
    - mean_score: average custom score
    """

    X = df[FEATURES].drop(columns=["LoanTier"], errors="ignore").reset_index(drop=True)
    y = df[TARGET].reset_index(drop=True)
    loan_amounts = df["LoanAmount"].reset_index(drop=True)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        loan_val = loan_amounts.iloc[val_idx]

        # Clone and fit model
        clf = clone(model)
        clf.fit(X_train, y_train)

        # Predict probabilities and apply threshold
        y_proba = clf.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

                # Prepare DataFrames for scoring — with aligned indices
        solution_df = pd.DataFrame({
            'Risk': pd.Series(y_val.values, index=loan_val.index).map({1: 'Risk', 0: 'No Risk'}),
            'LoanAmount': loan_val
        })

        submission_df = pd.DataFrame({
            'Risk': pd.Series(y_pred, index=loan_val.index).map({1: 'Risk', 0: 'No Risk'}),
            'LoanAmount': loan_val
        })

        fold_score = score(solution_df, submission_df)
        scores.append(fold_score)

        if verbose:
            print(f"✅ Fold {fold + 1} — Custom Competition Score: {fold_score:.4f}")

    mean_score = np.mean(scores)
    print(f"\n📊 Mean Custom Competition Score (across {k} folds): {mean_score:.4f}")
    return scores, mean_score

from sklearn.model_selection import train_test_split

def create_splits(X, y, loan_amounts, test_size=0.1, random_state=1):
    """
    Creates stratified train/holdout splits preserving all relationships
    
    Returns:
        X_train, X_val, y_train, y_val, amt_train, amt_val
    """
    X_train, X_val, y_train, y_val, amt_train, amt_val = train_test_split(
        X, y, loan_amounts,
        test_size=test_size,
        stratify=y,  # Preserves class balance
        random_state=random_state
    )
    return X_train, X_val, y_train, y_val, amt_train, amt_val