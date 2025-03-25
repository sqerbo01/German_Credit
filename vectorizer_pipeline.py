import pandas as pd
import numpy as np
from dirty_cat import TableVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from metrics_k import score
from feature_engineering import engineer_features

# Load dataset
df = pd.read_csv('dsb-24-german-credit/german_credit_train.csv')
df = engineer_features(df)

X = df.drop(columns=['Risk'])
y = df['Risk'].map({'No Risk': 0, 'Risk': 1})

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TableVectorizer
tv = TableVectorizer()

# Fit-transform on training set, transform on validation
X_train_vec = tv.fit_transform(X_train)
X_val_vec = tv.transform(X_val)

# Train XGBoost model on vectorized features
model = XGBClassifier(
    max_depth=4,
    learning_rate=0.019545073364830568,
    n_estimators=111,
    subsample=0.943450282452676,
    colsample_bytree=0.7354447358381201,
    gamma=2.04067902399041,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train_vec, y_train)

# Predict and evaluate on validation
y_val_pred = model.predict(X_val_vec)
y_val_pred_labels = pd.Series(y_val_pred).map({0: 'No Risk', 1: 'Risk'})
y_val_labels = y_val.map({0: 'No Risk', 1: 'Risk'})

df_val_true = pd.DataFrame({'Risk': y_val_labels, 'LoanAmount': df.loc[y_val.index, 'LoanAmount']})
df_val_pred = pd.DataFrame({'Risk': y_val_pred_labels})

val_cost = score(df_val_true, df_val_pred, row_id_column_name=None)
print(f"Validation cost with TableVectorizer: {val_cost}")