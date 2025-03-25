import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model and returns it.
    """
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)
    return model


'''def train_xgboost(X_train, y_train):
    """
    Trains an XGBoost classifier and returns it.
    """
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model'''


def train_xgboost(X_train, y_train):
    """
    Trains an XGBoost classifier with tuned hyperparameters and returns it.
    """
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


class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

def train_neural_network(X_train, y_train, num_epochs=100, batch_size=64, learning_rate=0.001):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = NeuralNet(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    model.cpu()
    return model, scaler

def predict_neural_network(model, scaler, X):
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().flatten()
    return (preds > 0.5).astype(int)