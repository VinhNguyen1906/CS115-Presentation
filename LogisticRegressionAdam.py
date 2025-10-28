import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
class LogisticRegressionAdam:
    def __init__(self, lr=0.001, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.costs = []
    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        n_samples, n_features = X.shape
        self.weights = torch.zeros((n_features, 1), dtype=torch.float32, requires_grad=True)
        self.bias = torch.zeros(1, dtype=torch.float32, requires_grad=True)
        optimizer = Adam([self.weights, self.bias], lr=self.lr)
        criterion = nn.BCELoss()
        for _ in range(self.epochs):
            y_pred = self._sigmoid(X @ self.weights + self.bias)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.costs.append(loss.item())
    def _sigmoid(self, x):
        return torch.sigmoid(x)
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            y_pred = self._sigmoid(X @ self.weights + self.bias)
        return [1 if i > 0.5 else 0 for i in y_pred]