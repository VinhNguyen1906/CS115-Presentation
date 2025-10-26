import numpy as np

class LogisticRegressionSGD:
    def __init__(self, lr=0.001, epochs=1000, alpha=0.0001, t0=1.0):
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha
        self.t0 = t0
        self.weights = None
        self.bias = None
        self.costs = []
    def _get_optimal_lr(self, epoch):
        return 1.0 / (self.alpha * (self.t0 + epoch))
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for epoch in range(self.epochs):
            lr_t = self._get_optimal_lr(epoch)
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            for i in range(n_samples):
                xi = X[i]
                yi = y[i]
                linear_model = np.dot(xi, self.weights) + self.bias
                y_pred_i = self._sigmoid(linear_model)
                dw = xi * (y_pred_i - yi)
                db = y_pred_i - yi
                self.weights -= lr_t * dw
                self.bias -= lr_t * db
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            cost = -(1/n_samples) * np.sum(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
            self.costs.append(cost)
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x)) 
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]

        return y_predicted_cls
