import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from LogisticRegressionSGD import LogisticRegressionSGD
import matplotlib.pyplot as plt
data = load_breast_cancer()
X, y = data.data, data.target
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy
regressor = LogisticRegressionSGD(epochs=1000, random_state=42)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print("LR SGD classification accuracy:", accuracy(y_test, predictions))
plt.plot(range(len(regressor.costs)), regressor.costs)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Stochastic Gradient Descent')
plt.show()


