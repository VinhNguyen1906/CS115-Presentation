import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from logistic_regression import LogisticRegression
credit_dataset = pd.read_csv('german_credit_data.csv')
scaler = StandardScaler()
name_columns = credit_dataset.columns.values
credit_dataset[name_columns[:-1]] = scaler.fit_transform(credit_dataset[name_columns[:-1]])
X = credit_dataset.drop(columns = 'kredit')
y = credit_dataset['kredit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy
regressor = LogisticRegression(lr=1e-5, n_iters=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print("LR classification accuracy:", accuracy(y_test, predictions))
