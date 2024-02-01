import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def preprocess(data):
    data.dropna()
    features = ['TotalBsmtSF', 'GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    features = data[features]
    target = data['SalePrice']
    return train_test_split(features, target, test_size=0.3, random_state=42)


def train(model, xTrain, yTrain):
    print('\nTraining')
    print('--------------------------------')
    xTrain.info()
    model.fit(xTrain, yTrain)
    return model


def predict(model, xTest):
    print('\nPredicting')
    print('----------------------------')
    predictions = model.predict(xTest)
    print(pd.DataFrame(predictions))
    return model, predictions


def showMetrics(yTest, predictions):
    print('\n----------------------------')
    mse = mean_squared_error(yTest, predictions)
    r2 = r2_score(yTest, predictions)
    evs = explained_variance_score(yTest, predictions)
    print(f'R-squared: {r2:.6f}')
    print(f'Mean Squared Error: {round(mse)}')
    print(f'Explained Variance Score: {evs:.6f}')
    print(f'Normalized Mean Squared Error (% of Variance): {mse / np.var(yTest) * 100 :.2f}%')
    print(f'Root Mean Squared Error (% of Standard Deviation): {np.sqrt(mse) / np.std(yTest) * 100 :.2f}%')


def showPlot(yTest, predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(yTest, predictions)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual Prices vs Predicted Prices")
    plt.show()


model = LinearRegression()
data = pd.read_csv(os.path.join("dataset", "houseSales.csv"))
xTrain, xTest, yTrain, yTest = preprocess(data)
model = train(model, xTrain, yTrain)
model, predictions = predict(model, xTest)
showMetrics(yTest, predictions)
showPlot(yTest, predictions)
