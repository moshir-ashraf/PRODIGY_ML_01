import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


class houseSales:
    def __init__(self, files):
        self.model = LinearRegression()
        dataFrames = []
        for file in files:
            dataFrames.append(pd.read_csv(os.path.join('dataset', file)))
        self.xTrain = dataFrames[0].drop('SalePrice', axis=1)
        self.yTrain = dataFrames[0]['SalePrice']
        self.xTest = dataFrames[1]
        self.yTest = dataFrames[2]['SalePrice']
        self.preprocess()

    def preprocess(self):
        self.firstTestingId = self.xTest['Id'].iloc[0]
        features = ['TotalSF', 'Bath', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
        self.xTrain['Bath'] = self.xTrain['FullBath'] + self.xTrain['BsmtFullBath'] \
                              + (self.xTrain['HalfBath'] * 0.5) + (self.xTrain['BsmtHalfBath'] * 0.5)
        self.xTrain['TotalSF'] = self.xTrain['TotalBsmtSF'] + self.xTrain['GrLivArea']
        self.xTest = self.xTest.fillna(0)
        self.xTest['Bath'] = self.xTest['FullBath'] + self.xTest['BsmtFullBath'] \
                             + (self.xTest['HalfBath'] * 0.5) + (self.xTest['BsmtHalfBath'] * 0.5)
        self.xTest['TotalSF'] = self.xTest['TotalBsmtSF'] + self.xTest['GrLivArea']
        self.xTrain = self.xTrain[features]
        self.xTest = self.xTest[features]

    def train(self):
        print('\nTraining')
        print('--------------------------------')
        self.xTrain.info()
        self.model.fit(self.xTrain, self.yTrain)

    def predict(self):
        print('\nPredicting')
        print('----------------------------')
        self.predictions = self.model.predict(self.xTest)
        data = {
            'Id': range(self.firstTestingId, len(self.predictions) + self.firstTestingId),
            'SalePrice': self.predictions
        }
        print(pd.DataFrame(data))

    def showMetrics(self):
        mse = mean_squared_error(self.yTest, self.predictions)
        r2 = r2_score(self.yTest, self.predictions)
        print(f'Mean Squared Error: {mse}')
        print(f'R-squared: {r2}')

    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.yTest, self.predictions)
        plt.xlabel("Actual Prices")
        plt.ylabel("Predicted Prices")
        plt.title("Actual Prices vs Predicted Prices")
        plt.show()


files = ['train.csv', 'test.csv', 'sample_submission.csv']
model = houseSales(files)
model.train()
model.predict()
model.showMetrics()
model.plot()
