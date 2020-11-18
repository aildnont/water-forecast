from abc import ABCMeta, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from joblib import dump
import pandas as pd
import numpy as np
import os
from src.models.model import ModelStrategy

class SKLearnModel(ModelStrategy):
    '''
    A class for a model defined using scikit-learn and the standard operations on it
    '''
    __metaclass__ = ABCMeta

    def __init__(self, hparams, name, log_dir):
        self.univariate = hparams.get('UNIVARIATE', False)
        self.T_x = int(hparams.get('T_X', 32))
        self.standard_scaler = StandardScaler()
        model = None
        super(SKLearnModel, self).__init__(model, self.univariate, name, log_dir=log_dir)


    @abstractmethod
    def define_model(self):
        '''
        Abstract method for TensorFlow model definition
        '''
        pass


    def fit(self, dataset):
        '''
        Fits an RNN forecasting model
        :param dataset: A Pandas DataFrame with feature columns and a Consumption column
        '''
        train_df = dataset.copy()
        if self.univariate:
            train_df = train_df[['Date', 'Consumption']]
        train_df.loc[:, train_df.columns != 'Date'] = self.standard_scaler.fit_transform(train_df.loc[:, train_df.columns != 'Date'])

        # Make time series datasets
        train_dates, X_train, Y_train = self.make_windowed_dataset(train_df)
        self.n_pred_feats = Y_train.shape[1]

        self.model = self.define_model()  # Define model
        self.model.fit(X_train, Y_train)  # Fit model
        return


    def evaluate(self, train_set, test_set, save_dir=None):
        '''
        Evaluates performance of RNN model on test set
        :param train_set: A Pandas DataFrame with feature columns and a Consumption column
        :param test_set: A Pandas DataFrame with feature columns and a Consumption column
        '''

        if self.univariate:
            train_set = train_set[['Date', 'Consumption']]
            test_set = test_set[['Date', 'Consumption']]

        train_set.loc[:, train_set.columns != 'Date'] = self.standard_scaler.transform(train_set.loc[:, train_set.columns != 'Date'])
        test_set.loc[:, test_set.columns != 'Date'] = self.standard_scaler.transform(test_set.loc[:, test_set.columns != 'Date'])

        # Create windowed versions of the training and test sets
        consumption_idx = train_set.drop('Date', axis=1).columns.get_loc('Consumption')   # Index of consumption feature
        train_dates, X_train, Y_train = self.make_windowed_dataset(train_set)
        test_pred_dates, X_test, Y_test = self.make_windowed_dataset(pd.concat([train_set[-self.T_x:], test_set]))
        test_dates = test_set['Date']

        # Make predictions for training set and obtain forecast for test set
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)
        if len(train_preds.shape) < 2:
            train_preds = np.expand_dims(train_preds, axis=1)
            test_preds = np.expand_dims(test_preds, axis=1)
        test_forecast = self.forecast(test_dates.shape[0], recent_data=X_train[-1])

        # Rescale data
        train_set.loc[:, train_set.columns != 'Date'] = self.standard_scaler.inverse_transform(train_set.loc[:, train_set.columns != 'Date'])
        test_set.loc[:, test_set.columns != 'Date'] = self.standard_scaler.inverse_transform(test_set.loc[:, test_set.columns != 'Date'])
        train_preds = self.standard_scaler.inverse_transform(train_preds)
        test_preds = self.standard_scaler.inverse_transform(test_preds)
        test_forecast = self.standard_scaler.inverse_transform(test_forecast)

        # Create a DataFrame of combined training set predictions and test set forecast with ground truth
        df_train = pd.DataFrame({'ds': train_dates, 'gt': train_set.iloc[self.T_x:]['Consumption'],
                                 'model': train_preds[:,consumption_idx]})
        df_test = pd.DataFrame({'ds': test_dates, 'gt': test_set['Consumption'],
                                 'forecast': test_forecast[:,consumption_idx], 'test_pred': test_preds[:,consumption_idx]})
        df_forecast = df_train.append(df_test)

        # Compute evaluation metrics for the forecast
        test_metrics = self.evaluate_forecast(df_forecast, save_dir=save_dir)
        return test_metrics


    def forecast(self, days, recent_data=None):
        '''
        Create a forecast for the test set. Note that this is different than obtaining predictions for the test set.
        The model makes a prediction for the provided example, then uses the result for the next prediction.
        Repeat this process for a specified number of days.
        :param days: Number of days into the future to produce a forecast for
        :param recent_data: A factual example for the first prediction
        :return: An array of predictions
        '''
        if recent_data is None:
            raise Exception('Time series forecast requires input of shape (T_x, features).')
        if not (self.n_pred_feats and self.model):
            raise Exception('You must fit a model before forecasting.')
        preds = np.zeros((days, self.n_pred_feats))
        x = recent_data
        for i in range(days):
            y = self.model.predict(np.expand_dims(x, axis=0))
            x = np.roll(x, -self.n_pred_feats)
            preds[i] = y
            x[-self.n_pred_feats:] = y   # Prediction becomes latest data in the example
        return preds


    def save_model(self, save_dir):
        '''
        Saves the model to disk
        :param save_dir: Directory in which to save the model
        '''
        if self.model:
            model_path = os.path.join(save_dir, self.name + self.train_date + '.joblib')
            dump(self.model, model_path)  # Serialize and save the model object


    def make_windowed_dataset(self, dataset):
        '''
        Make time series datasets. Each example is a window of the last T_x data points and label is data point 1 day
        into the future.
        :param dataset: Pandas DataFrame indexed by date
        :return: A windowed time series dataset of shape (# rows, T_x, # features)
        '''
        dates = dataset['Date'][self.T_x:]
        unindexed_dataset = dataset.loc[:, dataset.columns != 'Date']
        X = np.zeros((unindexed_dataset.shape[0] - self.T_x, self.T_x, unindexed_dataset.shape[1]))
        Y = unindexed_dataset[self.T_x:].to_numpy()
        for i in range(X.shape[0]):
            X[i] = unindexed_dataset[i:i+self.T_x].to_numpy()
        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
        return dates, X, Y



class LinearRegressionModel(SKLearnModel):
    '''
    A class for an ordinary least squares linear regression model
    '''

    def __init__(self, hparams, log_dir=None):
        name = 'LinearRegression'
        super(LinearRegressionModel, self).__init__(hparams, name, log_dir)

    def define_model(self):
        return LinearRegression()

    def fit(self, dataset):
        super(LinearRegressionModel, self).fit(dataset)
        print('R^2 = ', self.model.score)



class RandomForestModel(SKLearnModel):
    '''
    A class for a random forest regression model
    '''

    def __init__(self, hparams, log_dir=None):
        name = 'RandomForest'
        self.n_estimators = int(hparams.get('N_ESTIMATORS', 100))
        self.loss = hparams.get('LOSS', 'mse') if hparams.get('LOSS', 'mse') in ['mse', 'mae'] else 'mse'
        super(RandomForestModel, self).__init__(hparams, name, log_dir)

    def define_model(self):
        return RandomForestRegressor(n_estimators=self.n_estimators, criterion=self.loss)

    def fit(self, dataset):
        super(RandomForestModel, self).fit(dataset)