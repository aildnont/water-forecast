import pmdarima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
from src.models.model import ModelStrategy

class SARIMAModel(ModelStrategy):
    '''
    A class for a Seasonal Autoregressive Integrated Moving Average Model and the standard operations on it
    '''

    def __init__(self, hparams, log_dir=None):
        univariate = True
        model = None
        name = 'SARIMA'
        self.auto_params = hparams.get('AUTO_PARAMS', False)
        self.trend_p = int(hparams.get('TREND_P', 10))
        self.trend_d = int(hparams.get('TREND_D', 2))
        self.trend_q = int(hparams.get('TREND_Q', 0))
        self.seasonal_p = int(hparams.get('SEASONAL_P', 5))
        self.seasonal_d = int(hparams.get('SEASONAL_D', 2))
        self.seasonal_q = int(hparams.get('SEASONAL_Q', 0))
        self.m = int(hparams.get('M', 12))
        super(SARIMAModel, self).__init__(model, univariate, name, log_dir=log_dir)


    def fit(self, dataset):
        '''
        Fits a SARIMA forecasting model
        :param dataset: A Pandas DataFrame with 2 columns: Date and Consumption
        '''
        if dataset.shape[1] != 2:
            raise Exception('Univariate models cannot fit with datasets with more than 1 feature.')
        dataset.rename(columns={'Date': 'ds', 'Consumption': 'y'}, inplace=True)
        series = dataset.set_index('ds')
        if self.auto_params:
            best_model = pmdarima.auto_arima(series, seasonal=True, stationary=False, m=self.m, information_criterion='aic',
                                             max_order=2*(self.p + self.q), max_p=2*self.p, max_d=2*self.d,
                                             max_q=2*self.q, max_P=2*self.p, max_D=2*self.d, max_Q=2*self.q,
                                             error_action='ignore')     # Automatically determine model parameters
            order = best_model.order
            seasonal_order = best_model.seasonal_order
            print("Best SARIMA params: (p, d, q):", best_model.order, " and  (P, D, Q, s):", best_model.seasonal_order)
        else:
            order = (self.trend_p, self.trend_d, self.trend_q)
            seasonal_order = (self.seasonal_p, self.seasonal_d, self.seasonal_q, self.m)
        self.model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                                 enforce_stationarity=True, enforce_invertibility=True).fit()
        print(self.model.summary())
        return


    def evaluate(self, train_set, test_set, save_dir=None, plot=False):
        '''
        Evaluates performance of SARIMA model on test set
        :param train_set: A Pandas DataFrame with 2 columns: Date and Consumption
        :param test_set: A Pandas DataFrame with 2 columns: Date and Consumption
        :param save_dir: Directory in which to save forecast metrics
        :param plot: Flag indicating whether to plot the forecast evaluation
        '''
        train_set.rename(columns={'Date': 'ds', 'Consumption': 'y'}, inplace=True)
        test_set.rename(columns={'Date': 'ds', 'Consumption': 'y'}, inplace=True)
        train_set = train_set.set_index('ds')
        test_set = test_set.set_index('ds')
        train_set["model"] = self.model.fittedvalues
        test_set["forecast"] = self.model.predict(start=train_set.shape[0], end=train_set.shape[0] + test_set.shape[0] - 1)

        df_forecast = train_set.append(test_set).rename(columns={'y': 'gt'})
        test_metrics = self.evaluate_forecast(df_forecast, save_dir=save_dir, plot=plot)
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
        predictions = self.model.forecast(steps=days)
        return predictions


    def save_model(self, save_dir):
        '''
        Saves the model to disk
        :param save_dir: Directory in which to save the model
        '''
        if self.model:
            model_path = os.path.join(save_dir, self.name + self.train_date + '.pkl')
            self.model.save(model_path)  # Serialize and save the model object