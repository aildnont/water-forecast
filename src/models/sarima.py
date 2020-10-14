import pmdarima
import statsmodels.tsa.api as smt
from src.models.model import ModelStrategy

class SARIMAModel(ModelStrategy):

    def __init__(self, hparams):
        univariate = True
        model = None
        name = 'SARIMA'
        self.auto_params = hparams['AUTO_PARAMS']
        self.trend_p = hparams['TREND_P']
        self.trend_d = hparams['TREND_D']
        self.trend_q = hparams['TREND_Q']
        self.seasonal_p = hparams['SEASONAL_P']
        self.seasonal_d = hparams['SEASONAL_D']
        self.seasonal_q = hparams['SEASONAL_Q']
        self.m = hparams['M']
        super(SARIMAModel, self).__init__(model, univariate, name)


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
                                             max_order=self.p_max + self.q_max, max_p=self.p_max, max_d=self.d_max,
                                             max_q=self.q_max, max_P=self.p_max, max_D=self.d_max, max_Q=self.q_max,
                                             error_action='ignore')     # Automatically determine model parameters
            order = best_model.order
            seasonal_order = best_model.seasonal_order
            print("Best SARIMA params: (p, d, q):", best_model.order, " and  (P, D, Q, s):", best_model.seasonal_order)
        else:
            order = (self.trend_p, self.trend_d, self.trend_q)
            seasonal_order = (self.seasonal_p, self.seasonal_d, self.seasonal_q, self.m)
        self.model = smt.SARIMAX(series, order=order, seasonal_order=seasonal_order,
                                 enforce_stationarity=False, enforce_invertibility=False).fit(disp=1)
        return


    def evaluate(self, train_set, test_set, save_dir=None):
        '''
        Evaluates performance of SARIMA model on test set
        :param train_set: A Pandas DataFrame with 2 columns: Date and Consumption
        :param test_set: A Pandas DataFrame with 2 columns: Date and Consumption
        '''
        train_set.rename(columns={'Date': 'ds', 'Consumption': 'y'}, inplace=True)
        test_set.rename(columns={'Date': 'ds', 'Consumption': 'y'}, inplace=True)
        train_set = train_set.set_index('ds')
        test_set = test_set.set_index('ds')
        train_set["model"] = self.model.fittedvalues
        test_set["forecast"] = self.model.predict(start=train_set.shape[0], end=train_set.shape[0] + test_set.shape[0] - 1)

        df_forecast = train_set.append(test_set).rename(columns={'y': 'gt'})
        test_metrics = self.evaluate_forecast(df_forecast, save_dir=save_dir)
        return test_metrics


    def forecast(self, days):
        predictions = self.model.forecast(steps=days)
        return predictions