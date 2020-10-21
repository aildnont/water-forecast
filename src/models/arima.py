import pmdarima
from statsmodels.tsa.arima_model import ARIMA
from src.models.model import ModelStrategy

class ARIMAModel(ModelStrategy):

    def __init__(self, hparams, log_dir=None):
        univariate = True
        model = None
        name = 'ARIMA'
        self.auto_params = hparams['AUTO_PARAMS']
        self.p = hparams['P']
        self.d = hparams['D']
        self.q = hparams['Q']
        super(ARIMAModel, self).__init__(model, univariate, name, log_dir=log_dir)


    def fit(self, dataset):
        '''
        Fits an ARIMA forecasting model
        :param dataset: A Pandas DataFrame with 2 columns: Date and Consumption
        '''
        if dataset.shape[1] != 2:
            raise Exception('Univariate models cannot fit with datasets with more than 1 feature.')
        dataset.rename(columns={'Date': 'ds', 'Consumption': 'y'}, inplace=True)
        series = dataset.set_index('ds')
        if self.auto_params:
            best_model = pmdarima.auto_arima(series, seasonal=False, stationary=False, m=self.m, information_criterion='aic',
                                             max_order=self.p + self.q, max_p=self.p, max_d=self.d,
                                             max_q=self.q, error_action='ignore')     # Automatically determine model parameters
            order = best_model.order
            print("Best ARIMA params: (p, d, q):", best_model.order)
        else:
            order = (self.p, self.d, self.q)
        self.model = ARIMA(series, order=order).fit(disp=1)
        return


    def evaluate(self, train_set, test_set, save_dir=None):
        '''
        Evaluates performance of ARIMA model on test set
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


    def forecast(self, days, recent_data=None):
        predictions = self.model.forecast(steps=days)
        return predictions