import pmdarima
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
import os
from src.models.model import ModelStrategy

class ARIMAModel(ModelStrategy):
    '''
    A class for an Autoregressive Integrated Moving Average Model and the standard operations on it
    '''

    def __init__(self, hparams, log_dir=None):
        univariate = True
        model = None
        name = 'ARIMA'
        self.auto_params = hparams['AUTO_PARAMS']
        self.p = hparams.get('P', 30)
        self.d = hparams.get('D', 0)
        self.q = hparams.get('Q', 0)
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
            best_model = pmdarima.auto_arima(series, seasonal=False, stationary=False, information_criterion='aic',
                                             max_order=2*(self.p + self.q), max_p=2*self.p, max_d=2*self.d,
                                             max_q=2*self.q, error_action='ignore')
            order = best_model.order
            print("Best ARIMA params: (p, d, q):", best_model.order)
        else:
            order = (self.p, self.d, self.q)
        self.model = ARIMA(series, order=order).fit()
        print(self.model.summary())
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
        test_set["forecast"] = self.forecast(test_set.shape[0])['Consumption'].tolist()

        df_forecast = train_set.append(test_set).rename(columns={'y': 'gt'})
        test_metrics = self.evaluate_forecast(df_forecast, save_dir=save_dir)
        return test_metrics


    def forecast(self, days, recent_data=None):
        forecast_df = self.model.forecast(steps=days).reset_index(level=0)
        forecast_df.columns = ['Date', 'Consumption']
        return forecast_df


    def save(self, save_dir, scaler_dir=None):
        '''
        Saves the model to disk
        :param save_dir: Directory in which to save the model
        '''
        if self.model:
            model_path = os.path.join(save_dir, self.name + self.train_date + '.pkl')
            self.model.save(model_path)  # Serialize and save the model object


    def load(self, model_path, scaler_path=None):
        '''
        Loads the model from disk
        :param model_path: Path to saved model
        '''
        if os.path.splitext(model_path)[1] != '.pkl':
            raise Exception('Model file path for ' + self.name + ' must have ".pkl" extension.')
        self.model = ARIMAResults.load(model_path)
        return