from fbprophet import Prophet
from dill import dump, load
import pandas as pd
import os
import json
import datetime
from src.models.model import ModelStrategy
from src.visualization.visualize import plot_prophet_components

class ProphetModel(ModelStrategy):
    '''
    A class representing a Prophet model and standard operations on it
    '''

    def __init__(self, hparams, log_dir=None):
        univariate = True
        name = 'Prophet'
        self.changepoint_prior_scale = hparams.get('CHANGEPOINT_PRIOR_SCALE', 0.05)
        self.seasonality_prior_scale = hparams.get('SEASONALITY_PRIOR_SCALE', 10)
        self.holidays_prior_scale = hparams.get('HOLIDAYS_PRIOR_SCALE', 10)
        self.seasonality_mode = hparams.get('SEASONALITY_MODE', 'additive')
        self.changepoint_range = hparams.get('CHANGEPOINT_RANGE', 0.95)
        self.country = hparams.get('COUNTRY', 'CA')

        # Build DataFrame of local holidays
        if hparams.get('HOLIDAYS', None) is None:
            self.local_holidays = None
        else:
            holiday_dfs = []
            for holiday in hparams.get('HOLIDAYS', []):
                holiday_dfs.append(pd.DataFrame({
                    'holiday': holiday,
                    'ds': pd.to_datetime(hparams['HOLIDAYS'][holiday]),
                    'lower_window': 0,
                    'upper_window': 1}))
            self.local_holidays = pd.concat(holiday_dfs)

        model = Prophet(yearly_seasonality=True, holidays=self.local_holidays, changepoint_prior_scale=self.changepoint_prior_scale,
                        seasonality_prior_scale=self.seasonality_prior_scale, holidays_prior_scale=self.holidays_prior_scale,
                        seasonality_mode=self.seasonality_mode, changepoint_range=self.changepoint_range)
        model.add_country_holidays(country_name=self.country)   # Add country-wide holidays
        super(ProphetModel, self).__init__(model, univariate, name, log_dir=log_dir)


    def fit(self, dataset):
        '''
        Fits a Prophet forecasting model
        :param dataset: A Pandas DataFrame with 2 columns: Date and Consumption
        '''
        if dataset.shape[1] != 2:
            raise Exception('Univariate models cannot fit with datasets with more than 1 feature.')
        dataset.rename(columns={'Date': 'ds', 'Consumption': 'y'}, inplace=True)
        self.model.fit(dataset)
        return


    def evaluate(self, train_set, test_set, save_dir=None, plot=False):
        '''
        Evaluates performance of Prophet model on test set
        :param train_set: A Pandas DataFrame with 2 columns: Date and Consumption
        :param test_set: A Pandas DataFrame with 2 columns: Date and Consumption
        :param save_dir: Directory in which to save forecast metrics
        :param plot: Flag indicating whether to plot the forecast evaluation
        '''
        train_set.rename(columns={'Date': 'ds', 'Consumption': 'y'}, inplace=True)
        test_set.rename(columns={'Date': 'ds', 'Consumption': 'y'}, inplace=True)
        df_prophet = self.model.make_future_dataframe(periods=test_set.shape[0], include_history=True, freq='D')
        self.future_prediction = self.model.predict(df_prophet)
        df_train = train_set.merge(self.future_prediction[["ds", "yhat"]],
                                    how="left").rename(columns={'yhat': 'model', 'y': 'gt'}).set_index("ds")
        df_test = test_set.merge(self.future_prediction[["ds", "yhat"]],
                                  how="left").rename(columns={'yhat': 'forecast', 'y': 'gt'}).set_index("ds")
        df_forecast = df_train.append(df_test)
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
        future_dates = self.model.make_future_dataframe(periods=days)
        self.future_prediction = self.model.predict(future_dates)
        forecast_df = self.future_prediction[['ds', 'yhat']]
        forecast_df.rename(columns={'ds': 'Date', 'yhat': 'Consumption'}, inplace=True)
        return forecast_df


    def save(self, save_dir, scaler_dir=None):
        '''
        Saves the model to disk
        :param save_dir: Directory in which to save the model
        '''
        if self.model:
            model_path = os.path.join(save_dir, self.name + self.train_date + '.pkl')
            dump(self.model, open(model_path, 'wb'))  # Serialize and save the model object


    def load(self, model_path, scaler_path=None):
        '''
        Loads the model from disk
        :param model_path: Path to saved model
        '''
        if os.path.splitext(model_path)[1] != '.pkl':
            raise Exception('Model file path for ' + self.name + ' must have ".pkl" extension.')
        self.model = load(open(model_path, 'rb'))
        return


    def decompose(self, save_dir):
        '''
        Decompose model into its trend, holiday, weekly and yearly components. Generate a plot and save parameters.
        Creates a new directory within save_dir to capture all components of the model in separate files.
        :param save_dir: Directory in which to save the results
        '''

        if not (self.model or self.future_prediction):
            return
        results_dir = save_dir + '/Prophet_components/'
        try:
            os.mkdir(results_dir)
        except OSError:
            print("Creation of directory %s failed" % results_dir)
        self.future_prediction[['ds', 'trend']].to_csv(results_dir + 'trend_component.csv', sep=',', header=True, index=False)
        self.future_prediction[['ds', 'holidays']].to_csv(results_dir + 'holidays_component.csv', sep=',', header=True, index=False)
        with open(results_dir + 'weekly_component.json', 'w') as fp:
            json.dump(self.model.seasonalities['weekly'], fp)
        with open(results_dir + 'yearly_component.json', 'w') as fp:
            json.dump(self.model.seasonalities['yearly'], fp)
        plot_prophet_components(self.model, self.future_prediction, save_dir=save_dir)





