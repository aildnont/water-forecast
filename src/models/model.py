import pandas as pd
import numpy as np
import datetime
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import TimeSeriesSplit
from src.visualization.visualize import plot_model_evaluation

class ModelStrategy(object):
    '''
    An abstract base class for defining models. The interface to be implemented by subclasses define standard operations
    on models in data science experiments.
    '''
    __metaclass__ = ABCMeta

    def __init__(self, model, univariate, name, log_dir=None):
        self.model = model
        self.univariate = univariate
        self.name = name
        self.train_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_dir = log_dir

    @abstractmethod
    def fit(self, dataset):
        '''
        Abstract method for model fitting
        '''
        pass


    @abstractmethod
    def evaluate(self, train_set, test_set, save_dir=None, plot=False):
        '''
        Abstract method for model evaluation
        '''
        return None


    @abstractmethod
    def forecast(self, timesteps, recent_data=None, start_date=None):
        '''
        Abstract method for forecasting with the model
        '''
        return None


    @abstractmethod
    def save(self, save_dir, scaler_dir=None):
        '''
        Abstract method for serializing the model
        '''
        return None

    @abstractmethod
    def load(self, model_path, scaler_path=None):
        '''
        Abstract method for restoring the model from persistent storage
        '''
        return


    def evaluate_forecast(self, forecast_df, save_dir=None, plot=False):
        '''
        Given ground truth data and forecasts, assess the model's performance by computing using time series regression
        metrics. Optionally visualize the ground truth, residuals and forecasts.
        :param forecast_df: A DataFrame containing ground truth, predictions on the training set, and predictions from
                            a test set forecast
        :param plot: Flag indicating whether to produce a forecast evaluation plot
        :param save_dir: Directory in which to save a CSV containing forecast metrics
        :return: A dict of forecast metrics
        '''
        try:
            # Residuals
            forecast_df["residuals"] = forecast_df["gt"] - forecast_df["model"]
            forecast_df["error"] = forecast_df["gt"] - forecast_df["forecast"]
            forecast_df["error_pct"] = forecast_df["error"] / forecast_df["gt"]

            # Key metrics
            metrics = {}
            metrics['residuals_mean'] = forecast_df["residuals"].mean()
            metrics['residuals_std'] = forecast_df["residuals"].std()
            metrics['error_mean'] = forecast_df["error"].mean()
            metrics['error_std'] = forecast_df["error"].std()
            metrics['MAE'] = forecast_df["error"].apply(lambda x: np.abs(x)).mean()
            metrics['MAPE'] = forecast_df["error_pct"].apply(lambda x: np.abs(x)).mean()
            metrics['MSE'] = forecast_df["error"].apply(lambda x: x ** 2).mean()
            metrics['RMSE'] = np.sqrt(metrics['MSE'])  # root mean squared error

            # 95% Confidence intervals
            STD_DEVS = 1.96
            forecast_df["conf_int_low"] = forecast_df["forecast"] - STD_DEVS * metrics['residuals_std']
            forecast_df["conf_int_up"] = forecast_df["forecast"] + STD_DEVS * metrics['residuals_std']
            forecast_df["pred_int_low"] = forecast_df["forecast"] - STD_DEVS * metrics['error_std']
            forecast_df["pred_int_up"] = forecast_df["forecast"] + STD_DEVS * metrics['error_std']

            if plot:
                plot_model_evaluation(forecast_df, self.name, metrics, save_dir=save_dir, save_fig=True, train_date=self.train_date)

            forecast_df = forecast_df[["gt", "model", "residuals", "conf_int_low", "conf_int_up",
                        "forecast", "error", "pred_int_low", "pred_int_up"]]
            if save_dir is not None:
                metrics_df = pd.DataFrame.from_records([metrics])
                metrics_df.to_csv(save_dir + self.name + '_eval_' + self.train_date + '.csv', sep=',')
            return metrics

        except Exception as e:
            print(e)
        return





