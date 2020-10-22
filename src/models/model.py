import pandas as pd
import numpy as np
import datetime
from abc import ABCMeta, abstractmethod
from src.visualization.visualize import plot_model_evaluation

class ModelStrategy(object):
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
    def evaulate(self, train_set, test_set):
        '''
        Abstract method for model evaluation
        '''
        pass


    @abstractmethod
    def forecast(self, days, recent_data=None):
        '''
        Abstract method for forecasting with the model
        '''
        return None


    def cross_validation(self, dataset, n_folds, valid_frac, metrics, file_path=None):
        '''
        Perform a nested cross-validation with day-forward chaining. Results are saved in CSV format.
        :param X: Training data indexed by date
        :param Y: Prediction target, i.e. total daily water consumption
        :param n_folds: Number of folds for cross validation
        :param valid_frac: Fraction of initial dataset to devote to validation
        :param metrics: List of metrics to keep track of
        :return DataFrame of metrics
        '''

        if (valid_frac*(n_folds + 1) > 1):
            raise Exception('Validation set should not be larger than training set. Decrease valid_frac and/or n_folds.')

        metrics_df = pd.DataFrame(np.zeros((n_folds + 2, len(metrics) + 1)), columns=['Fold'] + metrics)
        metrics_df['Fold'] = list(range(1, n_folds + 1)) + ['mean', 'std']

        # Train a model n_folds times with different folds
        for i in range(n_folds):
            self.train_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

            # Separate into training and test sets
            test_end_idx = dataset.shape[0] - i*valid_frac*dataset.shape[0]
            test_start_idx = test_end_idx - valid_frac*dataset.shape[0]
            train_set = dataset[0:test_start_idx]
            test_set = dataset[0:test_start_idx]

            # Train the model and evaluate performance on test set
            self.fit(train_set)
            test_metrics = self.evaulate(test_set)
            for metric in test_metrics:
                if metric in metrics_df.columns:
                    metrics_df[metric][i] = test_metrics[metric]

        # Record mean and standard deviation of test set results
        for metric in metrics:
            metrics_df[metric][n_folds] = metrics_df[metric][0:-2].mean()
            metrics_df[metric][n_folds + 1] = metrics_df[metric][0:-2].std()

        # Save results
        if file_path is not None:
            metrics_df.to_csv(file_path, columns=metrics_df.columns, index_label=False, index=False)
        return metrics_df


    def evaluate_forecast(self, forecast_df, plot=True, save_dir=None):
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
                plot_model_evaluation(forecast_df, self.name, metrics, save_fig=True)

            forecast_df = forecast_df[["gt", "model", "residuals", "conf_int_low", "conf_int_up",
                        "forecast", "error", "pred_int_low", "pred_int_up"]]
            if save_dir is not None:
                metrics_df = pd.DataFrame.from_records([metrics])
                metrics_df.to_csv(save_dir + self.name + '_eval_' + self.train_date + '.csv', sep=',')
            return forecast_df

        except Exception as e:
            print(e)
        return





