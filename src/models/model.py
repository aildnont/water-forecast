import pandas as pd
import numpy as np
import datetime
from abc import ABCMeta, abstractmethod

class ModelStrategy(object):
    __metaclass__ = ABCMeta


    @property
    def univariate(self):
        '''
        Boolean property representing whether the model is univariate
        '''
        pass


    @property
    def model(self):
        '''
        Model object
        '''
        pass


    @property
    def train_date(self):
        '''
        String describing the date at which the model was fit
        '''
        pass


    @abstractmethod
    def fit(self, dataset):
        '''
        Abstract method for model fitting
        '''
        return None


    @abstractmethod
    def evaulate(self, dataset):
        '''
        Abstract method for model evaluation
        '''
        return None


    @abstractmethod
    def predict(self, X):
        '''
        Abstract method for model evaluation
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
            _ = self.fit(train_set)
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






