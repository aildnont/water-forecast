from abc import ABCMeta, abstractmethod
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, GRU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
from src.models.model import ModelStrategy

class NNModel(ModelStrategy):
    __metaclass__ = ABCMeta

    def __init__(self, hparams, name, log_dir):
        self.univariate = hparams.get('UNIVARIATE', False)
        self.batch_size = hparams.get('BATCH_SIZE', 32)
        self.epochs = hparams.get('EPOCHS', 100)
        self.patience = hparams.get('PATIENCE', 15)
        self.val_set_size = hparams.get('VAL_SET_SIZE', 30)
        self.T_x = hparams.get('T_X', 32)
        self.metrics = [MeanSquaredError(name='mse'), RootMeanSquaredError(name='rmse'), MeanAbsoluteError(name='mae'),
                        MeanAbsolutePercentageError(name='mape')]
        self.standard_scaler = StandardScaler()
        model = None
        super(NNModel, self).__init__(model, self.univariate, name, log_dir=log_dir)


    @abstractmethod
    def define_model(self, input_dim):
        '''
        Abstract method for TensorFlow model definition
        '''
        pass


    def fit(self, dataset):
        '''
        Fits an RNN forecasting model
        :param dataset: A Pandas DataFrame with feature columns and a Consumption column
        '''
        df = dataset.copy()
        if self.univariate:
            df = df[['Date', 'Consumption']]
        df.loc[:, dataset.columns != 'Date'] = self.standard_scaler.fit_transform(dataset.loc[:, dataset.columns != 'Date'])

        train_df = df[0:-self.val_set_size]
        val_df = df[self.val_set_size:]

        # Make time series datasets
        train_dates, X_train, Y_train = self.make_windowed_dataset(train_df)
        val_dates, X_val, Y_val = self.make_windowed_dataset(val_df)

        # Define model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.define_model(input_shape)

        # Define model callbacks
        callbacks = [EarlyStopping(monitor='val_loss', verbose=1, patience=self.patience, mode='min', restore_best_weights=True)]
        if self.log_dir is not None:
            callbacks.append(TensorBoard(log_dir=os.path.join(self.log_dir, 'training', self.train_date), histogram_freq=1))

        # Train RNN model
        history = self.model.fit(X_train, Y_train, batch_size=self.batch_size, epochs=self.epochs,
                                 validation_data=(X_val, Y_val), callbacks=callbacks, verbose=1)
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
        test_dates = test_set['Date']

        # Make predictions for training set and obtain forecast for test set
        train_preds = self.model.predict(X_train)
        test_preds = self.forecast(test_dates.shape[0], recent_data=X_train[-1])

        # Rescale data
        train_set.loc[:, train_set.columns != 'Date'] = self.standard_scaler.inverse_transform(train_set.loc[:, train_set.columns != 'Date'])
        test_set.loc[:, test_set.columns != 'Date'] = self.standard_scaler.inverse_transform(test_set.loc[:, test_set.columns != 'Date'])
        train_preds = self.standard_scaler.inverse_transform(train_preds)
        test_preds = self.standard_scaler.inverse_transform(test_preds)

        # Create a DataFrame of combined training set predictions and test set forecast with ground truth
        df_train = pd.DataFrame({'ds': train_dates, 'gt': train_set.iloc[self.T_x:]['Consumption'],
                                 'model': train_preds[:,consumption_idx]})
        df_test = pd.DataFrame({'ds': test_dates, 'gt': test_set['Consumption'],
                                 'forecast': test_preds[:,consumption_idx]})
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
            raise Exception('RNNs require an input of shape (T_x, features) to initiate forecasting.')
        preds = np.zeros((days, recent_data.shape[1]))
        x = recent_data
        for i in range(days):
            preds[i] = self.model.predict(np.expand_dims(x, axis=0))
            x = np.roll(x, 1, axis=0)
            x[-1] = preds[i]    # Prediction becomes latest data point in the example
        return preds


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
        return dates, X, Y



class LSTMModel(NNModel):

    def __init__(self, hparams, log_dir=None):
        name = 'LSTM'
        self.units = hparams.get('UNITS', 128)
        self.lr = hparams.get('LR', 1e-3)
        self.loss = hparams.get('LOSS', 'mse') if hparams.get('LOSS', 'mse') in ['mae', 'mse', 'rmse'] else 'mse'
        super(LSTMModel, self).__init__(hparams, name, log_dir)

    def define_model(self, input_dim):
        X_input = Input(shape=input_dim)
        X = LSTM(self.units, activation='tanh')(X_input)
        Y = Dense(input_dim[1], activation='linear', name='output')(X)
        model = Model(inputs=X_input, outputs=Y, name=self.name)
        optimizer = Adam(learning_rate=self.lr)
        model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)
        print(model.summary())
        return model



class GRUModel(NNModel):

    def __init__(self, hparams, log_dir=None):
        name = 'GRU'
        self.units = hparams.get('UNITS', 128)
        self.lr = hparams.get('LR', 1e-3)
        self.loss = hparams.get('LOSS', 'mse') if hparams.get('LOSS', 'mse') in ['mae', 'mse', 'rmse'] else 'mse'
        super(GRUModel, self).__init__(hparams, name, log_dir)

    def define_model(self, input_dim):
        X_input = Input(shape=input_dim)
        X = GRU(self.units, activation='tanh')(X_input)
        Y = Dense(input_dim.shape[3], activation='linear', name='output')(X)
        model = Model(inputs=X_input, outputs=Y, name=self.name)
        optimizer = Adam(learning_rate=self.lr)
        model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)
        print(model.summary())
        return model


class CNN1DModel(NNModel):

    def __init__(self, hparams, log_dir=None):
        name = '1DCNN'
        self.filters = hparams.get('FILTERS', 128)
        self.fc_units = hparams.get('FC_UNITS', [32])
        self.lr = hparams.get('LR', 1e-3)
        self.loss = hparams.get('LOSS', 'mse') if hparams.get('LOSS', 'mse') in ['mae', 'mse', 'rmse'] else 'mse'
        super(CNN1DModel, self).__init__(hparams, name, log_dir)

    def define_model(self, input_dim):
        X_input = Input(shape=input_dim)
        X = Conv1D(self.filters, activation='relu')(X_input)
        X = Flatten()(X)
        for d in self.fc_units:
            X = Dense(d, activation='relu')(X)
        Y = Dense(input_dim.shape[3], activation='linear', name='output')(X)
        model = Model(inputs=X_input, outputs=Y, name=self.name)
        optimizer = Adam(learning_rate=self.lr)
        model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)
        print(model.summary())
        return model
