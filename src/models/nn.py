from abc import ABCMeta, abstractmethod
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, GRU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.keras.models import save_model, load_model
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import pandas as pd
import numpy as np
import os
from src.models.model import ModelStrategy
import datetime

class NNModel(ModelStrategy):
    '''
    A class representing a neural network model defined using TensorFlow and the standard operations on it
    '''
    __metaclass__ = ABCMeta

    def __init__(self, hparams, name, log_dir):
        self.univariate = hparams.get('UNIVARIATE', True)
        self.batch_size = int(hparams.get('BATCH_SIZE', 32))
        self.epochs = int(hparams.get('EPOCHS', 500))
        self.patience = int(hparams.get('PATIENCE', 15))
        self.val_frac = hparams.get('VAL_FRAC', 0.15)
        self.T_x = int(hparams.get('T_X', 32))
        self.metrics = [MeanSquaredError(name='mse'), RootMeanSquaredError(name='rmse'), MeanAbsoluteError(name='mae'),
                        MeanAbsolutePercentageError(name='mape')]
        self.standard_scaler = StandardScaler()
        self.forecast_start = datetime.datetime.today()
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

        train_df = df[0:-int(df.shape[0]*self.val_frac)]
        val_df = df[-int(df.shape[0]*self.val_frac):]

        # Make time series datasets
        train_dates, X_train, Y_train = self.make_windowed_dataset(train_df)
        val_dates, X_val, Y_val = self.make_windowed_dataset(pd.concat([train_df[-self.T_x:], val_df]))

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


    def evaluate(self, train_set, test_set, save_dir=None, plot=False):
        '''
        Evaluates performance of RNN model on test set
        :param train_set: A Pandas DataFrame with feature columns and a Consumption column
        :param test_set: A Pandas DataFrame with feature columns and a Consumption column
        :param save_dir: Directory in which to save forecast metrics
        :param plot: Flag indicating whether to plot the forecast evaluation
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
        test_forecast_dates = test_set['Date']
        prediction_dates = pd.date_range(train_set['Date'].iloc[-1] + pd.DateOffset(1), test_set['Date'].iloc[-1])

        test_forecast_dates_list = test_set['Date'].to_list()
        prediction_dates_list = prediction_dates.to_list()
        forecast_date_idxs = []
        for date in test_forecast_dates_list:
            if date in prediction_dates_list:
                forecast_date_idxs.append(prediction_dates_list.index(date))

        # Make predictions for training set and obtain forecast for test set
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)
        test_forecast_df = self.forecast(test_forecast_dates, prediction_dates, recent_data=X_train[-1])

        # Rescale data
        train_set.loc[:, train_set.columns != 'Date'] = self.standard_scaler.inverse_transform(train_set.loc[:, train_set.columns != 'Date'])
        test_set.loc[:, test_set.columns != 'Date'] = self.standard_scaler.inverse_transform(test_set.loc[:, test_set.columns != 'Date'])
        train_preds = self.standard_scaler.inverse_transform(train_preds)
        test_preds = self.standard_scaler.inverse_transform(test_preds)

        # Create a DataFrame of combined training set predictions and test set forecast with ground truth
        df_train = pd.DataFrame({'ds': train_dates, 'gt': train_set.iloc[self.T_x:]['Consumption'],
                                 'model': train_preds[:,consumption_idx]})
        df_test = pd.DataFrame({'ds': test_forecast_dates.tolist(), 'gt': test_set['Consumption'].tolist(),
                                 'forecast': test_forecast_df['Consumption'].tolist()})  # 'test_pred': test_preds[forecast_date_idxs,consumption_idx].tolist()
        df_forecast = df_train.append(df_test)

        # Compute evaluation metrics for the forecast
        test_metrics = self.evaluate_forecast(df_forecast, save_dir=save_dir, plot=plot)
        return test_metrics


    def forecast(self, test_forecast_dates, prediction_dates, recent_data=None):
        '''
        Create a forecast for the test set. Note that this is different than obtaining predictions for the test set.
        The model makes a prediction for the provided example, then uses the result for the next prediction.
        Repeat this process for a specified number of days.
        :param test_forecast_dates: Future dates to produce a forecast for
        :prediction_dates: Future dates to predict
        :param recent_data: A factual example for the first prediction
        :return: An array of predictions
        '''
        if recent_data is None:
            raise Exception('RNNs require an input of shape (T_x, features) to initiate forecasting.')
        preds = np.zeros((test_forecast_dates.shape[0], recent_data.shape[1]))
        x = recent_data
        idx = 0
        forecast_dates = test_forecast_dates.tolist()
        for date in prediction_dates:
            pred = self.model.predict(np.expand_dims(x, axis=0))
            if date in forecast_dates:
                preds[idx] = self.model.predict(np.expand_dims(x, axis=0))
                idx += 1
            x = np.roll(x, -1, axis=0)
            x[-1] = pred    # Prediction becomes latest data point in the example
        preds = self.standard_scaler.inverse_transform(preds)
        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Consumption': preds[:,0].tolist()})
        return forecast_df


    def save(self, save_dir, scaler_dir=None):
        '''
        Saves the model to disk
        :param save_dir: Directory in which to save the model
        '''
        if self.model:
            model_path = os.path.join(save_dir, self.name + self.train_date + '.h5')
            save_model(self.model, model_path)  # Save the model's weights
            dump(self.standard_scaler, scaler_dir + 'standard_scaler.joblib')


    def load(self, model_path, scaler_path=None):
        '''
        Loads the model from disk
        :param model_path: Path to saved model
        '''
        if os.path.splitext(model_path)[1] != '.h5':
            raise Exception('Model file path for ' + self.name + ' must have ".h5" extension.')
        if scaler_path is None:
            raise Exception('Missing a path to a serialized standard scaler.')
        self.model = load_model(model_path, compile=False)
        self.standard_scaler = load(scaler_path)
        return


    def make_windowed_dataset(self, dataset):
        '''
        Make time series datasets. Each example is a window of the last T_x data points and label is data point 1 day
        into the future.
        :param dataset: Pandas DataFrame indexed by date
        :return: A windowed time series dataset of shape (# rows, T_x, # features)
        '''
        dates = dataset['Date'][self.T_x:].tolist()
        unindexed_dataset = dataset.loc[:, dataset.columns != 'Date']
        X = np.zeros((unindexed_dataset.shape[0] - self.T_x, self.T_x, unindexed_dataset.shape[1]))
        Y = unindexed_dataset[self.T_x:].to_numpy()
        for i in range(X.shape[0]):
            X[i] = unindexed_dataset[i:i+self.T_x].to_numpy()
        return dates, X, Y


    def get_recent_data(self, dataset):
        '''
        Given a preprocessed dataset, get the most recent factual example
        :param dataset: A DataFrame representing a preprocessed dataset
        :return: Most recent factual example
        '''
        if self.univariate:
            dataset = dataset[['Date', 'Consumption']]
        dataset.loc[:, dataset.columns != 'Date'] = self.standard_scaler.transform(dataset.loc[:, dataset.columns != 'Date'])
        test_pred_dates, X, Y = self.make_windowed_dataset(dataset[-self.T_x:])
        self.forecast_start = test_pred_dates[-1]
        return X[-1]


class LSTMModel(NNModel):
    '''
    A class representing a recurrent neural network model with a single LSTM layer
    '''

    def __init__(self, hparams, log_dir=None):
        name = 'LSTM'
        self.units = int(hparams.get('UNITS', 128))
        self.dropout = hparams.get('DROPOUT', 0.25)
        self.fc0_units = int(hparams.get('FC0_UNITS', 32))
        self.fc1_units = int(hparams.get('FC1_UNITS', None))
        self.lr = hparams.get('LR', 1e-3)
        self.loss = hparams.get('LOSS', 'mse') if hparams.get('LOSS', 'mse') in ['mae', 'mse', 'rmse'] else 'mse'
        super(LSTMModel, self).__init__(hparams, name, log_dir)

    def define_model(self, input_dim):
        X_input = Input(shape=input_dim)
        X = LSTM(self.units, activation='tanh', return_sequences=True, name='lstm')(X_input)
        X = Flatten()(X)
        if self.fc0_units is not None:
            X = Dense(self.fc0_units, activation='relu', name='fc0')(X)
            X = Dropout(self.dropout)(X)
            if self.fc1_units is not None:
                X = Dense(self.fc1_units, activation='relu', name='fc1')(X)
                X = Dropout(self.dropout)(X)
        Y = Dense(input_dim[1], activation='linear', name='output')(X)
        model = Model(inputs=X_input, outputs=Y, name=self.name)
        optimizer = Adam(learning_rate=self.lr)
        model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)
        print(model.summary())
        return model



class GRUModel(NNModel):
    '''
    A class representing a recurrent neural network model with a single GRU layer
    '''

    def __init__(self, hparams, log_dir=None):
        name = 'GRU'
        self.units = int(hparams.get('UNITS', 128))
        self.dropout = hparams.get('DROPOUT', 0.25)
        self.fc0_units = int(hparams.get('FC0_UNITS', [32]))
        self.fc1_units = int(hparams.get('FC1_UNITS', None))
        self.lr = hparams.get('LR', 1e-3)
        self.loss = hparams.get('LOSS', 'mse') if hparams.get('LOSS', 'mse') in ['mae', 'mse', 'rmse'] else 'mse'
        super(GRUModel, self).__init__(hparams, name, log_dir)

    def define_model(self, input_dim):
        X_input = Input(shape=input_dim)
        X = GRU(self.units, activation='tanh', return_sequences=True, name='gru')(X_input)
        X = Flatten()(X)
        if self.fc0_units is not None:
            X = Dropout(self.dropout)(X)
            X = Dense(self.fc0_units, activation='relu', name='fc0')(X)
            if self.fc1_units is not None:
                X = Dropout(self.dropout)(X)
                X = Dense(self.fc1_units, activation='relu', name='fc1')(X)
        Y = Dense(input_dim[1], activation='linear', name='output')(X)
        model = Model(inputs=X_input, outputs=Y, name=self.name)
        optimizer = Adam(learning_rate=self.lr)
        model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)
        print(model.summary())
        return model


class CNN1DModel(NNModel):
    '''
    A class representing a 1D convolutional neural network model with a single 1D convolutional layer
    '''

    def __init__(self, hparams, log_dir=None):
        name = '1DCNN'
        self.init_filters = int(hparams.get('FILTERS', 128))
        self.filter_multiplier = int(hparams.get('FILTER_MULTIPLIER', 2))
        self.kernel_size = int(hparams.get('KERNEL_SIZE', 3))
        self.stride = int(hparams.get('STRIDE', 2))
        self.n_conv_layers = int(hparams.get('N_CONV_LAYERS', 2))
        self.fc0_units = int(hparams.get('FC0_UNITS', 32))
        self.fc1_units = int(hparams.get('FC1_UNITS', 16))
        self.dropout = hparams.get('DROPOUT', 0.25)
        self.lr = hparams.get('LR', 1e-3)
        self.loss = hparams.get('LOSS', 'mse') if hparams.get('LOSS', 'mse') in ['mae', 'mse', 'rmse'] else 'mse'
        super(CNN1DModel, self).__init__(hparams, name, log_dir)

    def define_model(self, input_dim):
        X_input = Input(shape=input_dim, name='input')
        X = X_input
        for i in range(self.n_conv_layers):
            try:
                X = Conv1D(self.init_filters * self.filter_multiplier**i, self.kernel_size, strides=self.stride,
                           activation='relu', name='conv' + str(i))(X)
            except Exception as e:
                print("Model cannot be defined with above hyperparameters", e)
        X = Flatten()(X)
        if self.fc0_units is not None:
            X = Dropout(self.dropout)(X)
            X = Dense(self.fc0_units, activation='relu', name='fc0')(X)
            if self.fc1_units is not None:
                X = Dropout(self.dropout)(X)
                X = Dense(self.fc1_units, activation='relu', name='fc1')(X)
        Y = Dense(input_dim[1], activation='linear', name='output')(X)
        model = Model(inputs=X_input, outputs=Y, name=self.name)
        optimizer = Adam(learning_rate=self.lr)
        model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)
        print(model.summary())
        return model
