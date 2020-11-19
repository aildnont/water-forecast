import yaml
import pandas as pd
import datetime
from src.models.prophet import ProphetModel
from src.models.arima import ARIMAModel
from src.models.sarimax import SARIMAXModel
from src.models.nn import *
from src.models.skmodels import *
from src.train import load_dataset

# Map model names to their respective class definitions
MODELS_DEFS = {
    'PROPHET': ProphetModel,
    'ARIMA': ARIMAModel,
    'SARIMAX': SARIMAXModel,
    'LSTM': LSTMModel,
    'GRU': GRUModel,
    '1DCNN': CNN1DModel,
    'LINEAR_REGRESSION': LinearRegressionModel,
    'RANDOM_FOREST': RandomForestModel
}


def forecast(days, cfg=None, save=False):
    '''
    Generate a forecast for a certain number of days
    :param days: Length of forecast
    :param cfg: Project config
    :param save: Flag indicating whether to save the forecast
    :return: DataFrame containing predicted consumption for each future date
    '''

    if cfg is None:
        cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

    model_name = cfg['FORECAST']['MODEL'].upper()
    model_def = MODELS_DEFS.get(model_name, lambda: "Invalid model specified in cfg['FORECAST']['MODEL']")
    hparams = cfg['HPARAMS'][model_name]
    model = model_def(hparams)  # Create instance of model
    model.load(cfg['FORECAST']['MODEL_PATH'], scaler_path=cfg['PATHS']['SERIALIZATIONS'] + 'standard_scaler.joblib')
    if isinstance(model, (NNModel, SKLearnModel)):
        train_df, test_df = load_dataset(cfg)
        recent_data = model.get_recent_data(train_df)
    else:
        recent_data = None
    results = model.forecast(days, recent_data=recent_data)
    if save:
        results.to_csv(cfg['PATHS']['PREDICTIONS'] + 'forecast_' + model_name + '_' + str(days) +
                       datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + 'd_' + '.csv',
                       index=False, index_label=False)
    return results


if __name__ == '__main__':
    cfg = yaml.full_load(open("./config.yml", 'r'))
    days = cfg['FORECAST']['DAYS']
    forecast(days, cfg=cfg, save=True)

