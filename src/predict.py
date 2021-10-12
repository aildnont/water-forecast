import yaml
import pandas as pd
import datetime
from src.models.prophet import ProphetModel
from src.models.arima import ARIMAModel
from src.models.sarimax import SARIMAXModel
from src.models.nn import *
from src.models.skmodels import *
from src.train import load_dataset
from src.visualization.visualize import plot_prophet_forecast

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


def forecast(timesteps, cfg=None, model=None, save=False):
    '''
    Generate a forecast for a certain number of days
    :param timesteps: Length of forecast
    :param cfg: Project config
    :param model: Model object
    :param save: Flag indicating whether to save the forecast
    :return: DataFrame containing predicted consumption for each future date
    '''

    if cfg is None:
        cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

    if model is None:
        model_name = cfg['FORECAST']['MODEL'].upper()
        model_def = MODELS_DEFS.get(model_name, lambda: "Invalid model specified in cfg['FORECAST']['MODEL']")
        hparams = cfg['HPARAMS'][model_name]
        model = model_def(hparams)  # Create instance of model
        model.load(cfg['FORECAST']['MODEL_PATH'], scaler_path=cfg['PATHS']['SERIALIZATIONS'] + 'standard_scaler.joblib')
    else:
        model_name = model.name
    if isinstance(model, (NNModel, SKLearnModel)):
        train_df, test_df = load_dataset(cfg)
        recent_data = model.get_recent_data(train_df)
    else:
        recent_data = None
    results = model.forecast(timesteps, recent_data=recent_data)
    if model.name == 'Prophet':
        plot_prophet_forecast(model.model, model.future_prediction, save_dir=cfg['PATHS']['FORECAST_VISUALIZATIONS'], train_date=model.train_date)
        model.future_prediction.to_csv(cfg['PATHS']['PREDICTIONS'] + 'detailed_forecast_' + model_name + '_' + str(timesteps) +
               model.train_date + '.csv',
               index=False, index_label=False)
    if save:
        results.to_csv(cfg['PATHS']['PREDICTIONS'] + 'forecast_' + model_name + '_' + str(timesteps) +
                       model.train_date + '.csv',
                       index=False, index_label=False)
    return results


if __name__ == '__main__':
    cfg = yaml.full_load(open("./config.yml", 'r'))
    timesteps = cfg['FORECAST']['TIMESTEPS']
    forecast(timesteps, cfg=cfg, save=True)

