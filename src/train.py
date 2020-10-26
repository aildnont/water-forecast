import pandas as pd
import yaml
import os
import datetime
from src.models.prophet import ProphetModel
from src.models.arima import ARIMAModel
from src.models.sarima import SARIMAModel
from src.models.nn import *
from src.models.skmodels import *
from src.data.preprocess import preprocess_ts

# Map model names to their respective class definitions
MODELS_DEFS = {
    'PROPHET': ProphetModel,
    'ARIMA': ARIMAModel,
    'SARIMA': SARIMAModel,
    'LSTM': LSTMModel,
    'GRU': GRUModel,
    '1DCNN': CNN1DModel,
    'LINEAR_REGRESSION': LinearRegressionModel,
    'RANDOM_FOREST': RandomForestModel
}


def load_dataset(cfg):
    '''
    Load preprocessed dataset and return training and test sets.
    :param cfg: Project config
    :return: DataFrames for training and test sets
    '''
    try:
        df = pd.read_csv(cfg['PATHS']['PREPROCESSED_DATA'])
    except FileNotFoundError:
        print("No file found at " + cfg['PATHS']['PREPROCESSED_DATA'] + ". Running preprocessing of client data.")
        df = preprocess_ts(cfg, save_df=False)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[50:-50]  # For now, take off dates at start and end due to incomplete data at boundaries

    # Define training and test sets
    train_df = df[:int((1 - cfg['DATA']['TEST_FRAC']) * df.shape[0])]
    test_df = df[int((1 - cfg['DATA']['TEST_FRAC']) * df.shape[0]):]
    return train_df, test_df


def train_model(cfg, model_def, hparams, train_df, test_df, save_model=False, write_logs=False):
    '''
    Train a model
    :param cfg: Project config
    :param model_def: Class definition of model to train
    :param hparams: A dict of hyperparameters specific to this model
    :param train_df: Training set as DataFrame
    :param test_df: Test set as DataFrame
    :param save_model: Flag indicating whether to save the model
    :param write_logs: Flag indicating whether to write any training logs to disk
    :return: Dictionary of test set forecast metrics
    '''
    log_dir = cfg['PATHS']['LOGS'] if write_logs else None
    model = model_def(hparams, log_dir=log_dir)  # Create instance of model

    # Fit the model
    if model.univariate:
        train_df = train_df[['Date', 'Consumption']]
        test_df = test_df[['Date', 'Consumption']]
    model.fit(train_df)
    if save_model:
        model.save_model(cfg['PATHS']['MODELS'])

    # Evaluate the model on the test set
    test_forecast_metrics = model.evaluate(train_df, test_df, save_dir=cfg['PATHS']['EXPERIMENTS'])
    return test_forecast_metrics



def train_single(cfg, save_model=False, write_logs=False):
    '''
    Train a single model
    :param cfg: Project config
    :param save_model: Flag indicating whether to save the model
    :param write_logs: Flag indicating whether to write any training logs to disk
    :return: Dictionary of test set forecast metrics
    '''
    train_df, test_df = load_dataset(cfg)
    model_def = MODELS_DEFS.get(cfg['TRAIN']['MODEL'].upper(), lambda: "Invalid model specified in cfg['TRAIN']['MODEL']")
    hparams = cfg['HPARAMS'][cfg['TRAIN']['MODEL'].upper()]
    test_forecast_metrics = train_model(cfg, model_def, hparams, train_df, test_df, save_model=save_model,
                                        write_logs=write_logs)
    print('Test forecast metrics: ', test_forecast_metrics)
    return


def train_all(cfg, save_models=False, write_logs=False):
    '''
    Train all models that have available definitions in this project
    :param cfg: Project config
    :param save_models: Flag indicating whether to save the trained models
    :param write_logs: Flag indicating whether to write any training logs to disk
    :return: DataFrame of test set forecast metrics for all models
    '''
    train_df, test_df = load_dataset(cfg)
    all_model_metrics = {}
    for model_name in MODELS_DEFS:
        print('*** Training ' + model_name + ' ***\n')
        model_def = MODELS_DEFS[model_name]
        hparams = cfg['HPARAMS'][model_name]
        test_forecast_metrics = train_model(cfg, model_def, hparams, train_df, test_df, save_model=save_models,
                                            write_logs=write_logs)
        if all_model_metrics:
            all_model_metrics['model'].append(model_name)
            for metric in test_forecast_metrics:
                all_model_metrics[metric].append(test_forecast_metrics[metric])
        else:
            all_model_metrics['model'] = [model_name]
            for metric in test_forecast_metrics:
                all_model_metrics[metric] = [test_forecast_metrics[metric]]
        print('Test forecast metrics for ' + model_name + ': ', test_forecast_metrics)
    metrics_df = pd.DataFrame(all_model_metrics)
    file_name = 'all_train' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.csv'
    metrics_df.to_csv(os.path.join(cfg['PATHS']['EXPERIMENTS'], file_name), columns=metrics_df.columns,
                      index_label=False, index=False)
    return metrics_df


def train_experiment(cfg=None, experiment='single_train', save_model=False, write_logs=False):
    '''
    Run a training experiment
    :param cfg: Project config
    :param experiment: String defining which experiment to run
    :param save_model: Flag indicating whether to save any models trained during the experiment
    :param write_logs: Flag indicating whether to write logs for training
    '''

    # Load project config data
    if cfg is None:
        cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

    # Conduct the desired train experiment
    if experiment == 'train_single':
        train_single(cfg, save_model=save_model)
    elif experiment == 'train_all':
        train_all(cfg, save_models=save_model)
    return


if __name__ == '__main__':
    cfg = yaml.full_load(open("./config.yml", 'r'))
    train_experiment(cfg=cfg, experiment=cfg['TRAIN']['EXPERIMENT'], save_model=True, write_logs=True)
