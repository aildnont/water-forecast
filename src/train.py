import pandas as pd
import yaml
import os
import datetime
from sklearn.model_selection import TimeSeriesSplit
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from src.models.prophet import ProphetModel
from src.models.arima import ARIMAModel
from src.models.sarimax import SARIMAXModel
from src.models.nn import *
from src.models.skmodels import *
from src.data.preprocess import preprocess_ts
from src.visualization.visualize import plot_bayesian_hparam_opt

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


def load_dataset(cfg, fixed_test_set=False):
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
    if fixed_test_set:
        if cfg['DATA']['TEST_DAYS'] <= 0:
            train_df = df[:int(df.shape[0])]
            test_df = df[int(df.shape[0]):]
        else:
            train_df = df[:int(-cfg['DATA']['TEST_DAYS'])]
            test_df = df[int(-cfg['DATA']['TEST_DAYS']):]
    else:
        train_df = df[:int((1 - cfg['DATA']['TEST_FRAC']) * df.shape[0])]
        test_df = df[int((1 - cfg['DATA']['TEST_FRAC']) * df.shape[0]):]
    print('Size of training set: ', train_df.shape[0])
    print('Size of test set: ', test_df.shape[0])
    return train_df, test_df


def train_model(cfg, model_def, hparams, train_df, test_df, save_model=False, write_logs=False, save_metrics=False):
    '''
    Train a model
    :param cfg: Project config
    :param model_def: Class definition of model to train
    :param hparams: A dict of hyperparameters specific to this model
    :param train_df: Training set as DataFrame
    :param test_df: Test set as DataFrame
    :param save_model: Flag indicating whether to save the model
    :param write_logs: Flag indicating whether to write any training logs to disk
    :param save_metrics: Flag indicating whether to save the forecast metrics to a CSV
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
        model.save(cfg['PATHS']['MODELS'], scaler_dir=cfg['PATHS']['SERIALIZATIONS'])

    # Evaluate the model on the test set, if it exists
    if test_df.shape[0] > 0:
        save_dir = cfg['PATHS']['EXPERIMENTS'] if save_metrics else None
        test_forecast_metrics = model.evaluate(train_df, test_df, save_dir=save_dir, plot=save_metrics)
        if cfg['TRAIN']['INTERPRETABILITY'] and model.name == 'Prophet':
            model.decompose(cfg['PATHS']['INTERPRETABILITY'])
    else:
        test_forecast_metrics = {}
    return test_forecast_metrics



def train_single(cfg, hparams=None, save_model=False, write_logs=False, save_metrics=False, fixed_test_set=False):
    '''
    Train a single model. Use the passed hyperparameters if possible; otherwise, use those in config.
    :param cfg: Project config
    :param hparams: Dict of hyperparameters
    :param save_model: Flag indicating whether to save the model
    :param write_logs: Flag indicating whether to write any training logs to disk
    :param save_metrics: Flag indicating whether to save the forecast metrics to a CSV
    :param fixed_test_set: Flag indicating whether to use a fixed number of days for test set
    :return: Dictionary of test set forecast metrics
    '''
    train_df, test_df = load_dataset(cfg, fixed_test_set=fixed_test_set)
    model_def = MODELS_DEFS.get(cfg['TRAIN']['MODEL'].upper(), lambda: "Invalid model specified in cfg['TRAIN']['MODEL']")
    if hparams is None:
        hparams = cfg['HPARAMS'][cfg['TRAIN']['MODEL'].upper()]
    test_forecast_metrics, model = train_model(cfg, model_def, hparams, train_df, test_df, save_model=save_model,
                                        write_logs=write_logs, save_metrics=save_metrics)
    print('Test forecast metrics: ', test_forecast_metrics)
    return test_forecast_metrics, model


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
        test_forecast_metrics, _ = train_model(cfg, model_def, hparams, train_df, test_df, save_model=save_models,
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


def cross_validation(cfg, dataset=None, metrics=None, model_name=None, hparams=None, last_folds=None, save_results=False):
    '''
    Perform a nested cross-validation with day-forward chaining. Results are saved in CSV format.
    :param cfg: project config
    :param dataset: A DataFrame consisting of the entire dataset
    :param metrics: list of metrics to report
    :param model_name: String identifying model
    :param last_folds: Limit cross validation to the most recent last_folds folds
    :param save_results: Flag indicating whether to save results
    :return DataFrame of metrics
    '''

    n_folds = cfg['TRAIN']['N_FOLDS']
    if dataset is None:
        dataset = pd.read_csv(cfg['PATHS']['PREPROCESSED_DATA'])
        dataset['Date'] = pd.to_datetime(dataset['Date'])
        dataset = dataset[50:-50]  # TODO: Update this!
    if last_folds is None:
        last_folds = n_folds
    if metrics is None:
        metrics = ['residuals_mean', 'residuals_std', 'error_mean', 'error_std', 'MAE', 'MAPE', 'MSE', 'RMSE']
    n_rows = n_folds if last_folds is None else last_folds
    metrics_df = pd.DataFrame(np.zeros((n_rows + 2, len(metrics) + 1)), columns=['Fold'] + metrics)
    metrics_df['Fold'] = list(range(n_folds - last_folds + 1, n_folds + 1)) + ['mean', 'std']
    ts_cv = TimeSeriesSplit(n_splits=n_folds)
    model_name = cfg['TRAIN']['MODEL'].upper() if model_name is None else model_name
    hparams = cfg['HPARAMS'][model_name] if hparams is None else hparams

    model_def = MODELS_DEFS.get(model_name, lambda: "Invalid model specified in cfg['TRAIN']['MODEL']")

    # Train a model n_folds times with different folds
    cur_fold = 0
    row_idx = 0
    for train_index, test_index in ts_cv.split(dataset):
        if cur_fold >= n_folds - last_folds:
            print('Fitting model for fold ' + str(cur_fold))
            model = model_def(hparams)
            model.train_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

            # Separate into training and test sets
            train_df, test_df = dataset.iloc[train_index], dataset.iloc[test_index]
            if model.univariate:
                train_df = train_df[['Date', 'Consumption']]
                test_df = test_df[['Date', 'Consumption']]

            if model_name in ['LSTM', 'GRU', '1DCNN']:
                if model.val_frac*train_df.shape[0] < model.T_x:
                    continue    # Validation set can't be larger than input sequence length

            # Train the model and evaluate performance on test set
            model.fit(train_df)
            test_metrics = model.evaluate(train_df, test_df, save_dir=None, plot=False)
            for metric in test_metrics:
                if metric in metrics_df.columns:
                    metrics_df[metric][row_idx] = test_metrics[metric]
            row_idx += 1
        cur_fold += 1

    # Record mean and standard deviation of test set results
    for metric in metrics:
        metrics_df[metric][last_folds] = metrics_df[metric][0:-2].mean()
        metrics_df[metric][last_folds + 1] = metrics_df[metric][0:-2].std()

    # Save results
    if save_results:
        file_path = cfg['PATHS']['EXPERIMENTS'] + 'cross_val_' + model_name + \
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
        metrics_df.to_csv(file_path, columns=metrics_df.columns, index_label=False, index=False)
    return metrics_df


def bayesian_hparam_optimization(cfg):
    '''
    Conducts a Bayesian hyperparameter optimization, given the parameter ranges and selected model
    :param cfg: Project config
    :return: Dict of hyperparameters deemed optimal
    '''

    dataset = pd.read_csv(cfg['PATHS']['PREPROCESSED_DATA'])
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset = dataset[50:-50]  # TODO: Update this!

    model_name = cfg['TRAIN']['MODEL'].upper()
    objective_metric = cfg['TRAIN']['HPARAM_SEARCH']['HPARAM_OBJECTIVE']
    results = {'Trial': [], objective_metric: []}
    dimensions = []
    default_params = []
    hparam_names = []
    for hparam_name in cfg['HPARAM_SEARCH'][model_name]:
        if cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'] is not None:
            if cfg['HPARAM_SEARCH'][model_name][hparam_name]['TYPE'] == 'set':
                dimensions.append(Categorical(categories=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'],
                                              name=hparam_name))
            elif cfg['HPARAM_SEARCH'][model_name][hparam_name]['TYPE'] == 'int_uniform':
                dimensions.append(Integer(low=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][0],
                                          high=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][1],
                                          prior='uniform', name=hparam_name))
            elif cfg['HPARAM_SEARCH'][model_name][hparam_name]['TYPE'] == 'float_log':
                dimensions.append(Real(low=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][0],
                                       high=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][1],
                                       prior='log-uniform', name=hparam_name))
            elif cfg['HPARAM_SEARCH'][model_name][hparam_name]['TYPE'] == 'float_uniform':
                dimensions.append(Real(low=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][0],
                                       high=cfg['HPARAM_SEARCH'][model_name][hparam_name]['RANGE'][1],
                                       prior='uniform', name=hparam_name))
            default_params.append(cfg['HPARAMS'][model_name][hparam_name])
            hparam_names.append(hparam_name)
            results[hparam_name] = []

    def objective(vals):
        hparams = dict(zip(hparam_names, vals))
        print('HPARAM VALUES: ', hparams)
        #scores = cross_validation(cfg, dataset=dataset, metrics=[objective_metric], model_name=model_name, hparams=hparams,
        #                          last_folds=cfg['TRAIN']['HPARAM_SEARCH']['LAST_FOLDS'])[objective_metric]
        #score = scores[scores.shape[0] - 2]     # Get the mean value for the error metric from the cross validation
        test_metrics, _ = train_single(cfg, hparams=hparams, save_model=False, write_logs=False, save_metrics=False)
        score = test_metrics['MAPE']
        return score   # We aim to minimize error
    search_results = gp_minimize(func=objective, dimensions=dimensions, acq_func='EI',
                                 n_calls=cfg['TRAIN']['HPARAM_SEARCH']['N_EVALS'], verbose=True)
    print(search_results)
    plot_bayesian_hparam_opt(model_name, hparam_names, search_results, save_fig=True)

    # Create table to detail results
    trial_idx = 0
    for t in search_results.x_iters:
        results['Trial'].append(str(trial_idx))
        results[objective_metric].append(search_results.func_vals[trial_idx])
        for i in range(len(hparam_names)):
            results[hparam_names[i]].append(t[i])
        trial_idx += 1
    results['Trial'].append('Best')
    results[objective_metric].append(search_results.fun)
    for i in range(len(hparam_names)):
        results[hparam_names[i]].append(search_results.x[i])
    results_df = pd.DataFrame(results)
    results_path = cfg['PATHS']['EXPERIMENTS'] + 'hparam_search_' + model_name + \
                   datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
    results_df.to_csv(results_path, index_label=False, index=False)


    return search_results


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
        train_single(cfg, save_model=save_model, save_metrics=True)
    elif experiment == 'train_all':
        train_all(cfg, save_models=save_model)
    elif experiment == 'hparam_search':
        bayesian_hparam_optimization(cfg)
    elif experiment == 'cross_validation':
        cross_validation(cfg, save_results=True)
    else:
        raise Exception("Invalid entry in TRAIN > EXPERIMENT field of config.yml.")
    return


if __name__ == '__main__':
    cfg = yaml.full_load(open("./config.yml", 'r'))
    train_experiment(cfg=cfg, experiment=cfg['TRAIN']['EXPERIMENT'], save_model=True, write_logs=True)
