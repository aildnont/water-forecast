import os
import argparse
import yaml
import shutil
import datetime
from azureml.core import Run
from src.train import *

parser = argparse.ArgumentParser()
parser.add_argument('--trainoutputdir', type=str, help="directory for pipeline outputs")
parser.add_argument('--preprocessedoutputdir', type=str, help="intermediate pipeline directory containing preprocessed data")
args = parser.parse_args()
cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
run = Run.get_context()

# All outputs from this run will be in the same directory
destination_dir = args.trainoutputdir + cur_date + '/'
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Get paths of data from Azure blob storage. Create new folders as necessary
cfg = yaml.full_load(open(os.getcwd() + "./config.yml", 'r'))
cfg['PATHS']['PREPROCESSED_DATA'] = args.preprocessedoutputdir + '/' + cfg['PATHS']['PREPROCESSED_DATA'].split('/')[-1]
cfg['PATHS']['MODELS'] = destination_dir + '/' + cfg['PATHS']['MODELS'].split('/')[-1]
if not os.path.exists(cfg['PATHS']['MODELS']):
    os.makedirs(cfg['PATHS']['MODELS'])
cfg['PATHS']['SERIALIZATIONS'] = destination_dir + '/' + cfg['PATHS']['SERIALIZATIONS'].split('/')[-1]
if not os.path.exists(cfg['PATHS']['SERIALIZATIONS']):
    os.makedirs(cfg['PATHS']['SERIALIZATIONS'])
cfg['PATHS']['EXPERIMENTS'] = destination_dir + '/' + cfg['PATHS']['EXPERIMENTS'].split('/')[-1]
if not os.path.exists(cfg['PATHS']['EXPERIMENTS']):
    os.makedirs(cfg['PATHS']['EXPERIMENTS'])
cfg['PATHS']['FORECAST_VISUALIZATIONS'] = destination_dir + '/' + cfg['PATHS']['FORECAST_VISUALIZATIONS'].split('/')[-1]
if not os.path.exists(cfg['PATHS']['FORECAST_VISUALIZATIONS']):
    os.makedirs(cfg['PATHS']['FORECAST_VISUALIZATIONS'])
cfg['PATHS']['LOGS'] = destination_dir + '/' + cfg['PATHS']['LOGS'].split('/')[-1]
if not os.path.exists(cfg['PATHS']['LOGS']):
    os.makedirs(cfg['PATHS']['LOGS'])

# Keep a copy of preprocessed data in persistent blob storage for this run
shutil.move(cfg['PATHS']['PREPROCESSED_DATA'], destination_dir)

# Train a model, using a fixed size test set and save the metrics, along with test set forecast visualization
cfg['DATA']['TEST_DAYS'] = 183      # Test set is half a year
test_forecast_metrics = train_single(cfg, save_model=True, save_metrics=True, fixed_test_set=True)

# Record test forecast metrics
for metric in test_forecast_metrics:
    run.log('forecast_' + metric, test_forecast_metrics[metric])

# Train a model using all available data
cfg['DATA']['TEST_DAYS'] = 0
train_single(cfg, save_model=True, save_metrics=False, fixed_test_set=True)

