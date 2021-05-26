import os
import argparse
import yaml
import shutil
import datetime
from distutils.dir_util import copy_tree
from azureml.core import Run
from src.train import *
from src.predict import *

parser = argparse.ArgumentParser()
parser.add_argument('--preprocessedintermediatedir', type=str, help="Directory containing preprocessed data for each rate class")
parser.add_argument('--trainoutputdir', type=str, help="directory for pipeline outputs")
parser.add_argument('--preprocessedoutputdir', type=str, help="Directory to save preprocessed data in once training is complete")
TEST_DAYS = int(os.getenv("AML_PARAMETER_TEST_DAYS"))            # Number of days in test set, for model evaluation
FORECAST_DAYS = int(os.getenv("AML_PARAMETER_FORECAST_DAYS"))    # Number of days to produce future forecast for
args = parser.parse_args()
run = Run.get_context()

# All outputs from this run will be in the same directory
DATED_OUTPUTS_DIR = args.trainoutputdir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/' # All outputs from this run go in a new folder
DESTINATION_DIR = args.trainoutputdir + 'latest/'                                                 # Folder to save latest outputs to
if not os.path.exists(DESTINATION_DIR):
    os.makedirs(DESTINATION_DIR)
if not os.path.exists(DATED_OUTPUTS_DIR):
    os.makedirs(DATED_OUTPUTS_DIR)

# Keep track of updated preprocessed data
PREPROCESSED_OUTPUT_DIR = args.preprocessedoutputdir
if not os.path.exists(PREPROCESSED_OUTPUT_DIR):
    os.makedirs(PREPROCESSED_OUTPUT_DIR)

# Train a model for the whole system and for each rate class
RATE_CLASSES = ['all', 'RESI', 'COMM', 'IND', 'INS']
for rate_class in RATE_CLASSES:
    print("TRAINING MODEL FOR " + rate_class + ":\n****************************")

    rc_destination_dir = DESTINATION_DIR + rate_class.lower() + '/'
    if not os.path.exists(rc_destination_dir):
        os.makedirs(rc_destination_dir)
    rc_destination_eval_model_dir = rc_destination_dir + 'eval_model/'
    if not os.path.exists(rc_destination_eval_model_dir):
        os.makedirs(rc_destination_eval_model_dir)
    rc_destination_final_model_dir = rc_destination_dir + 'final_model/'
    if not os.path.exists(rc_destination_final_model_dir):
        os.makedirs(rc_destination_final_model_dir)
    rc_preprocessed_dir = PREPROCESSED_OUTPUT_DIR + rate_class.lower() + '/'
    if not os.path.exists(rc_preprocessed_dir):
        os.makedirs(rc_preprocessed_dir)

    # Get relevant paths on Azure blob storage. Create new folders as necessary.
    cfg = yaml.full_load(open("./config.yml", 'r'))
    cfg['PATHS']['PREPROCESSED_DATA'] = args.preprocessedintermediatedir + '/' + rate_class.lower() + '/' + cfg['PATHS']['PREPROCESSED_DATA'].split('/')[-1]
    cfg['PATHS']['MODELS'] = rc_destination_eval_model_dir
    cfg['PATHS']['SERIALIZATIONS'] = rc_destination_eval_model_dir
    cfg['PATHS']['EXPERIMENTS'] = rc_destination_eval_model_dir
    cfg['PATHS']['FORECAST_VISUALIZATIONS'] = rc_destination_eval_model_dir
    cfg['PATHS']['LOGS'] = rc_destination_eval_model_dir
    cfg['PATHS']['PREDICTIONS'] = rc_destination_eval_model_dir
    cfg['PATHS']['INTERPRETABILITY'] = rc_destination_eval_model_dir
    cfg['PATHS']['INTERPRETABILITY_VISUALIZATIONS'] = rc_destination_eval_model_dir

    # Train a model, using a fixed size test set and save the metrics, along with test set forecast visualization
    cfg['DATA']['TEST_DAYS'] = TEST_DAYS      # Test set is half a year
    cfg['TRAIN']['INTERPRETABILITY'] = True   
    test_forecast_metrics, _ = train_single(cfg, save_model=True, save_metrics=True, fixed_test_set=True, dated_paths=False)

    # Record test forecast metrics
    for metric in test_forecast_metrics:
        run.log(rate_class + '_test_' + metric, test_forecast_metrics[metric])
        
    # Update paths to final model directory
    cfg['PATHS']['MODELS'] = rc_destination_final_model_dir
    cfg['PATHS']['SERIALIZATIONS'] = rc_destination_final_model_dir
    cfg['PATHS']['EXPERIMENTS'] = rc_destination_final_model_dir
    cfg['PATHS']['FORECAST_VISUALIZATIONS'] = rc_destination_final_model_dir
    cfg['PATHS']['LOGS'] = rc_destination_final_model_dir
    cfg['PATHS']['PREDICTIONS'] = rc_destination_final_model_dir
    cfg['PATHS']['INTERPRETABILITY'] = rc_destination_final_model_dir
    cfg['PATHS']['INTERPRETABILITY_VISUALIZATIONS'] = rc_destination_final_model_dir

    # Train a model using all available data
    cfg['DATA']['TEST_DAYS'] = 0
    cfg['TRAIN']['INTERPRETABILITY'] = True   # Ensure model components are saved
    _, model = train_single(cfg, save_model=True, save_metrics=False, fixed_test_set=True, dated_paths=False)

    # Produce a water consumption forecast and save it
    forecast(FORECAST_DAYS, cfg=cfg, model=model, save=True)
    
    # Keep a copy of preprocessed data in persistent blob storage for this run and update the historical preprocessed dataset for this rate class
    shutil.copy(cfg['PATHS']['PREPROCESSED_DATA'], rc_preprocessed_dir)
    shutil.move(cfg['PATHS']['PREPROCESSED_DATA'], rc_destination_dir)
    
    # Copy all outputs to a dated folder for logging purposes
    copy_tree(DESTINATION_DIR, DATED_OUTPUTS_DIR)

