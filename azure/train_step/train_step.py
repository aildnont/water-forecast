import os
import argparse
import yaml
import shutil
from azureml.core import Run
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
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
DESTINATION_DIR = args.trainoutputdir
if not os.path.exists(DESTINATION_DIR):
    os.makedirs(DESTINATION_DIR)

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
    test_forecast_metrics, _ = train_single(cfg, save_model=True, save_metrics=True, fixed_test_set=True)

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
    _, model = train_single(cfg, save_model=True, save_metrics=False, fixed_test_set=True)

    # Produce a water consumption forecast and save it
    forecast(FORECAST_DAYS, cfg=cfg, model=model, save=True)
    
    # Keep a copy of preprocessed data in persistent blob storage for this run and update the historical preprocessed dataset for this rate class
    shutil.copy(cfg['PATHS']['PREPROCESSED_DATA'], rc_preprocessed_dir)
    shutil.move(cfg['PATHS']['PREPROCESSED_DATA'], rc_destination_dir)

    # Send an email to relevant parties indicating completion of training
    email_content = 'Hello,\n\nThe water demand forecasting model has successfully trained.'
    cfg_private = yaml.full_load(open("./config-private.yml", 'r'))  # Load private config data
    message = Mail(from_email='COLWaterForecastModelAlerts@no-reply.ca', to_emails=cfg_private['EMAIL']['TO_EMAILS_COMPLETION'],
                   subject='Water demand forecasting training pipeline complete', html_content=email_content)
    for email_address in cfg_private['EMAIL']['CC_EMAILS_COMPLETION']:
        message.add_cc(email_address)
    try:
        sg = SendGridAPIClient(cfg_private['EMAIL']['SENDGRID_API_KEY'])
        response = sg.send(message)
    except Exception as e:
        print('Could not send email indicating training completion. Encountered the following error:\n\n', e)


