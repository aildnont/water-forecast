import os
import argparse
import yaml
import shutil
import datetime
from azureml.core import Run
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from src.train import *
from src.predict import *

parser = argparse.ArgumentParser()
parser.add_argument('--trainoutputdir', type=str, help="directory for pipeline outputs")
parser.add_argument('--preprocesseddatadir', type=str, help="Directory containing preprocessed data for each rate class")
TEST_DAYS = os.getenv("AML_PARAMETER_TEST_DAYS")            # Number of days in test set, for model evaluation
FORECAST_DAYS = os.getenv("AML_PARAMETER_FORECAST_DAYS")    # Number of days to produce future forecast for
args = parser.parse_args()
cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
run = Run.get_context()

# All outputs from this run will be in the same directory
destination_dir = args.trainoutputdir + cur_date + '/'
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Train a model for the whole system and for each rate class
RATE_CLASSES = ['all', 'RESI', 'COMM', 'IND', 'INS']
for rate_class in RATE_CLASSES:

    rc_destination_dir = destination_dir + rate_class + '/'
    if not os.path.exists(rc_destination_dir):
        os.makedirs(rc_destination_dir)

    # Get relevant paths on Azure blob storage. Create new folders as necessary.
    cfg = yaml.full_load(open(os.getcwd() + "./config.yml", 'r'))
    cfg['PATHS']['PREPROCESSED_DATA'] = args.preprocesseddatadir + '/' + cfg['PATHS']['PREPROCESSED_DATA'].split('/')[-1]
    cfg['PATHS']['MODELS'] = rc_destination_dir + cfg['PATHS']['MODELS'].split('/')[-1]
    if not os.path.exists(cfg['PATHS']['MODELS']):
        os.makedirs(cfg['PATHS']['MODELS'])
    cfg['PATHS']['SERIALIZATIONS'] = rc_destination_dir + cfg['PATHS']['SERIALIZATIONS'].split('/')[-1]
    if not os.path.exists(cfg['PATHS']['SERIALIZATIONS']):
        os.makedirs(cfg['PATHS']['SERIALIZATIONS'])
    cfg['PATHS']['EXPERIMENTS'] = rc_destination_dir + cfg['PATHS']['EXPERIMENTS'].split('/')[-1]
    if not os.path.exists(cfg['PATHS']['EXPERIMENTS']):
        os.makedirs(cfg['PATHS']['EXPERIMENTS'])
    cfg['PATHS']['FORECAST_VISUALIZATIONS'] = rc_destination_dir + cfg['PATHS']['FORECAST_VISUALIZATIONS'].split('/')[-1]
    if not os.path.exists(cfg['PATHS']['FORECAST_VISUALIZATIONS']):
        os.makedirs(cfg['PATHS']['FORECAST_VISUALIZATIONS'])
    cfg['PATHS']['LOGS'] = rc_destination_dir + cfg['PATHS']['LOGS'].split('/')[-1]
    if not os.path.exists(cfg['PATHS']['LOGS']):
        os.makedirs(cfg['PATHS']['LOGS'])
    cfg['PATHS']['PREDICTIONS']= rc_destination_dir + cfg['PATHS']['PREDICTIONS'].split('/')[-1]
    if not os.path.exists(cfg['PATHS']['PREDICTIONS']):
        os.makedirs(cfg['PATHS']['PREDICTIONS'])

    # Keep a copy of preprocessed data in persistent blob storage for this run
    shutil.move(cfg['PATHS']['PREPROCESSED_DATA'], rc_destination_dir)

    # Train a model, using a fixed size test set and save the metrics, along with test set forecast visualization
    cfg['DATA']['TEST_DAYS'] = TEST_DAYS      # Test set is half a year
    test_forecast_metrics, _ = train_single(cfg, save_model=True, save_metrics=True, fixed_test_set=True)

    # Record test forecast metrics
    for metric in test_forecast_metrics:
        run.log(rate_class + '_test_' + metric, test_forecast_metrics[metric])

    # Train a model using all available data
    cfg['DATA']['TEST_DAYS'] = 0
    _, model = train_single(cfg, save_model=True, save_metrics=False, fixed_test_set=True)

    # Produce a water consumption forecast and save it
    forecast(FORECAST_DAYS, model, save=True)

    # Send an email to AI manager indicating completion of training
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


