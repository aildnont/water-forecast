import argparse
import traceback
from azureml.core import Run
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from src.data.preprocess import *

parser = argparse.ArgumentParser()
parser.add_argument('--rawdatadir', type=str, help="Water consumption raw data directory")
parser.add_argument('--rawdataset', type=str, help="Path to saved raw consumption data CSV")
parser.add_argument('--preprocesseddata', type=str, help="Path to preprocessed data CSV")
args = parser.parse_args()
run = Run.get_context()

# Modify paths in config file based the Azure datastore paths passed as arguments.
cfg = yaml.full_load(open("./config.yml", 'r'))  # Load config data
cfg['PATHS']['RAW_DATA_DIR'] = args.rawdatadir + '/' + cfg['PATHS']['RAW_DATA'].split('/')[-1]
cfg['PATHS']['RAW_DATASET'] = args.rawdataset + '/' + cfg['PATHS']['RAW_DATASET'].split('/')[-1]
cfg['PATHS']['PREPROCESSED_DATA'] = args.preprocesseddata + '/' + cfg['PATHS']['PREPROCESSED_DATA'].split('/')[-1]

# For conveying email errors, if necessary
error_email_msg = ''

# Perform raw data merge and de-duplication
try:
    merge_raw_data(cfg)
except:
    error_email_msg += 'Error encountered in merging raw data:\n' + traceback.format_exc() + '\n\n'

# Load and preprocess new raw data, merge with old preprocessed data, and save the result
try:
    preprocess_ts(cfg, save_df=True)
except:
    error_email_msg += 'Error encountered in merging raw data:\n' + traceback.format_exc() + '\n\n'

# Send email containing error information to prespecified recipients and cancel subsequent execution
if len(error_email_msg) > 0:
    error_email_msg = 'Preprocessing step of training pipeline failed. See the below error.\n\n' + error_email_msg
    cfg_private = yaml.full_load(open("./config-private.yml", 'r'))  # Load private config data
    message = Mail(from_email='COLWaterForecastModelAlerts@no-reply.ca', to_emails=cfg_private['EMAIL']['TO_EMAILS'],
                   subject='Water forecasting training pipeline failed', html_content=error_email_msg)
    for email_address in cfg_private['EMAIL']['CC_EMAILS']:
        message.add_cc(email_address)
    try:
        sg = SendGridAPIClient(cfg_private['EMAIL']['SENDGRID_API_KEY'])
        response = sg.send(message)
    except Exception as e:
        print(e)
    raise Exception(error_email_msg)    # Cancel remainder of pipeline


