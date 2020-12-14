import yaml
import argparse
import traceback
from azureml.core import Run
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from src.data.preprocess import merge_raw_data, preprocess_new_data

parser = argparse.ArgumentParser()
parser.add_argument('--newrawdatadir', type=str, help="Directory containing new raw datasets")
parser.add_argument('--intermediaterawdatasets', type=str, help="Path to intermediate raw datasets")
parser.add_argument('--rawmargeddataset', type=str, help="Path to aggregated raw consumption data CSV")
parser.add_argument('--preprocesseddatadir', type=str, help="Directory containing preprocessed data for each rate class")
args = parser.parse_args()
run = Run.get_context()

# Modify paths in config file based the Azure datastore paths passed as arguments.
cfg = yaml.full_load(open("./config.yml", 'r'))  # Load config data
cfg['PATHS']['RAW_DATA_DIR'] = args.newrawdatadir
cfg['PATHS']['FULL_RAW_DATASET'] = args.rawmergeddataset

# For conveying email errors, if necessary
error_email_msg = ''

# Perform raw data merge and de-duplication
try:
    merge_raw_data(cfg)
except:
    error_email_msg += 'Error encountered in merging raw data:\n' + traceback.format_exc() + '\n\n'

# Run preprocessing for all rate classes in the below list
RATE_CLASSES = ['all', 'RESI', 'COMM', 'IND', 'INS']
for i in range(len(RATE_CLASSES)):

    # Modify paths in config file based the Azure datastore paths passed as arguments.
    cfg['PATHS']['RAW_DATASET'] = args.rawdataset + '/' + RATE_CLASSES[i] + '/' \
                                  + cfg['PATHS']['RAW_DATASET'].split('/')[-1]
    cfg['PATHS']['PREPROCESSED_DATA'] = args.preprocesseddatadir + '/' + RATE_CLASSES[i] + '/' \
                                        + cfg['PATHS']['PREPROCESSED_DATA'].split('/')[-1]

    # Load and preprocess new raw data, merge with old preprocessed data, and save the result
    save_raw_data = (i == len(RATE_CLASSES) - 1)
    try:
        preprocess_new_data(cfg, save_df=True, rate_class=RATE_CLASSES[i])
    except:
        error_email_msg += 'Error encountered in preprocessing new raw data:\n' + traceback.format_exc() + '\n\n'


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

