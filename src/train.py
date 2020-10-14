import pandas as pd
import yaml
import os
from src.models.prophet import ProphetModel
from src.models.arima import ARIMAModel
from src.models.sarima import SARIMAModel
from src.data.preprocess import preprocess_ts

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

# Load preprocessed client data
try:
    df = pd.read_csv(cfg['PATHS']['PREPROCESSED_DATA'])
except FileNotFoundError:
    print("No file found at " + cfg['PATHS']['PREPROCESSED_DATA'] + ". Running preprocessing of client data.")
    df = preprocess_ts(cfg, save_df=False)
df['Date'] = pd.to_datetime(df['Date'])
df = df[50:-50]     # For now, take off dates at start and end due to incomplete data at boundaries


# Define training and test sets
train_df = df[:int((1 - cfg['DATA']['TEST_FRAC'])*df.shape[0])]
test_df = df[int((1 - cfg['DATA']['TEST_FRAC'])*df.shape[0]):]

# Define model
MODELS_DEFS = {
    'PROPHET': ProphetModel,
    'ARIMA': ARIMAModel,
    'SARIMA': SARIMAModel,
}
model_def = MODELS_DEFS.get(cfg['TRAIN']['MODEL'].upper(), lambda: "Invalid model specified in cfg['TRAIN']['MODEL']")
model = model_def(cfg['HPARAMS'][cfg['TRAIN']['MODEL'].upper()])

# Fit the model
if model.univariate:
    train_df = train_df[['Date', 'Consumption']]
    test_df = test_df[['Date', 'Consumption']]
model.fit(train_df)

# Evaluate the model on the test set
model.evaluate(train_df, test_df, save_dir=cfg['PATHS']['EXPERIMENTS'])
