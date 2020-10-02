import pandas as pd
import yaml
import glob
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

def load_raw_data(cfg, save_int_df=False):
    '''
    Load all entries for water consumption and combine into a single dataframe
    :param cfg: project config
    :return: a Pandas dataframe containing all water consumption records
    '''

    cat_feats = cfg['DATA']['CATEGORICAL_FEATS']
    num_feats = cfg['DATA']['NUMERICAL_FEATS']
    bool_feats = cfg['DATA']['BOOLEAN_FEATS']
    raw_data_filenames = glob.glob(cfg['PATHS']['RAW_DATA'] + "/**/*.csv")
    raw_cons_dfs = []
    print('Loading raw data from spreadsheets.')
    for filename in tqdm(raw_data_filenames):
        df = pd.read_csv(filename, encoding='ISO-8859-1', low_memory=False)    # Load a water demand CSV
        for f in df.columns:
            if ' 'in f or '"' in f:
                df.rename(columns={f: f.replace(' ', '').replace('"', '')}, inplace=True)
        df = df[['CONTRACT_ACCOUNT', 'EFFECTIVE_DATE', 'END_DATE', 'CONSUMPTION'] + num_feats + bool_feats + cat_feats]
        raw_cons_dfs.append(df) # Add to list of DFs
    raw_df = pd.concat(raw_cons_dfs, axis=0, ignore_index=True)     # Concatenate all water demand data

    print('Dropping duplicate rows.')
    raw_df = raw_df.drop_duplicates()   # Drop duplicate entries appearing in different data slices
    raw_df['EFFECTIVE_DATE'] = pd.to_datetime(raw_df['EFFECTIVE_DATE'], errors='coerce')
    raw_df['END_DATE'] = pd.to_datetime(raw_df['END_DATE'], errors='coerce')

    # Replace X's representing true for boolean feats with 1
    print('Cleaning data.')
    raw_df[bool_feats] = raw_df[bool_feats].replace({'X': 1, 'On': 1, 'Discon': 0, ' ': 0})
    raw_df['EST_READ'] = raw_df['EST_READ'].astype('object')

    # Fill in missing data
    if 'EST_READ' in cat_feats:
        raw_df['EST_READ'] = raw_df['EST_READ'].astype('object')    # Ensure all categorical features are strings
    raw_df[['CONSUMPTION'] + num_feats + bool_feats] = raw_df[['CONSUMPTION'] + num_feats + bool_feats].fillna(0)
    raw_df[cat_feats] = raw_df[cat_feats].fillna("NULL")

    if save_int_df:
        raw_df.to_csv(cfg['PATHS']['INTERMEDIATE_DATA'], sep=',', header=True, index_label=False, index=False)
    return raw_df


def calculate_ts_data(cfg, raw_df):
    '''
    Calculates estimates for daily water consumption based on provided historical data. Assumes each client consumes
    water at a uniform rate over the billing period. Produces a time series dataset indexed by date.
    at a uniform rate by each
    :param cfg: project config
    :return: a Pandas dataframe containing estimated daily water consumption
    '''

    raw_df.drop('CONTRACT_ACCOUNT', axis=1, inplace=True)
    min_date = raw_df['EFFECTIVE_DATE'].min()
    max_date = raw_df['END_DATE'].max() - timedelta(days=1)

    cat_feats = cfg['DATA']['CATEGORICAL_FEATS']
    num_feats = cfg['DATA']['NUMERICAL_FEATS']
    bool_feats = cfg['DATA']['BOOLEAN_FEATS']

    # Determine feature names for preprocessed dataset
    date_range = pd.date_range(start=min_date, end=max_date)
    daily_df_feat_init = {'Date': date_range, 'Consumption': 0}
    for f in num_feats:
        daily_df_feat_init[f + '_avg'] = 0
        daily_df_feat_init[f + '_std'] = 0
    for f in bool_feats:
        daily_df_feat_init[f] = 0
    for f in cat_feats:
        for val in raw_df[f].unique():
            daily_df_feat_init[f + '_' + str(val)] = 0
    daily_df = pd.DataFrame(daily_df_feat_init)
    daily_df.set_index('Date', inplace=True)

    def daily_consumption(cons, start_date, end_date):
        bill_period = (end_date - start_date).days
        if bill_period > 0:
            return cons / bill_period
        else:
            return 0

    # Populating features for daily prediction
    for date in tqdm(date_range):
        daily_snapshot = raw_df.loc[(raw_df['EFFECTIVE_DATE'] <= date) & (raw_df['END_DATE'] > date)]
        for f in num_feats:
            daily_df.loc[date, f + '_avg'] = daily_snapshot[f].mean()
            daily_df.loc[date, f + '_std'] = daily_snapshot[f].std()
        for f in bool_feats:
            daily_df.loc[date, f] = daily_snapshot[f].mean()
        for f in cat_feats:
            fractions = daily_snapshot[f].value_counts(normalize=True)
            for val, fraction in fractions.items():
                daily_df.loc[date, f + '_' + str(val)] = fraction
        daily_df.loc[date, 'Consumption'] = (daily_snapshot.apply(lambda row : daily_consumption(row['CONSUMPTION'],
                     row['EFFECTIVE_DATE'], row['END_DATE']), axis=1)).sum()
    return daily_df


def prepare_for_clustering(cfg, raw_df, eval_date=None):
    '''
    Create a DataFrame, indexed by client, that contains client attributes as of a given date. Computes clients' monthly
    consumption over the last months. This DataFrame is to be used for clustering clients based on their attributes and
    water usage.
    :param cfg: project config
    :param raw_df: DataFrame containing all rows from raw data
    :param eval_date: date at which to consider clients' state
    :return: DataFrame of client attributes and monthly consumption over the past year
    '''

    if eval_date is None:
        eval_date = pd.to_datetime(cfg['DATA']['EVAL_DATE'])
    min_date = eval_date - relativedelta(years=1)
    raw_df = raw_df.loc[raw_df['END_DATE'] >= min_date]

    client_df = raw_df.sort_values('END_DATE', ascending=False).drop_duplicates(subset='CONTRACT_ACCOUNT', keep='first')
    client_df.set_index('CONTRACT_ACCOUNT', inplace=True)
    raw_df = raw_df[['CONTRACT_ACCOUNT', 'EFFECTIVE_DATE', 'END_DATE', 'CONSUMPTION']]

    for m in range(12):
        monthly_cons_feat = 'CONS_' + str(m) + 'm_AGO'
        month_end_date = eval_date - relativedelta(months=m)
        month_start_date = eval_date - relativedelta(months=m+1)
        month_range = pd.date_range(month_start_date, month_end_date)
        temp_df = raw_df[(raw_df['EFFECTIVE_DATE'].between(month_start_date, month_end_date)) |
                         (raw_df['END_DATE'].between(month_start_date, month_end_date))]

        # Overlap of 2 date ranges [d11, d12] and [d21, d22] = (min(d12, d22) - max(d11,d21)).days + 1
        temp_df[monthly_cons_feat] = (temp_df['END_DATE'].clip(upper=month_end_date) - temp_df['EFFECTIVE_DATE'].clip(lower=month_start_date)).dt.days
        temp_df[monthly_cons_feat] = temp_df[monthly_cons_feat] / (temp_df['END_DATE'] - temp_df['EFFECTIVE_DATE']).dt.days * temp_df['CONSUMPTION']
        temp_df = temp_df.groupby('CONTRACT_ACCOUNT').sum()
        client_df = client_df.join(temp_df[monthly_cons_feat])

    client_df.fillna(0, inplace=True)
    return client_df


def preprocess_ts(cfg=None):
    '''
    Transform raw water demand data into a time series dataset ready to be fed into a model.
    :param cfg: project config
    '''

    run_start = datetime.today()
    tqdm.pandas()
    if cfg is None:
        cfg = yaml.full_load(open("./config.yml", 'r'))       # Load project config data

    raw_df = load_raw_data(cfg)
    client_df = prepare_for_clustering(cfg, raw_df)
    #daily_df = calculate_ts_data(cfg, raw_df)
    #daily_df.to_csv(cfg['PATHS']['PREPROCESSED_DATA'], sep=',', header=True)

    print("Done. Runtime = ", ((datetime.today() - run_start).seconds / 60), " min")
    return

if __name__ == '__main__':
    preprocess_ts()