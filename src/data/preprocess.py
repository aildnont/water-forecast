import os
import pandas as pd
import yaml
import glob
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

def load_raw_data(cfg, save_raw_df=False, rate_class='all'):
    '''
    Load all entries for water consumption and combine into a single dataframe
    :param cfg: project config
    :param save_raw_df: Flag indicating whether to save the accumulated raw dataset
    :param rate_class: Rate class to filter raw data by
    :return: a Pandas dataframe containing all water consumption records
    '''

    cat_feats = cfg['DATA']['CATEGORICAL_FEATS']
    num_feats = cfg['DATA']['NUMERICAL_FEATS']
    bool_feats = cfg['DATA']['BOOLEAN_FEATS']
    feat_names = ['CONTRACT_ACCOUNT', 'EFFECTIVE_DATE', 'END_DATE', 'CONSUMPTION'] + num_feats + bool_feats + cat_feats
    raw_data_filenames = glob.glob(cfg['PATHS']['RAW_DATA_DIR'] + "/*.csv")
    raw_cons_dfs = []
    print('Loading raw data from spreadsheets.')
    for filename in tqdm(raw_data_filenames):
        df = pd.read_csv(filename, encoding='ISO-8859-1', low_memory=False)    # Load a water demand CSV
        for f in df.columns:
            if ' 'in f or '"' in f:
                df.rename(columns={f: f.replace(' ', '').replace('"', '')}, inplace=True)
        for f in feat_names:
            if f not in df.columns:
                if f in cat_feats:
                    df[f] = 'Unknown'
                else:
                    df[f] = 0.0
            if f in num_feats and df[f].dtype == 'object':
                try:
                    invalid_mask = df[f].fillna('0').str.contains('/')
                    df[f][invalid_mask] = 0
                    df[f] = df[f].astype('float64')
                except Exception as e:
                    print("Exception ", e, " in file ", filename, " feature ", f)
        df = df[feat_names]
        raw_cons_dfs.append(df) # Add to list of DFs
    raw_df = pd.concat(raw_cons_dfs, axis=0, ignore_index=True)     # Concatenate all water demand data
    print(raw_df.shape)

    print('Dropping duplicate rows.')
    raw_df = raw_df.drop_duplicates()   # Drop duplicate entries appearing in different data slices
    print('Consumption total: ', raw_df['CONSUMPTION'].sum())
    print(raw_df.shape)
    raw_df['EFFECTIVE_DATE'] = pd.to_datetime(raw_df['EFFECTIVE_DATE'], errors='coerce')
    raw_df['END_DATE'] = pd.to_datetime(raw_df['END_DATE'], errors='coerce')

    # Replace X's representing true for boolean feats with 1
    print('Cleaning data.')
    raw_df[bool_feats] = raw_df[bool_feats].replace({'X': 1, 'On': 1, 'Discon': 0, ' ': 0})
    raw_df['EST_READ'] = raw_df['EST_READ'].astype('object')

    # Fill in missing data
    if 'EST_READ' in cat_feats:
        raw_df['EST_READ'] = raw_df['EST_READ'].astype('str') + '_'     # Force treatment as string
    raw_df[['CONSUMPTION'] + num_feats + bool_feats] = raw_df[['CONSUMPTION'] + num_feats + bool_feats].fillna(0)
    raw_df[cat_feats] = raw_df[cat_feats].fillna('MISSING')

    # Filter by a rate class if desired
    if rate_class.upper() in raw_df['RATE_CLASS'].unique().tolist():
        raw_df = raw_df[raw_df['RATE_CLASS'] == rate_class.upper()]

    if save_raw_df:
        raw_df.to_csv(cfg['PATHS']['RAW_DATASET'], sep=',', header=True, index_label=False, index=False)
    return raw_df


def calculate_ts_data(cfg, raw_df):
    '''
    Calculates estimates for daily water consumption based on provided historical data. Assumes each client consumes
    water at a uniform rate over the billing period. Produces a time series dataset indexed by date.
    at a uniform rate by each
    :param cfg: project config
    :return: a Pandas dataframe containing estimated daily water consumption
    '''

    print('Calculating estimates for daily consumption and contextual features.')
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
        daily_df_feat_init[f + '_avg'] = 0.0
        daily_df_feat_init[f + '_std'] = 0.0
    for f in bool_feats:
        daily_df_feat_init[f] = 0.0
    for f in cat_feats:
        for val in raw_df[f].unique():
            daily_df_feat_init[f + '_' + str(val)] = 0.0

    daily_df = pd.DataFrame(daily_df_feat_init)
    daily_df.set_index('Date', inplace=True)

    def daily_consumption(cons, start_date, end_date):
        bill_period = (end_date - start_date + timedelta(days=1)).days      # Get length of billing period
        if bill_period > 0:
            return cons / bill_period                   # Estimate consumption per day over billing period
        else:
            return 0

    # Populating features for daily prediction
    for date in tqdm(date_range):
        daily_snapshot = raw_df.loc[(raw_df['EFFECTIVE_DATE'] <= date) & (raw_df['END_DATE'] >= date)]

        for f in num_feats:
            daily_df.loc[date, f + '_avg'] = daily_snapshot[f].mean()
            daily_df.loc[date, f + '_std'] = daily_snapshot[f].std()
        for f in bool_feats:
            daily_df.loc[date, f] = daily_snapshot[f].mean()
        for f in cat_feats:
            fractions = daily_snapshot[f].value_counts(normalize=True)
            for val, fraction in fractions.items():
                daily_df.loc[date, f + '_' + str(val)] = fraction

        try:
            daily_df.loc[date, 'Consumption'] = (daily_snapshot.apply(lambda row : daily_consumption(row['CONSUMPTION'],
                         row['EFFECTIVE_DATE'], row['END_DATE']), axis=1)).sum()
        except Exception as e:
            print(date, e)
            daily_df.loc[date, 'Consumption'] = 0.0
    return daily_df


def preprocess_new_data(cfg, save_df=True, rate_class='all'):
    '''
    Preprocess a new raw data file and merge it with preexisting preprocessed data.
    :param cfg: Project config
    :param save_df: Flag indicating whether to save the combined preprocessed dataset
    :param rate_class: Rate class to filter raw data by
    :param save_raw_data: Flag indicating whether to save the new raw dataset
    '''

    # Load new raw data and remove any rows that appear in old raw data
    old_raw_df = pd.read_csv(cfg['PATHS']['RAW_DATASET'])
    old_raw_df['EFFECTIVE_DATE'] = pd.to_datetime(old_raw_df['EFFECTIVE_DATE'], errors='coerce')
    old_raw_df['END_DATE'] = pd.to_datetime(old_raw_df['END_DATE'], errors='coerce')
    new_raw_df = load_raw_data(cfg, rate_class=rate_class, save_raw_df=True)
    if new_raw_df.shape[1] > old_raw_df.shape[1]:
        new_raw_df = new_raw_df[old_raw_df.columns]  # If additional features added, remove them
    recent_old_raw_df = old_raw_df[(old_raw_df['EFFECTIVE_DATE'] > new_raw_df['EFFECTIVE_DATE'].min()) &
        (old_raw_df['END_DATE'] > new_raw_df['END_DATE'].min())]
    new_raw_df = pd.concat([recent_old_raw_df, recent_old_raw_df, new_raw_df], axis=0, ignore_index=True)\
        .drop_duplicates(keep=False)  # Keep all rows of the new raw data that don't appear in the old one

    # Load old preprocessed data
    old_preprocessed_df = pd.read_csv(cfg['PATHS']['PREPROCESSED_DATA'])
    old_preprocessed_df['Date'] = pd.to_datetime(old_preprocessed_df['Date'], errors='coerce')
    old_preprocessed_df.set_index('Date', inplace=True)

    # Preprocess new raw data
    new_preprocessed_df = calculate_ts_data(cfg, new_raw_df)

    # Get rows in new preprocessed data that don't exist in old preprocessed data
    overlapping_dates = pd.merge(old_preprocessed_df, new_preprocessed_df, how='inner', on='Date').index
    new_df_nonoverlap = new_preprocessed_df[~new_preprocessed_df.index.isin(overlapping_dates)]

    # Combine old and new preprocessed data and update saved preprocessed data if specified
    preprocessed_df = pd.concat([old_preprocessed_df, new_df_nonoverlap], axis=0)
    if save_df:
        preprocessed_df.to_csv(cfg['PATHS']['PREPROCESSED_DATA'], sep=',', header=True)
    return preprocessed_df


def merge_raw_data(cfg=None):
    '''
    Loads all raw water demand CSVs available and merges it into one dataset, keeping the latest consumption records
    for each client if readings are duplicated.
    :param cfg: Project config
    '''

    if cfg is None:
        cfg = yaml.full_load(open("./config.yml", 'r'))  # Load project config data

    raw_dfs = []
    if os.path.exists(cfg['PATHS']['RAW_DATASET']):
        raw_dfs.append(pd.read_csv(cfg['PATHS']['FULL_RAW_DATASET'], encoding='ISO-8859-1', low_memory=False))

    new_raw_data_filenames = glob.glob(cfg['PATHS']['RAW_DATA_DIR'] + "/*.csv")
    for filename in tqdm(new_raw_data_filenames):
        new_raw_df = pd.read_csv(filename, encoding='ISO-8859-1', low_memory=False)    # Load a water demand CSV
        raw_dfs.append(new_raw_df)

    merged_raw_df = pd.concat(raw_dfs, axis=0, ignore_index=True)  # Concatenate all available water demand data
    print('Shape before deduplication: ', merged_raw_df.shape)
    merged_raw_df.drop_duplicates(['CONTRACT_ACCOUNT', 'EFFECTIVE_DATE', 'END_DATE'], keep='last', inplace=True)  # De-duplication
    print('Shape after deduplication: ', merged_raw_df.shape)
    merged_raw_df.to_csv(cfg['PATHS']['FULL_RAW_DATASET'], sep=',', header=True, index_label=False, index=False)
    return


def prepare_for_clustering(cfg, raw_df, eval_date=None, save_df=True):
    '''
    Create a DataFrame, indexed by client, that contains client attributes as of a given date. Computes clients' monthly
    consumption over the last months. This DataFrame is to be used for clustering clients based on their attributes and
    water usage.
    :param cfg: project config
    :param raw_df: DataFrame containing all rows from raw data
    :param eval_date: date at which to consider clients' state
    :return: DataFrame of client attributes and monthly consumption over the past year
    '''

    print(raw_df['CONTRACT_ACCOUNT'].nunique())
    if eval_date is None:
        eval_date = pd.to_datetime(cfg['K-PROTOTYPES']['EVAL_DATE'])
    min_date = eval_date - relativedelta(years=1)
    raw_df = raw_df.loc[raw_df['END_DATE'] >= min_date]

    # Get a DataFrame indexed by CONTRACT_ACOUNT, a key identifying clients
    client_df = raw_df.sort_values('END_DATE', ascending=False).drop_duplicates(subset=['CONTRACT_ACCOUNT'], keep='first')
    client_df.set_index('CONTRACT_ACCOUNT', inplace=True)
    raw_df = raw_df[['CONTRACT_ACCOUNT', 'EFFECTIVE_DATE', 'END_DATE', 'CONSUMPTION']]
    cat_feats = cfg['DATA']['CATEGORICAL_FEATS']
    numcl_feats = cfg['DATA']['NUMERICAL_FEATS']
    bool_feats = cfg['DATA']['BOOLEAN_FEATS']

    # Estimate monthly consumption over last year for each client
    for m in range(12):
        monthly_cons_feat = 'CONS_' + str(m) + 'm_AGO'
        numcl_feats += [monthly_cons_feat]
        month_end_date = eval_date - relativedelta(months=m)
        month_start_date = eval_date - relativedelta(months=m+1)
        temp_df = raw_df[(raw_df['EFFECTIVE_DATE'].between(month_start_date, month_end_date)) |
                         (raw_df['END_DATE'].between(month_start_date, month_end_date))]

        # Overlap of 2 date ranges [d11, d12] and [d21, d22] = (min(d12, d22) - max(d11,d21)).days + 1
        temp_df[monthly_cons_feat] = (temp_df['END_DATE'].clip(upper=month_end_date) - temp_df['EFFECTIVE_DATE'].clip(lower=month_start_date)).dt.days
        temp_df[monthly_cons_feat] = temp_df[monthly_cons_feat] / (temp_df['END_DATE'] - temp_df['EFFECTIVE_DATE']).dt.days * temp_df['CONSUMPTION']
        temp_df = temp_df.groupby('CONTRACT_ACCOUNT').sum()
        client_df = client_df.join(temp_df[monthly_cons_feat])

    # Clean up the data and save
    client_df[cat_feats] = client_df[cat_feats].fillna('MISSING')
    client_df[numcl_feats + bool_feats] = client_df[numcl_feats + bool_feats].fillna(0)
    client_df.drop(['EFFECTIVE_DATE', 'END_DATE', 'CONSUMPTION'], axis=1, inplace=True)
    client_df = client_df.drop_duplicates()
    if save_df:
        client_df.to_csv(cfg['PATHS']['CLIENT_DATA'], sep=',', header=True)
    return client_df


def preprocess_ts(cfg=None, save_df=True):
    '''
    Transform raw water demand data into a time series dataset ready to be fed into a model.
    :param cfg: project config
    :param save_df: Flag indicating whether to save the preprocessed data
    '''

    run_start = datetime.today()
    tqdm.pandas()
    if cfg is None:
        cfg = yaml.full_load(open("./config.yml", 'r'))       # Load project config data

    raw_df = load_raw_data(cfg)
    daily_df = calculate_ts_data(cfg, raw_df)
    if save_df:
        daily_df.to_csv(cfg['PATHS']['PREPROCESSED_DATA'], sep=',', header=True)

    print("Done. Runtime = ", ((datetime.today() - run_start).seconds / 60), " min")
    return daily_df

if __name__ == '__main__':
    df = preprocess_ts()