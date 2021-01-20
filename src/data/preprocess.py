import os
import pandas as pd
import yaml
import glob
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

def load_raw_data(cfg, save_raw_df=True, rate_class='all'):
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
    rate_class_str = 'W&S_' + rate_class.upper()
    print('Loading raw data from spreadsheets.')
    raw_df = pd.DataFrame()
    for filename in tqdm(raw_data_filenames):
        df = pd.read_csv(filename, encoding='ISO-8859-1', low_memory=False, index_col=False)    # Load a water demand CSV

        if rate_class_str in df['RATE_CLASS'].unique().tolist():
            df = df[df['RATE_CLASS'] == rate_class_str]         # Filter by a rate class if desired

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
                    df[f] = pd.to_numeric(df[f], errors='coerce')
                    df[f].fillna(0, inplace=True)
                except Exception as e:
                    print("Exception ", e, " in file ", filename, " feature ", f)
        df = df[feat_names]
        df['EFFECTIVE_DATE'] = pd.to_datetime(df['EFFECTIVE_DATE'], errors='coerce')
        df['END_DATE'] = pd.to_datetime(df['END_DATE'], errors='coerce')
        raw_df = pd.concat([raw_df, df], axis=0, ignore_index=True)     # Concatenate next batch of data
        shape1 = raw_df.shape
        raw_df.drop_duplicates(['CONTRACT_ACCOUNT', 'EFFECTIVE_DATE', 'END_DATE'], keep='last', inplace=True)   # Drop duplicate entries appearing in different data slices
        print("Deduplication: ", shape1, "-->", raw_df.shape)
    
    print('Consumption total: ', raw_df['CONSUMPTION'].sum())
    print(raw_df.shape)

    # Replace X's representing true for boolean feats with 1
    print('Cleaning data.')
    raw_df[bool_feats] = raw_df[bool_feats].replace({'X': 1, 'On': 1, 'Discon': 0, ' ': 0})
    raw_df['EST_READ'] = raw_df['EST_READ'].astype('object')

    # Fill in missing data
    if 'EST_READ' in cat_feats:
        raw_df['EST_READ'] = raw_df['EST_READ'].astype('str') + '_'     # Force treatment as string
    raw_df[['CONSUMPTION'] + num_feats + bool_feats] = raw_df[['CONSUMPTION'] + num_feats + bool_feats].fillna(0)
    raw_df[cat_feats] = raw_df[cat_feats].fillna('MISSING')

    if save_raw_df:
        raw_df.to_csv(cfg['PATHS']['RAW_DATASET'], sep=',', header=True, index_label=False, index=False)
    return raw_df


def calculate_ts_data(cfg, raw_df, start_date=None):
    '''
    Calculates estimates for daily water consumption based on provided historical data. Assumes each client consumes
    water at a uniform rate over the billing period. Produces a time series dataset indexed by date.
    :param cfg: project config
    :param raw_df: A DataFrame containing raw water consumption data
    :param start_date: The minimum date at which at which to create daily estimates for
    :return: a Pandas dataframe containing estimated daily water consumption
    '''

    print('Calculating estimates for daily consumption and contextual features.')
    raw_df.drop('CONTRACT_ACCOUNT', axis=1, inplace=True)
    if start_date is None:
        min_date = raw_df['EFFECTIVE_DATE'].min()
    else:
        min_date = start_date
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

    # TODO delete once we have no missing data
    for missing_range_endpts in cfg['DATA']['MISSING_RANGES']:
        missing_range = pd.date_range(pd.to_datetime(missing_range_endpts[0]), pd.to_datetime(missing_range_endpts[1]))
        daily_df = daily_df[~daily_df.index.isin(missing_range)] # Remove noise from missing date ranges

    return daily_df


def preprocess_ts(cfg=None, save_raw_df=True, save_prepr_df=True, rate_class='all', out_path=None):
    '''
    Transform raw water demand data into a time series dataset ready to be fed into a model.
    :param cfg: project config
    :param save_raw_df: Flag indicating whether to save intermediate raw data
    :param save_prepr_df: Flag indicating whether to save the preprocessed data
    :param rate_class: Rate class to filter by
    :param out_path: Path to save updated preprocessed data
    '''

    run_start = datetime.today()
    tqdm.pandas()
    if cfg is None:
        cfg = yaml.full_load(open("./config.yml", 'r'))       # Load project config data

    raw_df = load_raw_data(cfg, rate_class=rate_class, save_raw_df=save_raw_df)
    preprocessed_df = calculate_ts_data(cfg, raw_df)
    preprocessed_df = preprocessed_df[cfg['DATA']['START_TRIM']:-cfg['DATA']['END_TRIM']]
    if save_prepr_df:
        out_path = cfg['PATHS']['PREPROCESSED_DATA'] if out_path is None else out_path
        preprocessed_df.to_csv(out_path, sep=',', header=True)

    print("Done. Runtime = ", ((datetime.today() - run_start).seconds / 60), " min")
    return preprocessed_df


def preprocess_new_data(cfg, save_raw_df=True, save_prepr_df=True, rate_class='all', out_path=None):
    '''
    Preprocess a new raw data file and merge it with preexisting preprocessed data.
    :param cfg: Project config
    :param save_df: Flag indicating whether to save the combined preprocessed dataset
    :param rate_class: Rate class to filter raw data by
    :param out_path: Path to save updated preprocessed data
    '''

    # Load new raw data and remove any rows that appear in old raw data
    old_raw_df = pd.read_csv(cfg['PATHS']['RAW_DATASET'], low_memory=False)
    old_raw_df['EFFECTIVE_DATE'] = pd.to_datetime(old_raw_df['EFFECTIVE_DATE'], errors='coerce')
    min_preprocess_date = old_raw_df['EFFECTIVE_DATE'].max() - timedelta(days=183)  # Latest date in old raw dataset minus 1/2 year, to be safe
    new_raw_df = load_raw_data(cfg, rate_class=rate_class, save_raw_df=save_raw_df)
    
    if new_raw_df.shape[1] > old_raw_df.shape[1]:
        new_raw_df = new_raw_df[old_raw_df.columns]  # If additional features added, remove them

    # Preprocess new raw data
    new_preprocessed_df = calculate_ts_data(cfg, new_raw_df, start_date=min_preprocess_date)

    # Load old preprocessed data
    old_preprocessed_df = pd.read_csv(cfg['PATHS']['PREPROCESSED_DATA'])
    old_preprocessed_df['Date'] = pd.to_datetime(old_preprocessed_df['Date'], errors='coerce')
    old_preprocessed_df = old_preprocessed_df[old_preprocessed_df['Date'] < min_preprocess_date]
    old_preprocessed_df.set_index('Date', inplace=True)

    # Combine old and new preprocessed data
    preprocessed_df = pd.concat([old_preprocessed_df, new_preprocessed_df], axis=0)
    preprocessed_df = preprocessed_df[:-cfg['DATA']['END_TRIM']]

    if save_prepr_df:
        out_path = cfg['PATHS']['PREPROCESSED_DATA'] if out_path is None else out_path
        preprocessed_df.to_csv(out_path, sep=',', header=True)
    return preprocessed_df


def merge_raw_data(cfg=None):
    '''
    Loads all raw water demand CSVs available and merges it into one dataset, keeping the latest consumption records
    for each client if readings are duplicated.
    :param cfg: Project config
    '''

    if cfg is None:
        cfg = yaml.full_load(open("./config.yml", 'r'))  # Load project config data

    # Load old merged raw data file
    merged_raw_df = pd.DataFrame()
        
    # Loop through all raw data files and concatenate them with the old merged one, de-duplicating rows as needed
    quarterly_raw_data_filenames = glob.glob(cfg['PATHS']['RAW_DATA_DIR'] + "/*.csv")
    for filename in tqdm(quarterly_raw_data_filenames):
        quarterly_raw_df = pd.read_csv(filename, encoding='ISO-8859-1', low_memory=False)    # Load a water demand CSV
        quarterly_raw_df['EFFECTIVE_DATE'] = pd.to_datetime(quarterly_raw_df['EFFECTIVE_DATE'], errors='coerce')
        quarterly_raw_df['END_DATE'] = pd.to_datetime(quarterly_raw_df['END_DATE'], errors='coerce')
        merged_raw_df = pd.concat([merged_raw_df, quarterly_raw_df], axis=0, ignore_index=True)
        merged_raw_df.drop_duplicates(['CONTRACT_ACCOUNT', 'EFFECTIVE_DATE', 'END_DATE'], keep='last', inplace=True)  # De-duplication
    print('Shape of new merged raw data: ', merged_raw_df.shape)

    # Save the new merged raw data file
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


if __name__ == '__main__':
    df = preprocess_ts(rate_class='ins', save_raw_df=True, save_prepr_df=True)
    #cfg = yaml.full_load(open("./config.yml", 'r'))
    #df = preprocess_new_data(cfg, save_raw_df=False, save_prepr_df=True, rate_class='all')
    #merge_raw_data(cfg)
