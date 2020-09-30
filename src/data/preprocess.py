import pandas as pd
import numpy as np
import yaml
import glob
import os
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from joblib import dump, load

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
    raw_df.drop('CONTRACT_ACCOUNT', axis=1, inplace=True)
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


def calculate_daily_data(cfg, raw_df):
    '''
    Calculates estimates for daily water consumption based on provided historical data. Assumes each client consumes
    water at a uniform rate over the billing period.
    at a uniform rate by each
    :param cfg: project config
    :return: a Pandas dataframe containing estimated daily water consumption
    '''

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

    # Populating features for dail
    for date in tqdm(date_range):
        #date = date.strftime('%Y-%m-%d')
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

    '''
    raw_cons_df = raw_cons_df.groupby(['EFFECTIVE_DATE', 'END_DATE']).agg({'CONSUMPTION': 'sum'}).reset_index()

    cons_df = pd.DataFrame({'Date': pd.date_range(start=min_date, end=max_date), 'Consumption': 0})

    def increment_consumption(start_date, end_date, cons):
        bill_period = (end_date - start_date).days
        if bill_period > 0:
            cons_df.loc[(cons_df['Date'] >= start_date) & (cons_df['Date'] < end_date), 'Consumption'] += cons / bill_period

    print('Estimating daily water consumption.')
    raw_cons_df.progress_apply(lambda row: increment_consumption(row['EFFECTIVE_DATE'], row['END_DATE'], row['CONSUMPTION']), axis=1)
    '''
    return daily_df


def calculate_context_feats(cfg, load_ct=False, save_client_df=False):
    '''
    Calculate city-wide aggregated values for numerical and categorical features other than consumption.
    For numerical features, calculate average and standard deviation.
    For categorical features, calculate fraction of clients for each value.
    :param cfg: project config
    :param load_ct: Flag indicating whether to load saved column transformers
    :param save_client_df: Flag indicating whether to save the non-aggregated dataframe
    :return: a Pandas dataframe consisting of aggregated values for features at each available snapshot of consumption data
    '''

    raw_data_filenames = glob.glob(cfg['PATHS']['RAW_DATA'] + "/**/*.csv")
    cat_feats = cfg['DATA']['CATEGORICAL_FEATS']
    num_feats = cfg['DATA']['NUMERICAL_FEATS']
    bool_feats = cfg['DATA']['BOOLEAN_FEATS']

    # Load context features from all raw data
    print('Loading raw consumption data.')
    raw_cont_df = pd.DataFrame()
    for filename in tqdm(raw_data_filenames):
        df = pd.read_csv(filename, encoding='ISO-8859-1', low_memory=False)    # Load a water demand CSV
        df = df[['CONTRACT_ACCOUNT', 'EFFECTIVE_DATE', 'END_DATE'] + num_feats + bool_feats + cat_feats]
        df.insert(0, 'File_Date', pd.to_datetime((os.path.splitext(filename)[0]).split('_')[-1], format='%Y%m%d'))
        raw_cont_df = pd.concat([raw_cont_df, df], axis=0, ignore_index=True)
    raw_cont_df = raw_cont_df.drop_duplicates()   # Drop duplicate entries appearing in different data slices
    raw_cont_df.drop('CONTRACT_ACCOUNT', axis=1, inplace=True)

    # Replace X's representing true for boolean feats with 1
    raw_cont_df[bool_feats] = raw_cont_df[bool_feats].replace({'X': 1})

    # Fill in missing data
    if 'EST_READ' in cat_feats:
        df['EST_READ'] = df['EST_READ'].astype('object')    # Ensure all categorical features are strings
    raw_cont_df[num_feats + bool_feats] = raw_cont_df[num_feats + bool_feats].fillna(0)
    raw_cont_df[cat_feats] = raw_cont_df[cat_feats].fillna("NULL")

    if save_client_df:
        raw_cont_df.to_csv(cfg['PATHS']['CONTRACT_DATA'], sep=',', header=True, index_label=False, index=False)

    # One hot encode the single-valued categorical features
    cat_feature_idxs = [raw_cont_df.columns.get_loc(c) for c in cat_feats if c in raw_cont_df]  # List of categorical column indices
    cat_value_names = {}  # Dictionary of categorical feature indices and corresponding names of feature values
    if load_ct:
        col_trans_ohe = load(cfg['PATHS']['OHE_COL_TRANSFORMER_SV'])
        raw_cont_df = pd.DataFrame(col_trans_ohe.transform(raw_cont_df), index=raw_cont_df.index.copy())
    else:
        col_trans_ohe = ColumnTransformer(
            transformers=[('col_trans_ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'), cat_feature_idxs)],
            remainder='passthrough'
        )
        raw_cont_df = pd.DataFrame(col_trans_ohe.fit_transform(raw_cont_df), index=raw_cont_df.index.copy())
        dump(col_trans_ohe, cfg['PATHS']['OHE_COL_TRANSFORMER_SV'], compress=True)  # Save the column transformer

    # Build list of feature names for OHE dataset
    ohe_feat_names = []
    for feat in raw_cont_df.columns:
        if feat not in cat_feats:
            ohe_feat_names.append(feat)
    for i in range(len(cat_feats)):
        for value in cat_value_names[cat_feature_idxs[i]]:
            ohe_feat_names.append(cat_feats[i] + '_' + str(value))
    vec_cat_feats = ohe_feat_names.copy()
    raw_cont_df.columns = ohe_feat_names

    # Duplicate numerical features to have columns for both average and standard deviation
    #for f in num_feats:
    #    df.insert(df.columns.index(f), f + '_std', df[f], inplace=True)
    #    df.rename({f: f + '_avg'}, axis=1, inplace=True)

    num_feat_names = []
    #cont_df = pd.DataFrame(columns=['File_Date']+num_feats+bool_feats+vec_sv_cat_features)
    agg_funcs = {}
    for f in num_feats:
        agg_funcs[f] = [np.mean, np.std]
    for f in bool_feats + vec_cat_feats:
        agg_funcs[f] = [np.mean]
    cont_df = raw_cont_df.groupby('File_Date').agg(agg_funcs)


    return df


def preprocess(cfg=None, load_ct=False):
    '''
    Transform raw water demand data into a preprocessed dataset ready to be fed into a model.
    :param cfg: project config
    '''

    run_start = datetime.today()
    tqdm.pandas()
    if cfg is None:
        cfg = yaml.full_load(open("./config.yml", 'r'))       # Load project config data

    raw_df = load_raw_data(cfg)
    daily_df = calculate_daily_data(cfg, raw_df)
    daily_df.to_csv(cfg['PATHS']['PREPROCESSED_DATA'], sep=',', header=True)

    print("Done. Runtime = ", ((datetime.today() - run_start).seconds / 60), " min")
    return

if __name__ == '__main__':
    preprocess()