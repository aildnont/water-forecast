import pandas as pd
import yaml
import glob
from datetime import datetime, timedelta
from tqdm import tqdm

def load_raw_consumption_data(cfg):
    '''
    Load all entries for water consumption and combine into a single dataframe
    :param cfg: project config
    :return: a Pandas dataframe containing all water consumption records
    '''

    raw_data_filenames = glob.glob(cfg['PATHS']['RAW_DATA'] + "/**/*.csv")
    raw_cons_dfs = []
    print('Loading raw consumption data.')
    for filename in tqdm(raw_data_filenames):
        df = pd.read_csv(filename, encoding='ISO-8859-1', low_memory=False)    # Load a water demand CSV
        df = df[['CONTRACT_ACCOUNT', 'EFFECTIVE_DATE', 'END_DATE', 'CONSUMPTION']]
        raw_cons_dfs.append(df) # Add to list of DFs
    raw_df = pd.concat(raw_cons_dfs, axis=0, ignore_index=True)     # Concatenate all water demand data

    print('Dropping duplicate rows.')
    raw_df = raw_df.drop_duplicates()   # Drop duplicate entries appearing in different data slices
    raw_df.drop('CONTRACT_ACCOUNT', axis=1, inplace=True)
    raw_df['EFFECTIVE_DATE'] = pd.to_datetime(raw_df['EFFECTIVE_DATE'], errors='coerce')
    raw_df['END_DATE'] = pd.to_datetime(raw_df['END_DATE'], errors='coerce')
    raw_df.dropna(inplace=True)
    return raw_df


def calculate_daily_consumption(cfg):
    '''
    Calculates estimates for daily water consumption based on provided historical data. Assumes each client consumes
    water at a uniform rate over the billing period.
    at a uniform rate by each
    :param cfg: project config
    :return: a Pandas dataframe containing estimated daily water consumption
    '''

    raw_cons_df = load_raw_consumption_data(cfg)
    min_date = raw_cons_df['EFFECTIVE_DATE'].min()
    max_date = raw_cons_df['END_DATE'].max() - timedelta(days=1)
    raw_cons_df = raw_cons_df.groupby(['EFFECTIVE_DATE', 'END_DATE']).agg({'CONSUMPTION': 'sum'}).reset_index()

    cons_df = pd.DataFrame({'Date': pd.date_range(start=min_date, end=max_date), 'Consumption': 0})

    def increment_consumption(start_date, end_date, cons):
        bill_period = (end_date - start_date).days
        if bill_period > 0:
            cons_df.loc[(cons_df['Date'] >= start_date) & (cons_df['Date'] < end_date), 'Consumption'] += cons / bill_period

    print('Estimating daily water consumption.')
    raw_cons_df.progress_apply(lambda row: increment_consumption(row['EFFECTIVE_DATE'], row['END_DATE'], row['CONSUMPTION']), axis=1)
    return cons_df


def preprocess(cfg=None):
    '''
    Transform raw water demand data into a preprocessed dataset ready to be fed into a model.
    :param cfg: project config
    '''

    run_start = datetime.today()
    tqdm.pandas()
    if cfg is None:
        cfg = yaml.full_load(open("./config.yml", 'r'))       # Load project config data

    # Calculate daily water consumption
    cons_df = calculate_daily_consumption(cfg)
    cons_df.to_csv(cfg['PATHS']['CONSUMPTION_DATA'], sep=',', header=True)

    print("Done. Runtime = ", ((datetime.today() - run_start).seconds / 60), " min")
    return

if __name__ == '__main__':
    preprocess()