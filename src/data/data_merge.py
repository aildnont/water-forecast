import pandas as pd
import yaml
import os
import glob
from tqdm import tqdm


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
        raw_dfs.append(pd.read_csv(cfg['PATHS']['RAW_DATASET'], encoding='ISO-8859-1', low_memory=False))

    new_raw_data_filenames = glob.glob(cfg['PATHS']['RAW_DATA_DIR'] + "/*/*.csv")
    for filename in tqdm(new_raw_data_filenames):
        new_raw_df = pd.read_csv(filename, encoding='ISO-8859-1', low_memory=False)    # Load a water demand CSV
        raw_dfs.append(new_raw_df)

    merged_raw_df = pd.concat(raw_dfs, axis=0, ignore_index=True)  # Concatenate all available water demand data
    print(merged_raw_df.shape)
    merged_raw_df.drop_duplicates(['CONTRACT_ACCOUNT', 'EFFECTIVE_DATE', 'END_DATE'], keep='last', inplace=True)  # De-duplication
    print(merged_raw_df.shape)
    merged_raw_df.to_csv(cfg['PATHS']['RAW_DATASET'], sep=',', header=True, index_label=False, index=False)
    return


if __name__ == '__main__':
    cfg = yaml.full_load(open("./config.yml", 'r'))
    merge_raw_data(cfg=cfg)