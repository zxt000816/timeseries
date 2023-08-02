import pandas as pd
import numpy as np

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters, ComprehensiveFCParameters
from typing import List, Dict, Tuple, Union, Any, Optional
from statsmodels.tsa.seasonal import STL

def create_stl_features(df: pd.DataFrame, period: int) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    stl = STL(df['price'], period=period)
    result = stl.fit()
    
    df['trend'] = result.trend
    df['seasonal'] = result.seasonal
    df['resid'] = result.resid
    return df.reset_index()
    
# def create_time_embedding(df: pd.DataFrame) -> pd.DataFrame:
#     df['day_of_week'] = df["date"].apply(lambda row: row.weekday() / 4 - 0.5, 1)
#     df['day_of_month'] = df["date"].apply(lambda row: (row.day - 1) / 30 - 0.5, 1)
#     df['month_of_year'] = df["date"].apply(lambda row: (row.month - 1) / 11 - 0.5, 1)
#     df['day_of_year'] = df["date"].apply(lambda row: (row.dayofyear - 1) / 365 - 0.5, 1)
#     return df

def create_time_embedding(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    feature_funcs = {
        'day_of_week': lambda row: row.weekday() / 4 - 0.5,
        'day_of_month': lambda row: (row.day - 1) / 30 - 0.5,
        'month_of_year': lambda row: (row.month - 1) / 11 - 0.5,
        'day_of_year': lambda row: (row.dayofyear - 1) / 365 - 0.5
    }

    for feature in features:
        if feature in feature_funcs:
            df[feature] = df["date"].apply(feature_funcs[feature], 1)

    return df

def feature_engineering(df: pd.DataFrame, freq: str, mode: str ='min') -> pd.DataFrame:
    max_timeshift: int
    min_timeshift: int
    params: ComprehensiveFCParameters

    df['symbol'] = 'symbol'

    if freq == 'D':
        max_timeshift = 21
        min_timeshift = 4
    elif freq == 'W':
        max_timeshift = 15
        min_timeshift = 3
    elif freq == 'M':
        max_timeshift = 11
        min_timeshift = 2

    if mode == 'min':
        params = MinimalFCParameters()
        params.pop('length', None)
    elif mode == 'efficient':
        params = EfficientFCParameters()
    elif mode == 'comprehensive':
        params = ComprehensiveFCParameters()
    
    df_rolled = roll_time_series(
        df,
        column_id="symbol",
        column_sort="date",
        max_timeshift=max_timeshift,
        min_timeshift=min_timeshift
    )

    features = extract_features(
        df_rolled.drop("symbol", axis=1), 
        column_id="id", column_sort="date", column_value="price", 
        impute_function=impute, show_warnings=False, default_fc_parameters=params
    )

    features = features.set_index(features.index.map(lambda x: x[1]), drop=True)
    features = features.reset_index(names=['date'])

    new_df = pd.merge(df.drop('symbol', axis=1), features, how='left', on='date').dropna()
    new_df.index = range(len(new_df))
    return new_df

if __name__ == '__main__':
    data_path = '/home/ubuntu/projects/Future_Trading_Agricultural_Forecasting/data/과일류_감귤_감귤.feather'

    df = pd.read_feather(data_path)

    # ['item_code', 'kind_code', 'child_code', 'unit', 'day', 'sale', 'grade', 'price']
    df = df[['day', 'price']]
    df = df.rename(columns={'day': 'date'})
    df['symbol'] = '감귤'
    df['date'] = pd.to_datetime(df['date'])

    df_rolled = roll_time_series(
        df,
        column_id="symbol",
        column_sort="date",
        max_timeshift=21,
        min_timeshift=4
    )

    params = MinimalFCParameters()
    params.pop('length', None)

    features = extract_features(
        df_rolled.drop("symbol", axis=1), 
        column_id="id", column_sort="date", column_value="price", 
        impute_function=impute, show_warnings=False, default_fc_parameters=params
    )

    features = features.set_index(features.index.map(lambda x: x[1]), drop=True)
    features = features.reset_index(names=['date'])
    
    new_df = pd.merge(df.drop('symbol', axis=1), features, how='left', on='date').dropna()