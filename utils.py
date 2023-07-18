import numpy as np
import pandas as pd
from termcolor import colored
from datetime import datetime
from statsmodels.tsa.seasonal import STL
from typing import List, Dict, Tuple, Union, Any, Optional
import os

def determine_period(freq: str) -> int:
    if freq == 'D':
        return count_weekdays(2022)
    elif freq == 'W':
        return 52
    elif freq == 'M':
        return 12

def create_stl_features(df: pd.DataFrame, period: int) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    stl = STL(df['price'], period=period)
    result = stl.fit()
    
    df['trend'] = result.trend
    df['seasonal'] = result.seasonal
    df['resid'] = result.resid
    return df.reset_index()

def count_weekdays(year: int):
    count = 0
    for month in range(1, 13):
        for day in range(1, 32):
            try:
                date = datetime(year, month, day)
            except ValueError:
                continue
            if date.weekday() < 5:
                count += 1
    return count

def normalize_data(df: pd.DataFrame, mean: pd.Series = None, std: pd.Series = None, numeric_columns: List = []) -> Tuple[pd.DataFrame, float, float]:
    if len(numeric_columns) == 0:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if mean is None or std is None:
        mean = df[numeric_columns].mean()
        std = df[numeric_columns].std()
    
    df[numeric_columns] = (df[numeric_columns] - mean) / std
    
    return df, mean, std

# covert list to dict, key is the index of the list, value is the element of the list
def list2dict(l: List) -> Dict:
    d = {}
    for i, v in enumerate(l):
        d[v] = i
    return d

def resample_price(dataframe: pd.DataFrame, freq: str) -> pd.DataFrame:
    dataframe['date'] = pd.to_datetime(dataframe['date'])

    dataframe.set_index('date', inplace=True)
    dataframe_resampled = dataframe['price'].resample(freq).mean()
    dataframe_resampled = dataframe_resampled.reset_index().dropna()

    return dataframe_resampled

def create_folder(*args: List[str]) -> None:
    for arg in args:
        if not os.path.exists(arg):
            os.makedirs(arg)
            print(colored(f"Create folder: {arg}", "green"))