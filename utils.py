import numpy as np
import pandas as pd
from termcolor import colored
from typing import List, Dict, Tuple, Union, Any, Optional
import os

def normalize_data(df, mean=None, std=None, numeric_columns=[]):
    if len(numeric_columns) == 0:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if mean is None or std is None:
        mean = df[numeric_columns].mean()
        std = df[numeric_columns].std()
    
    df[numeric_columns] = (df[numeric_columns] - mean) / std
    
    return df, mean, std
# covert list to dict, key is the index of the list, value is the element of the list
def list2dict(l):
    d = {}
    for i, v in enumerate(l):
        d[v] = i
    return d

def resample_price(dataframe, freq):
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