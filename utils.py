import numpy as np
import pandas as pd
from termcolor import colored
from datetime import datetime
from typing import List, Dict, Tuple, Union, Any, Optional
import os, random, torch
from pandas.tseries.offsets import BDay

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def determine_period(freq: str) -> int:
    if freq == 'D':
        return count_weekdays(2022)
    elif freq == 'W':
        return 52
    elif freq == 'M':
        return 12

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

def normalize_data(df: pd.DataFrame, mean: float = None, std: float = None, numeric_columns: List = []) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    if len(numeric_columns) == 0:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if mean is None or std is None:
        mean = df[numeric_columns].mean()
        std = df[numeric_columns].std()
    
    df[numeric_columns] = (df[numeric_columns] - mean) / std
    
    return df, mean, std

def extend_dataframe(df, extend_length):
    # Get the last date in df_to_predict
    last_date = df['date'].iloc[-1]

    # Create new dates starting from the day after the last date, excluding weekends
    new_dates = pd.date_range(start=last_date + BDay(1), periods=extend_length, freq=BDay())

    # Create a new dataframe with the new dates and fill the 'price' column with 0
    new_df = pd.DataFrame({'date': new_dates})

    # Append the new data to df_to_predict
    df = pd.concat([df, new_df], ignore_index=True)

    # Fill any other missing values with 0
    df = df.fillna(0)
    df.index = range(len(df))

    return df

# covert list to dict, key is the index of the list, value is the element of the list
def list2dict(l: List) -> Dict:
    d = {}
    for i, v in enumerate(l):
        d[v] = i
    return d

def resample_price(dataframe: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = dataframe.copy()
    df['date'] = pd.to_datetime(df['date'])

    df.set_index('date', inplace=True)
    df_resampled = df['price'].resample(freq).mean()
    df_resampled = df_resampled.reset_index().dropna()
    df_resampled.index = range(len(df_resampled))

    return df_resampled

def create_folder(*args: List[str]) -> None:
    for arg in args:
        if not os.path.exists(arg):
            os.makedirs(arg)
            print(colored(f"Create folder: {arg}", "green"))