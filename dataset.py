from torch.utils.data import Dataset
import numpy as np
import os, json
from termcolor import colored
from .db import MYSQL_DB_API
import pandas as pd
from typing import List, Dict, Tuple, Union, Any, Optional
from .utils import resample_price

class InformerDataset(Dataset):
    def __init__(self, df, input_vars, time_vars, time_idx=None, seq_len=128, label_len=24, pred_len=12):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.input_vars = input_vars
        self.time_vars = time_vars
        self.df = df
        self.time_idx = time_idx
        self.__read_data__(df)

    def __read_data__(self, df):
        df_inputs = df[self.input_vars].values
        df_stamp = df[self.time_vars].values
        x_axis = np.arange(len(df))
        if self.time_idx:
            self.time_idx = df[self.time_idx].values

        X, y, X_stamp, y_stamp = [], [], [], []
        x_x_axis, y_x_axis = [], []
        
        for i in range(len(df) - self.seq_len - self.pred_len + 1):
            x_begin = i
            x_end = x_begin + self.seq_len
            y_begin = x_end - self.label_len
            y_end = x_end + self.pred_len

            seq_x = df_inputs[x_begin: x_end]
            seq_y = df_inputs[y_begin: y_end]
            seq_x_stamp = df_stamp[x_begin: x_end]
            seq_y_stamp = df_stamp[y_begin: y_end]
            seq_x_x_axis = x_axis[x_begin: x_end]
            seq_y_x_axis = x_axis[y_begin: y_end]

            X.append(seq_x)
            y.append(seq_y)
            X_stamp.append(seq_x_stamp)
            y_stamp.append(seq_y_stamp)
            x_x_axis.append(seq_x_x_axis)
            y_x_axis.append(seq_y_x_axis)

        X = np.array(X)
        y = np.array(y)
        X_stamp = np.array(X_stamp)
        y_stamp = np.array(y_stamp)
        x_x_axis = np.array(x_x_axis)
        y_x_axis = np.array(y_x_axis)

        self.data_x = X
        self.data_y = y
        self.X_stamp = X_stamp
        self.y_stamp = y_stamp
        self.x_x_axis = x_x_axis
        self.y_x_axis = y_x_axis
        
    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index], self.X_stamp[index], self.y_stamp[index], self.x_x_axis[index], self.y_x_axis[index]
    
    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        return len(self.data_x)

def save_seq_label_pred(saved_seq_label_pred_path: str, **kwargs) -> Union[Dict[str, Any], None]:
    saved_seq_label_pred_path_is_exist = os.path.exists(saved_seq_label_pred_path)
    if saved_seq_label_pred_path_is_exist:
        print(colored(f"save_seq_label file is already exist. Load saved data.", 'green'))
        with open(saved_seq_label_pred_path, 'r') as f:
            saved_kwargs = json.load(f)
            return saved_kwargs
        
    for key in kwargs.keys():
        if key not in ['seq_len', 'label_len', 'pred_len']:
            raise ValueError(f"Key must be one of ['seq_len', 'label_len', 'pred_len'] but {key}")
        
    with open(saved_seq_label_pred_path, 'w') as f:
        json.dump(kwargs, f)

def removeOutlier(df: pd.DataFrame, target_col: str, upper: float = 0.75, lower: float = 0.25) -> pd.DataFrame:
    Q1 = df[target_col].quantile(lower)
    Q3 = df[target_col].quantile(upper)
    IQR = Q3 - Q1
    df = df[~((df[target_col] < (Q1 - 1.5 * IQR)) | (df[target_col] > (Q3 + 1.5 * IQR)))]
    return df

def smoothingTimeSeries(df: pd.DataFrame, target_col: str, alpha: float = 0.1) -> pd.DataFrame:
    df[target_col] = df[target_col].ewm(alpha=alpha).mean()
    return df

def preprocessTimeSeries(df: pd.DataFrame, target_cols: List[str], remove_outlier: bool = True, smoothing: bool = True, **kwargs) -> pd.DataFrame:
    if remove_outlier:
        upper = kwargs['upper'] if 'upper' in kwargs.keys() else 0.75
        lower = kwargs['lower'] if 'lower' in kwargs.keys() else 0.25
        for col in target_cols:
            df = removeOutlier(df, col, upper, lower)
    if smoothing:
        alpha = kwargs['alpha'] if 'alpha' in kwargs.keys() else 0.1
        for col in target_cols:
            df = smoothingTimeSeries(df, col, alpha)
    return df

def data_pipeline(df: pd.DataFrame, freq: str, preprocess: bool = False, **kwargs) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['date'], inplace=True)
    df['price'] = df['price'].astype(float)
    df = df[['date', 'price']]

    if freq != 'Y':
        df = resample_price(df, freq)
    
    if preprocess:
        preprocess_cols = kwargs['target_cols'] if 'target_cols' in kwargs.keys() else ['price']
        remove_outlier = kwargs['remove_outlier'] if 'remove_outlier' in kwargs.keys() else True
        smoothing = kwargs['smoothing'] if 'smoothing' in kwargs.keys() else True
        df = preprocessTimeSeries(df, preprocess_cols, remove_outlier, smoothing, **kwargs)

    return df

def load_most_frequently_data(df: pd.DataFrame) -> pd.DataFrame:
    sorted_grade_ls = df['grade'].value_counts().index.tolist()
    if len(sorted_grade_ls) == 0:
        raise ValueError("grade column has no data")
    
    df = df[df['grade'] == sorted_grade_ls[0]]
    sorted_unit_ls = df['unit'].value_counts().index.tolist()
    if len(sorted_unit_ls) == 0:
        raise ValueError("unit column has no data")
    
    df = df[df['unit'] == sorted_unit_ls[0]]
    return df

def load_data(row: pd.Series, data_root: str='./data') -> Union[Dict[str, Any], None]:
    item_code_name = row.get('item_code_name')
    item_code = row.get('item_code')
    kind_code_name = row.get('kind_code_name')
    kind_code = row.get('kind_code')
    child_code_name = row.get('child_code_name')
    child_code = row.get('child_code')


    item_code_name: str = item_code_name.replace(' ', '_')
    kind_code_name: str = kind_code_name.replace(' ', '_')
    child_code_name: str = child_code_name.replace(' ', '_')
    save_data_path: str = os.path.join(data_root, f"{item_code_name}_{kind_code_name}_{child_code_name}.feather")
    if os.path.exists(save_data_path):
        print(colored(f'load existing data: {save_data_path}', 'yellow'))
        df: pd.DataFrame = pd.read_feather(save_data_path)
    else:
        print(colored(f'{save_data_path} does not exist. Load data from db', 'red'))
        db_instance: MYSQL_DB_API = MYSQL_DB_API(host='118.67.151.107', port=3306, user='ksj', password='Tipa_2023', database='forward' )
        df: pd.DataFrame = db_instance.load_data(
            f"""
            SELECT * FROM kamis_data_daily 
            WHERE item_code = '{item_code}' AND kind_code = '{kind_code}' AND child_code = '{child_code}' AND sale = '도매';
            """
        )
        if len(df) == 0:
            raise ValueError(f'item_code: {item_code}, kind_code: {kind_code}, child_code: {child_code} has no data')
        df: pd.DataFrame = load_most_frequently_data(df)
        if len(df) < 2000:
            print(colored(f'item_code: {item_code}, kind_code: {kind_code}, child_code: {child_code} has less than 2000 data!', 'red'))
            return False
        df.index = range(len(df))
        df.to_feather(save_data_path)

    df: pd.DataFrame = df.rename(columns={'day': 'date'})
    unit: str = df['unit'].unique().item()
    grade: str = df['grade'].unique().item()
    ret_dict: Dict = {
        'df': df, 
        'item_code': item_code,
        'kind_code': kind_code,
        'child_code': child_code,
        'item_code_name': item_code_name, 
        'kind_code_name': kind_code_name, 
        'child_code_name': child_code_name, 
        'unit': unit, 
        'grade': grade
    }
    return ret_dict

def data_info_printer(**args):
    item_code_name = args['item_code_name']
    kind_code_name = args['kind_code_name']
    child_code_name = args['child_code_name']
    item_code = args['item_code']
    kind_code = args['kind_code']
    child_code = args['child_code']
    unit = args['unit']
    grade = args['grade']
    freq = args['freq']
    seq_len = args['seq_len']
    label_len = args['label_len']
    pred_len = args['pred_len']
    tot_rows = args['tot_rows']

    print(f"item_code_name: {item_code_name} kind_code_name: {kind_code_name} child_code_name: {child_code_name}")
    print(f"item_code: {item_code} kind_code: {kind_code} child_code: {child_code} unit: {unit} grade: {grade}")
    print(colored(f"freq: {freq} seq_len: {seq_len} label_len: {label_len} pred_len: {pred_len}", "yellow"))
    print(colored(f"Total rows: {tot_rows}", "yellow"))

def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    # calculate date of the 1 year ago by the last date
    last_date = df['date'].max()
    last_date = last_date.replace(year=last_date.year - 1)

    df_train = df[df['date'] <= last_date]
    df_test = df[df['date'] > last_date]

    return df_train, df_test, last_date