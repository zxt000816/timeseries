import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from typing import List, Dict, Tuple, Union, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def save_prediction_performance(
        prediction_result_eval_path: str, 
        item_code_name: str,
        kind_code_name: str, 
        child_code_name: str,
        unit: str,
        grade: str,
        error_rate: float, 
        rmse: float,
        mape: float,
        shreshold: float,
        freq: str,
        method: Union[str, None] = None
    ) -> None:
        if freq not in ['DWM', 'Y']:
            raise ValueError(f'freq must be one of DWM or Y, but got {freq}')

        with open(prediction_result_eval_path, 'a') as f:
            if freq == 'DWM':
                if not method:
                    raise ValueError('method must be given when freq is DWM')
                f.write(f'{method} {item_code_name} {kind_code_name} {child_code_name} {unit} {grade} {error_rate} {rmse} {mape} {shreshold}\n')
            else:
                f.write(f'{item_code_name} {kind_code_name} {child_code_name} {unit} {grade} {error_rate} {rmse} {mape} {shreshold}\n')

def error_report(ground_truth: Union[List, np.ndarray], predictions: Union[List, np.ndarray]) -> Dict[str, float]:
    # rmse
    rmse = mean_squared_error(ground_truth, predictions, squared=False)

    # mape
    mape = mean_absolute_percentage_error(ground_truth, predictions)

    # r2
    r2 = r2_score(ground_truth, predictions)

    return {
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'R2': float(r2),
    }

def timeseries_plot(
    ground_truth: Union[np.ndarray, List],
    predictions: Union[np.ndarray, List],
    x_axis: pd.Series,
    freq: str,
    figsize: Tuple=(10, 6),
    title: Optional[str]=None,
    angle: int=45,
    interval: Optional[int]=None,
    axvline: Optional[pd.Timestamp]=None,
    save_path: Optional[str]=None,
    show: bool=True,
):
    if interval is None:
        interval= int(ground_truth.shape[0] / 20)
    
    x_axis_locator, x_axis_offset = None, None
    if freq == 'M':
        x_axis_format = '%Y-%m'
        x_axis_locator = mdates.MonthLocator(interval=interval)
        x_axis_offset = pd.DateOffset(months=1)
    else:
        x_axis_format = '%Y-%m-%d'
        x_axis_locator = mdates.DayLocator(interval=interval)
        x_axis_offset = pd.DateOffset(days=1)

    plt.style.use('default')
    plt.figure(figsize=figsize)
    plt.plot(x_axis, ground_truth, label="ground-truth")
    plt.plot(x_axis, predictions, label="predictions")
    plt.xticks(rotation=angle)
    plt.legend()

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(x_axis_format))
    plt.gca().xaxis.set_major_locator(x_axis_locator)
    plt.gca().xaxis.set_tick_params(rotation=angle)
    plt.gca().set_xbound(
        x_axis.min()-x_axis_offset, 
        x_axis.max()+x_axis_offset
    )
    plt.grid()

    if title is not None:
        plt.title(title)
    if axvline is not None:
        plt.axvline(x=axvline, color='r', linestyle='--', label='train-test split')
    if save_path is not None:
        if not os.path.exists(save_path):
            plt.savefig(save_path)
        else:
            print(f'{save_path} already exists')
    if show is True:
        plt.show()

    plt.close()