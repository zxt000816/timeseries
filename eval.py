import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from typing import List, Dict, Tuple, Union, Any, Optional
import matplotlib.pyplot as plt
from .dataset import InformerDataset

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
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
    }

def save_plot_img(
    img_save_path: str, 
    pred: Union[List, np.ndarray], 
    real: Union[List, np.ndarray], 
    mape: float, 
    rmse: float, 
    shreshold: float,
    x_axis: Union[List, np.ndarray] = [], 
    show: bool = False
) -> None:
    plt.figure(figsize=(18, 6))
    if len(x_axis) > 0: 
        plt.plot(x_axis, real, label="real")
        plt.plot(x_axis, pred, label="predict")
    else:
        plt.plot(real, label="real")
        plt.plot(pred, label="predict")
        
    
    plt.xticks(rotation=90)
    plt.legend()
    plt.title(f'MAPE: {mape*100:.2f}%  RMSE / Shreshold: {rmse:.2f} / {shreshold:.2f}')
    plt.savefig(img_save_path)
    if show:
        plt.show()
    plt.close()
