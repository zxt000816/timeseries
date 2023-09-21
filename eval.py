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

def eval_informer(
    test_dataset: InformerDataset,
    test_loader: DataLoader, 
    pred_len: int,
    label_len: int,
    model: nn.Module, 
    target: str,
    column_idxes: Dict[str, int], 
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds = np.array([])
    trues = np.array([])
    cnt = 0
    target_idx = column_idxes[target]
    # X -> Input, y -> Output
    for (batch_X, batch_y, batch_X_stamp, batch_y_stamp, x_x_axis, y_x_axis) in test_loader:
        batch_X = batch_X.float().to(device)
        batch_y = batch_y.float()
        batch_X_stamp = batch_X_stamp.float().to(device)
        batch_y_stamp = batch_y_stamp.float().to(device)

        dec_inp = torch.zeros(batch_y.shape[0], pred_len, batch_y.shape[-1]).float()
        dec_inp = torch.cat([batch_y[:, -label_len:, :], dec_inp], dim=1).to(device)

        pred = model(batch_X, batch_X_stamp, dec_inp, batch_y_stamp)
        pred = pred[:, -pred_len:, [target_idx]].to(device)
        true = batch_y[:, -pred_len:, [target_idx]].to(device)
        temp_x_axis = np.squeeze(test_dataset.time_idx[y_x_axis][:, -pred_len:, :], axis=2)

        # pred = pred[:, -pred_len:, :].to(device)
        # true = batch_y[:, -pred_len:, :].to(device)
        
        preds = np.append(preds, pred.cpu().detach().numpy()[::pred_len])
        trues = np.append(trues, true.cpu().detach().numpy()[::pred_len])
        
        if cnt == 0:
            test_x_axis = temp_x_axis[::pred_len]
        else:
            test_x_axis = np.concatenate([
                test_x_axis, 
                temp_x_axis[::pred_len]
            ])
        
        cnt += 1

    test_x_axis = test_x_axis.ravel()

    return preds, trues, test_x_axis

def informer_forecasting(
    data: pd.DataFrame, 
    target: str, 
    input_vars: List[str], 
    time_vars: List[str], 
    model: torch.nn.Module, 
    column_idxes: Dict[str, int], 
    mean: float, 
    std: float,
    seq_len: int, 
    label_len: int, 
    pred_len: int, 
    pred_dates: pd.Series, 
    device: torch.device, 
    date_type: str = 'str',
) -> Tuple[List[Union[pd.Timestamp, str]], np.ndarray]:
    model.eval()
    target = 'price'

    input = data[input_vars].values
    stamp = data[time_vars].values

    # last index
    last_idx = input.shape[0] - 1
    # start index of last input sequence
    last_seq_start_idx = last_idx - seq_len + 1
    # start index of last label sequence
    last_label_start_idx = last_idx - label_len + 1

    last_batch_X = input[last_seq_start_idx:].reshape(1, seq_len, len(input_vars))
    last_batch_X_stamp = stamp[last_seq_start_idx:].reshape(1, seq_len, len(time_vars))

    last_batch_y_stamp = []
    for pred_date in pred_dates:
        last_batch_y_stamp.append([
            pred_date.weekday() / 4 - 0.5, 
            (pred_date.day - 1) / 30 - 0.5, 
            (pred_date.month - 1) / 11 - 0.5,
            (pred_date.dayofyear - 1) / 365 - 0.5
        ])

    last_batch_y_stamp = np.concatenate([stamp[last_label_start_idx:], last_batch_y_stamp], axis=0).reshape(1, pred_len+label_len, len(time_vars))
    last_batch_y_stamp = torch.tensor(last_batch_y_stamp).float().to(device)

    last_batch_X = torch.from_numpy(last_batch_X).float().to(device)
    last_batch_X_stamp = torch.from_numpy(last_batch_X_stamp).float().to(device)
    last_batch_y_stamp = torch.tensor(last_batch_y_stamp).float().to(device)

    # create last label sequence
    last_batch_label_y = input[last_label_start_idx:].reshape(1, label_len, len(input_vars))
    last_batch_label_y = torch.from_numpy(last_batch_label_y).float()

    last_dec_inp = torch.zeros([1, pred_len, len(input_vars)]).float()
    last_dec_inp = torch.cat([last_batch_label_y, last_dec_inp], dim=1).float().to(device)

    last_pred = model(last_batch_X, last_batch_X_stamp, last_dec_inp, last_batch_y_stamp)

    last_pred = last_pred[:, -pred_len:, :].to(device)
    last_pred[:, :, column_idxes[target]] = last_pred[:, :, column_idxes[target]] * std[target] + mean[target]

    last_pred = last_pred.detach().cpu().numpy()[0, :, column_idxes[target]]
    if date_type == 'str':
        x_axis = [datetime.strftime(d, '%Y-%m-%d') for d in pred_dates]
    else:
        x_axis = list(pred_dates)

    return x_axis, last_pred

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
