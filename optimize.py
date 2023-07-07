from .informer.model import Informer
import torch, json, os, optuna
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from termcolor import colored
from typing import List, Dict, Tuple, Union, Any, Optional

def optimize_informer(trial, input_vars, time_vars, loader, pred_len, label_len, device):
    d_k = trial.suggest_categorical('d_k', [8, 16, 32])
    d_v = d_k
    n_heads = trial.suggest_categorical('n_heads', [1, 2, 4, 8])
    d_model = d_k * n_heads
    d_ff = d_model
    e_layer = trial.suggest_categorical('e_layer', [1, 2, 3])
    d_layer = trial.suggest_categorical('d_layer', [1, 2, 3])
    dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4, 0.5])
    learning_rate = trial.suggest_categorical('lr', [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05])
    weight_decay = trial.suggest_categorical('weight_decay', [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05])

    model = Informer(
        d_feature=len(input_vars), 
        d_mark=len(time_vars),
        d_k=d_k,
        d_v=d_v,
        d_model=d_model,
        d_ff=d_ff,
        e_layer=e_layer,
        d_layer=d_layer,
        dropout=dropout
    ).to(device)
    
    critertion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for _ in range(10):
        losses = []
        for (batch_X, batch_y, batch_X_stamp, batch_y_stamp, _, _) in loader:
            optimizer.zero_grad()

            batch_X = batch_X.float().to(device)
            batch_y = batch_y.float()
            batch_X_stamp = batch_X_stamp.float().to(device)
            batch_y_stamp = batch_y_stamp.float().to(device)

            dec_inp = torch.zeros(batch_y.shape[0], pred_len, batch_y.shape[-1]).float()
            dec_inp = torch.cat([batch_y[:, -label_len:, :], dec_inp], dim=1).to(device)

            pred = model(batch_X, batch_X_stamp, dec_inp, batch_y_stamp)
            pred = pred[:, -pred_len:, :].to(device)
            true = batch_y[:, -pred_len:, :].to(device)

            loss = critertion(pred, true)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            
    return np.mean(losses)

def optimize_hyperparameters(num_of_trials: int, input_vars: List[str], time_vars: List[str], temp_loader: DataLoader, pred_len: int, label_len: int, device: torch.device, best_params_path: str) -> Dict[str, Any]:
    if os.path.exists(best_params_path):
        print(colored(f'Loading best params from: {best_params_path}', 'green'))
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
    else:
        print(colored(f'Optimizing hyperparameters for {num_of_trials} trials...', 'yellow'))
        study = optuna.create_study(direction='minimize')
        def objective(trial):
            train_loss = optimize_informer(trial, input_vars, time_vars, temp_loader, pred_len, label_len, device)
            return train_loss
        study.optimize(objective, n_trials=num_of_trials)
        best_params = study.best_params
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f)

    return best_params