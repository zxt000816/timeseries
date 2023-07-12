import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import nn
from termcolor import colored
from typing import List, Dict, Tuple, Union, Any, Optional

def train_informer(
    train_loader: DataLoader,
    pred_len: int,
    label_len: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    critertion: nn.Module,
    target: str,
    column_idxes: Dict[str, int], 
    device: torch.device,
) -> float:
    model.train()
    losses = []

    # with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", ncols='50%') as pbar:
    for (batch_X, batch_y, batch_X_stamp, batch_y_stamp, _, _) in train_loader:
        optimizer.zero_grad()

        batch_X = batch_X.float().to(device)
        batch_y = batch_y.float()
        batch_X_stamp = batch_X_stamp.float().to(device)
        batch_y_stamp = batch_y_stamp.float().to(device)

        dec_inp = torch.zeros(batch_y.shape[0], pred_len, batch_y.shape[-1]).float()
        dec_inp = torch.cat([batch_y[:, -label_len:, :], dec_inp], dim=1).to(device)

        pred = model(batch_X, batch_X_stamp, dec_inp, batch_y_stamp).to(device)
        # true = batch_y.to(device)
        pred = pred[:, -pred_len:, [column_idxes[target]]].to(device)
        true = batch_y[:, -pred_len:, [column_idxes[target]]].to(device)

        loss = critertion(pred, true)
        losses.append(loss.item())

        # pbar.set_postfix(loss=loss.item(), mean_loss=np.mean(losses), best_loss=best_loss)
        # pbar.refresh()

        loss.backward()
        optimizer.step()
    
    mean_loss = np.mean(losses)
    
    return mean_loss