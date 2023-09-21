import torch, json, os, optuna
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Tuple, Union, Any, Optional
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
from .models import InformerPL

def informerPL_Optimizer(
    trial: optuna.Trial,
    informer_optimize_args: Dict[str, Any],
    train_dataloader: DataLoader,
    device: torch.device,
    val_dataloader: Optional[DataLoader] = None,
):
    d_k = trial.suggest_categorical('d_k', [8, 16, 32])
    d_v = d_k
    n_heads = trial.suggest_categorical('n_heads', [1, 2, 4, 8])
    d_model = d_k * n_heads
    d_ff = d_model
    e_layer = trial.suggest_categorical('e_layer', [1, 2, 3])
    d_layer = trial.suggest_categorical('d_layer', [1, 2, 3])
    dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4, 0.5])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    weight_decay = trial.suggest_categorical('weight_decay', [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05])

    target_loss = "train_loss" if val_dataloader is None else "val_loss"

    trainer = pl.Trainer(
        max_epochs=20,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor=target_loss)],
        enable_model_summary=False,
        enable_checkpointing=False
    )

    hyperparameters = dict(
        d_k=d_k,
        d_v=d_v,
        n_heads=n_heads,
        d_model=d_model,
        d_ff=d_ff,
        e_layer=e_layer,
        d_layer=d_layer,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay
    )

    model = InformerPL(
        d_feature=informer_optimize_args["d_feature"],
        d_mark=informer_optimize_args["d_mark"],
        pred_len=informer_optimize_args["pred_len"],
        label_len=informer_optimize_args["label_len"],
        c=5,
        target=informer_optimize_args["target"],
        column_idxs=informer_optimize_args["column_idxs"],
        **hyperparameters
    )

    trainer.logger.log_hyperparams(hyperparameters)
    if val_dataloader is not None:
        trainer.fit(model, train_dataloader, val_dataloader)
    else:
        trainer.fit(model, train_dataloader)

    return trainer.callback_metrics[target_loss].item()


def optimize_informerPL(
    num_of_trials: int,
    informer_optimize_args: Dict[str, Any],
    train_dataloader: DataLoader,
    device: torch.device,
    best_params_path: str,
    val_dataloader: Optional[DataLoader] = None,
) -> Dict[str, Any]:
    if os.path.exists(best_params_path):
        print(f'Loading best params from: {best_params_path}')
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
    else:
        print(f'Optimizing hyperparameters for {num_of_trials} trials...')
        study = optuna.create_study(direction="minimize")
        def objective(trial):
            loss = informerPL_Optimizer(trial, informer_optimize_args, train_dataloader, device, val_dataloader)
            return loss
        study.optimize(objective, n_trials=num_of_trials)
        best_params = study.best_params
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f)
    return best_params