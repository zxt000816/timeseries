import torch
import numpy as np

def parse_informerPL_output(outputs, pred_len):
    _preds = torch.Tensor([])
    _trues = torch.Tensor([])
    for (__preds, __trues) in outputs:
        _preds = torch.cat([_preds, __preds], dim=0) 
        _trues = torch.cat([_trues, __trues], dim=0)

    _preds = _preds.cpu().numpy()
    _trues = _trues.cpu().numpy()

    # if scaler is not None:
    #     _preds = scaler.inverse_transform(_preds)
    #     _trues = scaler.inverse_transform(_trues)

    all_preds = []
    all_trues = []
    num_of_samples = len(_preds)
    for i in range(num_of_samples):
        point = num_of_samples - (num_of_samples % pred_len)
        if i < point:
            if (i % pred_len == 0):
                all_preds.append(_preds[i])
                all_trues.append(_trues[i])
        elif i == point:
            all_preds.append(_preds[i])
            all_trues.append(_trues[i])
        else:
            all_preds.append(_preds[i][-1:])
            all_trues.append(_trues[i][-1:])

    all_preds = np.concatenate(all_preds)        
    all_trues = np.concatenate(all_trues)
    return all_preds, all_trues
    