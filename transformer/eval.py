import torch
import numpy as np

def parse_informerPL_output(outputs, pred_len, parse_method='last_point'):
    if parse_method not in ['last_point', 'sequence']:
        raise ValueError("parse_method must be one of ['last_point', 'sequence']")
    
    if parse_method == 'last_point':
        all_preds = []
        all_trues = []
        for (__preds, __trues) in outputs:
            all_preds.append(__preds.cpu().numpy()[:, -1, :])
            all_trues.append(__trues.cpu().numpy()[:, -1, :])

        all_preds = np.concatenate(all_preds)        
        all_trues = np.concatenate(all_trues)
        return all_preds, all_trues

    if parse_method == 'sequence':
        _preds = torch.Tensor([])
        _trues = torch.Tensor([])
        for (__preds, __trues) in outputs:
            _preds = torch.cat([_preds, __preds], dim=0) 
            _trues = torch.cat([_trues, __trues], dim=0)

        _preds = _preds.cpu().numpy()
        _trues = _trues.cpu().numpy()
        
        num_of_samples = len(_preds)
        if num_of_samples % pred_len == 0:
            point = num_of_samples - pred_len
        else:
            point = num_of_samples - (num_of_samples % pred_len)

        all_preds = []
        all_trues = []
        for i in range(num_of_samples):
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
    