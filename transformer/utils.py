from typing import List, Dict, Tuple, Union, Any, Optional
import pandas as pd
import numpy as np

def determine_sequence_len(pred_dates: pd.Series) -> Tuple[int, int, int]:
    pred_len = int(len(pred_dates))
    seq_len = int(pred_len * 2)
    label_len = int(pred_len * 0.5)
    return seq_len, label_len, pred_len