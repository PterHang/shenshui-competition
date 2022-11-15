import numpy as np
import pandas as pd
from config import time_steps, pred_time_steps 


def build_dataset(data):
    X, y = [], []
    seq_len = len(data) - (time_steps + pred_time_steps) + 1
    for i in range(seq_len):
        X.append(data[i: i + time_steps])
        y.append(data[i + time_steps: i + time_steps + pred_time_steps])
    return np.array(X), np.array(y)





