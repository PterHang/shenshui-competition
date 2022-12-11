import numpy as np
import pandas as pd
from config import time_steps, pred_time_steps, n_features 


def build_dataset(data):
    X, y = [], []
    seq_len = len(data) - (time_steps + pred_time_steps) + 1
    for i in range(seq_len):
        X.append(data[i: i + time_steps])
        y.append(data[i + time_steps: i + time_steps + pred_time_steps])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], n_features))
    return X, y 


if __name__ == '__main__':
    train = pd.read_csv("train.csv")
    X_train, y_train = build_dataset(train[['flow_1', "flow_2", "flow_3"]])
    print(X_train.shape, y_train.shape)
