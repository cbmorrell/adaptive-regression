import numpy as np
import torch


def check_input(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x


def calculate_r2(predictions, labels, mean = True):
    predictions = check_input(predictions)
    labels = check_input(labels)

    ssr = np.sum((predictions - labels) ** 2, axis=0)
    sst = np.sum((labels - labels.mean(axis=0)) ** 2, axis=0)
    r2 = 1 - ssr/sst
    if mean:
        r2 = r2.mean()
    return r2


def calculate_mse(predictions, labels, mean = True):
    predictions = check_input(predictions)
    labels = check_input(labels)

    values = (labels - predictions) ** 2
    mse = np.sum(values, axis=0) / labels.shape[0]
    if mean:
        mse = mse.mean()
    return mse


def calculate_mape(predictions, labels, mean = True):
    predictions = check_input(predictions)
    labels = check_input(labels)

    values = np.abs((labels - predictions) / np.maximum(np.abs(labels), np.finfo(np.float64).eps))    # some values could be 0, so take epsilon if that's the case to avoid inf
    mape = np.sum(values, axis=0) / labels.shape[0]
    if mean:
        mape = mape.mean()
    return mape


def calculate_rmse(predictions, labels, mean = True):
    predictions = check_input(predictions)
    labels = check_input(labels)

    rmse = np.sqrt(calculate_mse(predictions, labels, mean=False))
    if mean:
        rmse = rmse.mean()
    return rmse


def calculate_nrmse(predictions, labels, mean = True):
    predictions = check_input(predictions)
    labels = check_input(labels)

    nrmse = calculate_rmse(predictions, labels, mean=False) / (labels.max(axis=0) - labels.min(axis=0))
    if mean:
        nrmse = nrmse.mean()
    return nrmse


def calculate_mae(predictions, labels, mean = True):
    predictions = check_input(predictions)
    labels = check_input(labels)

    residuals = np.abs(predictions - labels)
    mae = np.mean(residuals, axis=0)
    if mean:
        mae = mae.mean()
    return mae

    
def calculate_maxae(predictions, labels, mean = True):
    predictions = check_input(predictions)
    labels = check_input(labels)

    residuals = np.abs(predictions - labels)
    maxae = np.max(residuals, axis=1)
    if mean:
        maxae = maxae.mean()
    return maxae

