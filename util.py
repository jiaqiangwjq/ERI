import numpy as np


def calu_PCC(pred, truth):
    pred_mean = np.mean(pred, axis=0)
    pred_std = np.std(pred, axis=0)
    truth_mean = np.mean(truth, axis=0)
    truth_std = np.std(truth, axis=0)
    pcc = np.mean((pred - pred_mean) * (truth - truth_mean), axis=0) / (pred_std * truth_std)
    return pcc.tolist(), np.mean(pcc)
