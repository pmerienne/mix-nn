import numpy as np

from keras import backend as K


def r2(y_true, y_pred):
    '''
    R-squared metric. Only computes a batch-wise R-squared.
    Computes the r-squared, a metric for regression of how much
    variance in the target variable is explained.
    '''
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def r2_loss(y_true, y_pred):
    '''
    R-squared metric. Only computes a batch-wise R-squared.
    Computes the r-squared, a metric for regression of how much
    variance in the target variable is explained.
    '''
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return SS_res / (SS_tot + K.epsilon())


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
