# from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy import signal

import logging
logger = logging.getLogger(__name__)


def fourier_filter(field, window, mode):
    return signal.fftconvolve(field, np.ones(window), mode=mode)
        

def fourier_filter_eps(field, window, mode):
    return signal.fftconvolve(field, np.ones(window), mode=mode)
        

def fourier_fss(fcst, obs, threshold, window, percentiles, mode):
    """
    Compute the fractional skill score using convolution
    :paramfcst: nd-array, forecast field
    :paramobs: nd-array, observation field.
    :param window: integer, window size.
    :param percentiles: threshold list is treated as percentiles [0 ... 100]
    :return: tuple of FSS numerator, denominator and score.
    """
    ny, nx = fcst.shape
    if mode=='valid' and any(np.array(window) > np.array(fcst.shape)):
        return np.nan, np.nan, np.nan, np.nan
    if percentiles:
      fhat = fourier_filter(fcst >= np.percentile(fcst, threshold), window, mode)
      ohat = fourier_filter(obs >= np.percentile(obs, threshold), window, mode)
    else:
        fhat = fourier_filter(fcst > threshold, window, mode)
        ohat = fourier_filter(obs > threshold, window, mode)
    num = np.nanmean(np.power(fhat - ohat, 2))
    denom = np.nanmean(np.power(fhat,2) + np.power(ohat,2))
    ovest = (np.sum(fcst > threshold) - np.sum(obs >= threshold)) / fcst.size
    with np.errstate(divide='ignore', invalid='ignore'):
       fss_ret = 1.-num/denom
    return num, denom, fss_ret, ovest

def fourier_fss_eps(fcst, obs, threshold, window, percentiles, mode):
    """
    Compute the fractional skill score using convolution
    :paramfcst: nd-array, forecast field
    :paramobs: nd-array, observation field.
    :param window: integer, window size.
    :param percentiles: threshold list is treated as percentiles [0 ... 100]
    :return: tuple of FSS numerator, denominator and score.
    """
    # ny, nx = fcst.shape
    if mode=='valid' and any(np.array(window) > np.array(fcst.shape)):
      return np.nan, np.nan, np.nan, np.nan
    if percentiles:
      fhat = fourier_filter_eps(np.mean(fcst > np.percentile(fcst, threshold), axis=0), window, mode)
      ohat = fourier_filter_eps(obs > np.percentile(obs, threshold), window, mode)
    else:
      fhat = fourier_filter_eps(np.mean(fcst > threshold, axis=0), window, mode)
      ohat = fourier_filter_eps(obs > threshold, window, mode)
    num = np.nanmean(np.power(fhat - ohat, 2))
    denom = np.nanmean(np.power(fhat,2) + np.power(ohat,2))
    ovest = (np.sum(fcst > threshold) - np.sum(obs > threshold)) / fcst.size
    with np.errstate(divide='ignore', invalid='ignore'):
       fss_ret = 1.-num/denom
    return num, denom, fss_ret, ovest
    
def fss_frame(fcst, obs, windows, levels, percentiles=False, mode='same'):
    """
    Compute the fraction skill score data-frame.
    :paramfcst: nd-array, forecast field.
    :paramobs: nd-array, observation field.
    :param window: list, window sizes.
    :param levels: list, threshold levels.
    return: list, dataframes of the FSS: numerator, denominator and score.
    """
    num_data_fft, den_data_fft, fss_data_fft, overestimated = [], [], [], []
    
    for level in levels:
        _data_fft = [fourier_fss(fcst, obs, level, w, percentiles, mode) for w in windows]
        num_data_fft.append([x[0] for x in _data_fft])
        den_data_fft.append([x[1] for x in _data_fft])
        fss_data_fft.append([x[2] for x in _data_fft])
        overestimated.append([x[3] for x in _data_fft])
    col_windows = [w[0] for w in windows]
    return (pd.DataFrame(num_data_fft,  index=levels, columns=col_windows),
            pd.DataFrame(den_data_fft,  index=levels, columns=col_windows),
            pd.DataFrame(fss_data_fft,  index=levels, columns=col_windows),
            pd.DataFrame(overestimated, index=levels, columns=col_windows))

def fss_raw(fcst, obs, windows, levels, percentiles=False, mode='same'):
    """
    Compute the fraction skill score data-frame.
    :paramfcst: nd-array, forecast field.
    :paramobs: nd-array, observation field.
    :param window: list, window sizes.
    :param levels: list, threshold levels.
    return: list, dataframes of the FSS: numerator, denominator and score.
    """
    fss_data_fft = []
    
    for level in levels:
        _data_fft = [fourier_fss(fcst, obs, level, w, percentiles, mode) for w in windows]
        fss_data_fft.append([x[2] for x in _data_fft])
        
    return fss_data_fft

def fss_frame_eps(fcst, obs, windows, levels, percentiles=False, mode='same'):
    """
    Compute the fraction skill score data-frame.
    :paramfcst: nd-array, forecast field.
    :paramobs: nd-array, observation field.
    :param window: list, window sizes.
    :param levels: list, threshold levels.
    return: list, dataframes of the FSS: numerator, denominator and score.
    """
    num_data_fft, den_data_fft, fss_data_fft, overestimated = [], [], [], []
    
    for level in levels:
        _data_fft = [fourier_fss_eps(fcst, obs, level, w, percentiles, mode) for w in windows]
        num_data_fft.append([x[0] for x in _data_fft])
        den_data_fft.append([x[1] for x in _data_fft])
        fss_data_fft.append([x[2] for x in _data_fft])
        overestimated.append([x[3] for x in _data_fft])
    col_windows = [w[0] for w in windows]
    return (pd.DataFrame(num_data_fft,  index=levels, columns=col_windows),
            pd.DataFrame(den_data_fft,  index=levels, columns=col_windows),
            pd.DataFrame(fss_data_fft,  index=levels, columns=col_windows),
            pd.DataFrame(overestimated, index=levels, columns=col_windows))
