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
        

def _validity_mask(fcst, obs):
    """Boolean mask of points valid in *both* fields (a point is missing if
    either forecast or observation is NaN there)."""
    return ~(np.isnan(fcst) | np.isnan(obs))


def _fss_score_masked(Sf, So, C):
    """Missing-data FSS score with per-window valid-point weighting.

    Mirrors fss_SAT._fss_score_masked: Sf, So are box-sums of the mask-zeroed
    binary fields, C the number of valid points in each window.  Each window is
    weighted by C, so

        num = sum((Sf - So)^2 / C) / sum(C)
        den = sum((Sf^2 + So^2) / C) / sum(C)

    and the sum(C) normaliser cancels in the FSS ratio.  C is defined as the full
    window area minus the missing points it contains, so with no missing data C
    equals the constant window area everywhere and the score reduces exactly to
    the clean path.  A small (>0.5) threshold guards against FFT round-off noise
    making an empty window's count a tiny non-zero value (counts are integers in
    exact arithmetic).
    """
    valid = C > 0.5
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = np.where(valid, 1.0 / C, 0.0)
    diff = Sf - So
    wsum = np.sum(np.where(valid, C, 0.0))
    if wsum == 0.0:
        return 0.0, 0.0, np.nan
    num   = np.sum(diff * diff * inv) / wsum
    denom = np.sum((Sf * Sf + So * So) * inv) / wsum
    if denom == 0.0:
        return num, denom, np.nan
    return num, denom, 1.0 - num / denom


def _fourier_fss_masked(fcst, obs, threshold, window, percentiles, mode):
    """Missing-data FSS via convolution box-sums (always takes the masked path).

    A point is missing where obs or fcst is NaN.  The binary exceedance fields are
    zeroed at missing points before convolving (a raw NaN would otherwise spread
    across the whole FFT output), and the per-window valid count is obtained from
    one more convolution of the validity mask, giving the same weight = valid-count
    semantics as the SAT method.  Boundary handling is FFT's zero-padding, so this
    agrees with the SAT masked path only in the interior.
    """
    mask = _validity_mask(fcst, obs)
    with np.errstate(invalid='ignore'):  # NaN comparisons are intentional -> False
        if percentiles:
            t1f = np.nanpercentile(fcst, threshold)
            t1o = np.nanpercentile(obs, threshold)
            fbin = (fcst >= t1f) & mask
            obin = (obs >= t1o) & mask
        else:
            fbin = (fcst > threshold) & mask
            obin = (obs > threshold) & mask

    Sf = fourier_filter(fbin.astype(float), window, mode)
    So = fourier_filter(obin.astype(float), window, mode)
    area = float(np.prod(window))
    C = area - fourier_filter((~mask).astype(float), window, mode)

    num, denom, fss_ret = _fss_score_masked(Sf, So, C)

    nvalid = mask.sum()
    ovest = (np.sum(fbin) - np.sum(obin)) / nvalid if nvalid else np.nan
    return num, denom, fss_ret, ovest


def fourier_fss(fcst, obs, threshold, window, percentiles, mode):
    """
    Compute the fractional skill score using convolution
    :paramfcst: nd-array, forecast field
    :paramobs: nd-array, observation field.
    :param window: integer, window size.
    :param percentiles: threshold list is treated as percentiles [0 ... 100]
    :return: tuple of FSS numerator, denominator and score.

    Missing data (NaN in obs or fcst) is handled transparently: when any NaN is
    present the computation switches to a mask-aware path that excludes missing
    points from each window's sums and weights every window by its number of
    valid points (see ``_fourier_fss_masked``).  Clean fields take the original
    fast path unchanged.
    """
    ny, nx = fcst.shape
    if mode=='valid' and any(np.array(window) > np.array(fcst.shape)):
        return np.nan, np.nan, np.nan, np.nan

    if not _validity_mask(fcst, obs).all():
        return _fourier_fss_masked(fcst, obs, threshold, window, percentiles, mode)

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

def _fourier_fss_eps_masked(fcst, obs, threshold, window, percentiles, mode):
    """Missing-data ensemble FSS via convolution box-sums (always masked path).

    Mirrors fss_SAT.fss_threshold_eps's missing-data branch.  A grid point counts
    when ``obs`` is valid AND at least one forecast member is valid there; the
    exceedance probability is averaged over the valid members only (NaN members
    never count as exceeding and are dropped from the denominator).  Missing points
    are zeroed before convolving and each window is weighted by its valid-point
    count (see ``_fss_score_masked``).  Boundary handling is FFT's zero-padding, so
    this agrees with the SAT ensemble masked path only in the interior.
    """
    member_valid = ~np.isnan(fcst)            # 3D, valid forecast members
    obs_valid = ~np.isnan(obs)                # 2D
    mask = obs_valid & member_valid.any(axis=0)
    n_valid = member_valid.sum(axis=0)
    with np.errstate(divide='ignore'):
        inv_n = np.where(n_valid > 0, 1.0 / n_valid, 0.0)

    with np.errstate(invalid='ignore'):  # NaN comparisons are intentional -> False
        if percentiles:
            t1f = np.nanpercentile(fcst, threshold)
            t1o = np.nanpercentile(obs, threshold)
            exceed = (fcst >= t1f) & member_valid
            obin = (obs >= t1o) & mask
        else:
            exceed = (fcst > threshold) & member_valid
            obin = (obs > threshold) & mask
    p_f = exceed.sum(axis=0) * inv_n
    p_f = np.where(mask, p_f, 0.0)            # zero out missing points

    Sf = fourier_filter_eps(p_f, window, mode)
    So = fourier_filter_eps(obin.astype(float), window, mode)
    area = float(np.prod(window))
    C = area - fourier_filter_eps((~mask).astype(float), window, mode)

    num, denom, fss_ret = _fss_score_masked(Sf, So, C)

    nvalid = mask.sum()
    ovest = (p_f.sum() - obin.sum()) / nvalid if nvalid else np.nan
    return num, denom, fss_ret, ovest


def fourier_fss_eps(fcst, obs, threshold, window, percentiles, mode):
    """
    Compute the fractional skill score using convolution
    :paramfcst: nd-array, forecast field
    :paramobs: nd-array, observation field.
    :param window: integer, window size.
    :param percentiles: threshold list is treated as percentiles [0 ... 100]
    :return: tuple of FSS numerator, denominator and score.

    Missing data (NaN in obs or any forecast member) is handled transparently: when
    any NaN is present the computation switches to a mask-aware path that excludes
    missing points from each window's sums and weights every window by its number of
    valid points (see ``_fourier_fss_eps_masked``).  Clean fields take the original
    fast path unchanged.
    """
    # ny, nx = fcst.shape
    if mode=='valid' and any(np.array(window) > np.array(fcst.shape)):
      return np.nan, np.nan, np.nan, np.nan

    if np.isnan(fcst).any() or np.isnan(obs).any():
        return _fourier_fss_eps_masked(fcst, obs, threshold, window, percentiles, mode)

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
