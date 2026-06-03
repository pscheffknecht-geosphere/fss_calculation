import numpy as np
import time
import pandas as pd

import logging
logger = logging.getLogger(__name__)


def compute_integral_table(field):
    if field.ndim == 2:
        return field.cumsum(1).cumsum(0)
    elif field.ndim == 3:
        return field.cumsum(2).cumsum(1)
    else:
        logger.critical(f"FSS calculation received a {field.ndim}D array, only 2D and 3D is supported! Aborting...")
        exit(1)

def integral_filter(field, n):
    """
    Fast summed area table version of the sliding accumulator.

    Uses np.pad with edge replication + contiguous slicing instead of
    mgrid/clip/fancy-indexing.  Boundary clamping behaviour is identical
    to the original implementation.

    :param field: nd-array of binary hits/misses (2D or 3D).
    :param n: window size.
    """
    w = n // 2
    if w < 1:
        return field

    rows, cols = field.shape[-2], field.shape[-1]
    D = 2 * w

    # Pad with edge values so clamped lookups become simple slicing
    if field.ndim == 2:
        p = np.pad(field, w, mode='edge')
        return (p[D:D+rows, D:D+cols] + p[:rows, :cols]
              - p[:rows, D:D+cols] - p[D:D+rows, :cols])
    else:  # 3D
        p = np.pad(field, ((0, 0), (w, w), (w, w)), mode='edge')
        return (p[:, D:D+rows, D:D+cols] + p[:, :rows, :cols]
              - p[:, :rows, D:D+cols] - p[:, D:D+rows, :cols])


def _fss_score(fhat, ohat, inv_area_sq):
    """Compute FSS num, denom, score using BLAS dot products (zero temp arrays)."""
    fflat = fhat.ravel()
    oflat = ohat.ravel()
    n = fflat.size

    ff = np.dot(fflat, fflat)
    oo = np.dot(oflat, oflat)
    fo = np.dot(fflat, oflat)

    num   = inv_area_sq * (ff + oo - 2.0 * fo) / n
    denom = inv_area_sq * (ff + oo) / n

    if denom == 0.0:
        return 0.0, 0.0, np.nan
    return num, denom, 1.0 - num / denom


def fss(fcst, obs, window, fcst_cache, obs_cache, threshold_mode="over"):
    """
    Compute the fraction skill score using summed area tables.
    :param fcst: nd-array, forecast field.
    :param obs: nd-array, observation field.
    :param window: integer, window size.
    :return: tuple of FSS numerator, denominator and score.
    """
    fhat = integral_filter(fcst_cache, window)
    ohat = integral_filter(obs_cache, window)

    w = window // 2
    inv_area_sq = 1.0 / (2.0 * w + 1.0) ** 4   # (inv_window_area)^2

    return _fss_score(fhat, ohat, inv_area_sq)


def _build_binary_sat(fcst, obs, t1, t2, t1o, t1f, percentiles,
                      threshold_mode, tolerance):
    """Build integral tables for the binarised forecast and observation fields."""
    if threshold_mode == "over":
        obs_bin = compute_integral_table((obs > t1o).astype(int))
        mod_bin = compute_integral_table((fcst > t1f).astype(int))
    elif threshold_mode == "under":
        obs_bin = compute_integral_table((obs <= t1o).astype(int))
        mod_bin = compute_integral_table((fcst <= t1f).astype(int))
    elif threshold_mode == "between":
        t2o = np.percentile(obs, t2) if percentiles else t2
        t2f = np.percentile(fcst, t2) if percentiles else t2
        obs_bin = compute_integral_table(((obs > t1o) & (obs <= t2o)).astype(int))
        mod_bin = compute_integral_table(((fcst > t1f) & (fcst <= t2f)).astype(int))
    elif threshold_mode == "tolerance":
        obs_bin = compute_integral_table(
            ((obs > (1.-tolerance) * t1o) & (obs <= (1.+tolerance)*t1o)).astype(int))
        mod_bin = compute_integral_table(
            ((fcst > (1.-tolerance) * t1f) & (fcst <= (1.+tolerance)*t1o)).astype(int))
    return mod_bin, obs_bin


def fss_threshold(fcst, obs, t1, t2, windows, percentiles=False, threshold_mode="over", tolerance=0.1):
    num_t = np.zeros(windows.shape)
    den_t = np.zeros(windows.shape)
    fss_t = np.zeros(windows.shape)

    t1o = np.percentile(obs, t1) if percentiles else t1
    t1f = np.percentile(fcst, t1) if percentiles else t1

    mod_bin, obs_bin = _build_binary_sat(
        fcst, obs, t1, t2, t1o, t1f, percentiles, threshold_mode, tolerance)

    for jj, window in enumerate(windows):
        num_t[jj], den_t[jj], fss_t[jj] = fss(fcst, obs, window,
                                                fcst_cache=mod_bin, obs_cache=obs_bin,
                                                threshold_mode=threshold_mode)
    ovest_val = (np.sum(fcst > t1) - np.sum(obs > t1)) / fcst.size
    ovest = np.full(windows.shape, ovest_val)
    return [num_t, den_t, fss_t, ovest]


def fss_threshold_eps(fcst, obs, t1, t2, windows, percentiles=False, threshold_mode="over", tolerance=0.1):
    assert fcst.ndim == 3, "eFSS calculation requires Forecast to be a 3D array, but it is {fcst.ndim}D with shape {fcst.shape}"
    num_t = np.zeros(windows.shape)
    den_t = np.zeros(windows.shape)
    fss_t = np.zeros(windows.shape)

    t1o = np.percentile(obs, t1) if percentiles else t1
    t1f = np.percentile(fcst, t1) if percentiles else t1
    if percentiles and t2:
        t2o = np.percentile(obs, t2) if percentiles else t2
        t2f = np.percentile(fcst, t2) if percentiles else t2

    if threshold_mode == "over":
        obs_bin = compute_integral_table((obs > t1o).astype(int))
        mod_bin = compute_integral_table(np.mean(fcst > t1f, axis=0))
    elif threshold_mode == "under":
        obs_bin = compute_integral_table((obs <= t1o).astype(int))
        mod_bin = compute_integral_table(np.mean(fcst <= t1f, axis=0))
    elif threshold_mode == "between":
        obs_bin = compute_integral_table(((obs > t1o) & (obs <= t2o)).astype(int))
        mod_bin = compute_integral_table(np.mean((fcst > t1f) & (fcst <= t2f), axis=0))
    elif threshold_mode == "tolerance":
        obs_bin = compute_integral_table(
            ((obs > (1.-tolerance) * t1o) & (obs <= (1.+tolerance)*t1o)).astype(int))
        mod_bin = compute_integral_table(
            np.mean((fcst > (1.-tolerance) * t1f) & (fcst <= (1.+tolerance)*t1o), axis=0))
    for jj, window in enumerate(windows):
        num_t[jj], den_t[jj], fss_t[jj] = fss(fcst, obs, window,
                                                fcst_cache=mod_bin, obs_cache=obs_bin,
                                                threshold_mode=threshold_mode)
    ovest_val = (np.sum(fcst > t1) - np.sum(obs > t1)) / fcst.size
    ovest = np.full(windows.shape, ovest_val)
    return [num_t, den_t, fss_t, ovest]


def fss_cumsum_parallel(fcst, obs, thresholds, windows, percentiles=False, threshold_mode="over", tolerance=0.1,
                       eps=False, n_jobs=1):
    if not isinstance(thresholds, np.ndarray):
        thresholds = np.array(thresholds)
    if not isinstance(windows, np.ndarray):
        windows = np.array(windows)
    use_fss_threshold_func = fss_threshold_eps if eps else fss_threshold

    if threshold_mode == "between":
        thresholds = np.insert(thresholds, 0, -1.) # insert -1 to retain zero values as OK
        calls = [(thresholds[ii], thresholds[ii+1]) for ii in range(thresholds.size-1)]
    elif threshold_mode in ("over", "under", "tolerance"):
        calls = [(t, None) for t in thresholds]

    if n_jobs == 1:
        ret = [use_fss_threshold_func(
            fcst, obs, t1, t2, windows, percentiles=percentiles,
            threshold_mode=threshold_mode, tolerance=tolerance) for t1, t2 in calls]
    else:
        from joblib import Parallel, delayed
        ret = Parallel(n_jobs=n_jobs)(
            delayed(use_fss_threshold_func)(
                fcst, obs, t1, t2, windows, percentiles=percentiles,
                threshold_mode=threshold_mode, tolerance=tolerance) for t1, t2 in calls)

    ret_arr = np.swapaxes(np.array(ret), 0, 1)
    return ret_arr

def fss_cumsum_frame(fcst, obs, windows, thresholds, percentiles=False, threshold_mode="over", tolerance=0.1,
                    mode=None, eps=False, raw=False):
    # adjust windows from legacy format:
    windows = [w[0] for w in windows]
    ret_arr = fss_cumsum_parallel(fcst, obs, thresholds, windows, percentiles=percentiles,
                                  threshold_mode="over", tolerance=tolerance, eps=eps)
    if raw:
        return ret_arr[2]
    else:
        return (pd.DataFrame(ret_arr[0], index=thresholds, columns=windows),
                pd.DataFrame(ret_arr[1], index=thresholds, columns=windows),
                pd.DataFrame(ret_arr[2], index=thresholds, columns=windows),
                pd.DataFrame(ret_arr[3], index=thresholds, columns=windows))


### Randomized thresholds and windows
# pseudo-random 2D values with good coverage and non-fixed sample count
def R2(N):
    g = 1.32471795724474602596
    a1 = 1.0 / g
    a2 = 1.0 / (g * g)
    x = (0.5 + a1 * N) %1
    y = (0.5 + a2 * N) %1
    return x, y

class CWFSS:
    def __init__(self, fcst, obs, nsamples=500, threshold_limits=(0.1, 100.), window_limits=(1, 601),
                threshold_max_weight=2., window_max_weight=2., threshold_limiting="relative"):
        self.wmin = int(window_limits[0])
        self.wmax = int(window_limits[1])
        if threshold_limiting == "relative":
            self.tmin = threshold_limits[0] * obs.max() / 100.
            self.tmax = threshold_limits[1] * obs.max() / 100.
        elif threshold_limiting == "absolute":
            self.tmin = threshold_limits[0]
            self.tmax = threshold_limits[1]
        elif threshold_limiting == "percentiles":
            self.tmin = np.percentile(obs, threshold_limits[0])
            self.tmax = np.percentile(obs, threshold_limits[1])
        self.nsamples = nsamples
        self.denominators = np.zeros(nsamples)
        self.numerators = np.zeros(nsamples)
        self.values = np.zeros(nsamples)
        self.windows = np.zeros(nsamples)
        self.thresholds = np.zeros(nsamples)
        self.__calc__(fcst, obs)
        self.__calc_cwfss()

    def __calc__(self, fcst, obs):
        for N in range(self.nsamples):
            x, y = R2(N)
            t = self.tmin + y * (self.tmax - self.tmin)
            w = int(self.wmin + x * (self.wmax - self.wmin))
            self.windows[N] = w
            self.thresholds[N] = t
            obs_bin = compute_integral_table((obs > t).astype(int))
            mod_bin = compute_integral_table((fcst > t).astype(int))
            ohat = integral_filter(obs_bin, w)
            fhat = integral_filter(mod_bin, w)
            fflat = fhat.ravel()
            oflat = ohat.ravel()
            n = fflat.size
            ff = np.dot(fflat, fflat)
            oo = np.dot(oflat, oflat)
            fo = np.dot(fflat, oflat)
            self.numerators[N] = (ff + oo - 2.0 * fo) / n
            self.denominators[N] = (ff + oo) / n
            with np.errstate(divide='ignore', invalid='ignore'):
                self.values[N] = 1. - self.numerators[N] / self.denominators[N]

    def __calc_cwfss(self):
        t_factor = (self.tmax + self.thresholds) / self.tmax
        w_factor = 2. * self.wmax / (self.wmax + self.windows)
        weighted_fss = (2. * (self.values - 0.5)).clip(0., 1.) * t_factor * w_factor
        weighted_fss_max = t_factor * w_factor
        self.cwfss = np.nanmean(weighted_fss) / np.nanmean(weighted_fss_max)

    def bootstrap(self, N=500):
        self.bootstrap_info = np.zeros(N)
        t_factor = (self.tmax + self.thresholds) / self.tmax
        w_factor = 2. * self.wmax / (self.wmax + self.windows)
        weighted_fss = (2. * (self.values - 0.5)).clip(0., 1.) * t_factor * w_factor
        weighted_fss_max = t_factor * w_factor
        for ii in range(N):
            idx = (self.nsamples * np.random.rand(self.nsamples)).astype(int)
            self.bootstrap_info[ii] = np.nanmean(weighted_fss[idx]) / np.nanmean(weighted_fss_max[idx])
