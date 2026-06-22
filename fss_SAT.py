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


def _fss_score_masked(Sf, So, C):
    """
    Missing-data FSS score with per-window valid-point weighting.

    Sf, So are box-sums of the *mask-zeroed* binary forecast/observation fields,
    C is the number of valid points in each window.  The windowed fractions are
    Sf/C and So/C, and each window centre is weighted by C.  With weight C the
    weighted means collapse to:

        num = sum((Sf - So)^2 / C) / sum(C)
        den = sum((Sf^2 + So^2) / C) / sum(C)

    C is defined as the full window area minus the number of missing points in
    the (boundary-clamped) window, so with no missing data C equals the constant
    window area everywhere and the score reduces exactly to the clean fast path.
    Windows with no valid points (C == 0) drop out naturally.
    """
    with np.errstate(divide='ignore'):
        inv = np.where(C > 0, 1.0 / C, 0.0)
    diff = Sf - So
    wsum = C.sum()

    if wsum == 0.0:
        return 0.0, 0.0, np.nan

    num   = np.sum(diff * diff * inv) / wsum
    denom = np.sum((Sf * Sf + So * So) * inv) / wsum

    if denom == 0.0:
        return num, denom, np.nan
    return num, denom, 1.0 - num / denom


def fss(fcst, obs, window, fcst_cache, obs_cache, threshold_mode="over", invalid_cache=None):
    """
    Compute the fraction skill score using summed area tables.
    :param fcst: nd-array, forecast field.
    :param obs: nd-array, observation field.
    :param window: integer, window size.
    :param invalid_cache: optional integral table of the *invalid* (missing-point)
        mask.  When given, the missing-data path is used (per-window valid-point
        weighting); the forecast/observation caches must then be the mask-zeroed
        binary SATs.
    :return: tuple of FSS numerator, denominator and score.
    """
    fhat = integral_filter(fcst_cache, window)
    ohat = integral_filter(obs_cache, window)

    w = window // 2

    if invalid_cache is not None:
        # Valid count = full window area minus the missing points it contains.
        area = (2.0 * w + 1.0) ** 2
        C = area - integral_filter(invalid_cache, window)
        return _fss_score_masked(fhat, ohat, C)

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
            ((fcst > (1.-tolerance) * t1f) & (fcst <= (1.+tolerance)*t1f)).astype(int))
    return mod_bin, obs_bin


def _validity_mask(fcst, obs):
    """Boolean mask of points valid in *both* fields (a point is missing if
    either forecast or observation is NaN there)."""
    return ~(np.isnan(fcst) | np.isnan(obs))


def _invalid_sat(mask):
    """Integral table of the *invalid* (missing-point) mask, threshold-independent.

    Used to subtract missing points from the full window area, so the per-window
    valid count matches the clean path's constant area where nothing is missing.
    """
    return compute_integral_table((~mask).astype(float))


def _build_binary_sat_masked(fcst, obs, t1, t2, t1o, t1f, percentiles,
                             threshold_mode, tolerance, mask):
    """Like _build_binary_sat, but zeros out missing points (via `mask`) so they
    do not contribute to the windowed sums, and binarises to float."""
    with np.errstate(invalid='ignore'):  # NaN comparisons are intentional -> False
        if threshold_mode == "over":
            obs_b = (obs > t1o) & mask
            mod_b = (fcst > t1f) & mask
        elif threshold_mode == "under":
            obs_b = (obs <= t1o) & mask
            mod_b = (fcst <= t1f) & mask
        elif threshold_mode == "between":
            t2o = np.nanpercentile(obs, t2) if percentiles else t2
            t2f = np.nanpercentile(fcst, t2) if percentiles else t2
            obs_b = (obs > t1o) & (obs <= t2o) & mask
            mod_b = (fcst > t1f) & (fcst <= t2f) & mask
        elif threshold_mode == "tolerance":
            obs_b = (obs > (1.-tolerance) * t1o) & (obs <= (1.+tolerance)*t1o) & mask
            mod_b = (fcst > (1.-tolerance) * t1f) & (fcst <= (1.+tolerance)*t1f) & mask
    obs_bin = compute_integral_table(obs_b.astype(float))
    mod_bin = compute_integral_table(mod_b.astype(float))
    return mod_bin, obs_bin


def fss_threshold(fcst, obs, t1, t2, windows, percentiles=False, threshold_mode="over", tolerance=0.1):
    num_t = np.zeros(windows.shape)
    den_t = np.zeros(windows.shape)
    fss_t = np.zeros(windows.shape)

    mask = _validity_mask(fcst, obs)

    if mask.all():
        # ── clean fast path (unchanged) ──
        t1o = np.percentile(obs, t1) if percentiles else t1
        t1f = np.percentile(fcst, t1) if percentiles else t1

        mod_bin, obs_bin = _build_binary_sat(
            fcst, obs, t1, t2, t1o, t1f, percentiles, threshold_mode, tolerance)

        for jj, window in enumerate(windows):
            num_t[jj], den_t[jj], fss_t[jj] = fss(fcst, obs, window,
                                                    fcst_cache=mod_bin, obs_cache=obs_bin,
                                                    threshold_mode=threshold_mode)
        ovest_val = (np.sum(fcst > t1) - np.sum(obs > t1)) / fcst.size
    else:
        # ── missing-data path (per-window valid-point weighting) ──
        t1o = np.nanpercentile(obs, t1) if percentiles else t1
        t1f = np.nanpercentile(fcst, t1) if percentiles else t1

        mod_bin, obs_bin = _build_binary_sat_masked(
            fcst, obs, t1, t2, t1o, t1f, percentiles, threshold_mode, tolerance, mask)
        invalid_sat = _invalid_sat(mask)

        for jj, window in enumerate(windows):
            num_t[jj], den_t[jj], fss_t[jj] = fss(fcst, obs, window,
                                                    fcst_cache=mod_bin, obs_cache=obs_bin,
                                                    threshold_mode=threshold_mode,
                                                    invalid_cache=invalid_sat)
        nvalid = mask.sum()
        with np.errstate(invalid='ignore'):
            ovest_val = (np.sum((fcst > t1f) & mask) - np.sum((obs > t1o) & mask)) / nvalid

    ovest = np.full(windows.shape, ovest_val)
    return [num_t, den_t, fss_t, ovest]


def _eps_prob_masked(fcst, t1f, t2, t2f, threshold_mode, tolerance, member_valid, inv_n):
    """Per-point ensemble exceedance probability over the *valid* members.

    `member_valid` is the 3D ~isnan mask, `inv_n` the per-point reciprocal of the
    valid-member count (0 where no member is valid).  NaN members never count as
    exceeding (NaN comparisons are False) and are excluded from the denominator.
    """
    with np.errstate(invalid='ignore'):  # NaN comparisons are intentional -> False
        if threshold_mode == "over":
            exceed = (fcst > t1f) & member_valid
        elif threshold_mode == "under":
            exceed = (fcst <= t1f) & member_valid
        elif threshold_mode == "between":
            exceed = (fcst > t1f) & (fcst <= t2f) & member_valid
        elif threshold_mode == "tolerance":
            exceed = (fcst > (1.-tolerance) * t1f) & (fcst <= (1.+tolerance)*t1f) & member_valid
    return exceed.sum(axis=0) * inv_n


def fss_threshold_eps(fcst, obs, t1, t2, windows, percentiles=False, threshold_mode="over", tolerance=0.1):
    assert fcst.ndim == 3, "eFSS calculation requires Forecast to be a 3D array, but it is {fcst.ndim}D with shape {fcst.shape}"
    num_t = np.zeros(windows.shape)
    den_t = np.zeros(windows.shape)
    fss_t = np.zeros(windows.shape)

    member_valid = ~np.isnan(fcst)            # 3D, valid forecast members
    obs_valid = ~np.isnan(obs)                # 2D

    if member_valid.all() and obs_valid.all():
        # ── clean fast path (unchanged) ──
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
                np.mean((fcst > (1.-tolerance) * t1f) & (fcst <= (1.+tolerance)*t1f), axis=0))

        for jj, window in enumerate(windows):
            num_t[jj], den_t[jj], fss_t[jj] = fss(fcst, obs, window,
                                                    fcst_cache=mod_bin, obs_cache=obs_bin,
                                                    threshold_mode=threshold_mode)
        ovest_val = (np.sum(fcst > t1) - np.sum(obs > t1)) / fcst.size
    else:
        # ── missing-data path (per-window valid-point weighting) ──
        # A grid point counts when obs is valid AND at least one member is valid.
        mask = obs_valid & member_valid.any(axis=0)
        n_valid = member_valid.sum(axis=0)
        with np.errstate(divide='ignore'):
            inv_n = np.where(n_valid > 0, 1.0 / n_valid, 0.0)

        t1o = np.nanpercentile(obs, t1) if percentiles else t1
        t1f = np.nanpercentile(fcst, t1) if percentiles else t1
        t2o = (np.nanpercentile(obs, t2) if percentiles else t2) if threshold_mode == "between" else None
        t2f = (np.nanpercentile(fcst, t2) if percentiles else t2) if threshold_mode == "between" else None

        p_f = _eps_prob_masked(fcst, t1f, t2, t2f, threshold_mode, tolerance, member_valid, inv_n)
        p_f = np.where(mask, p_f, 0.0)        # zero out missing points

        with np.errstate(invalid='ignore'):
            if threshold_mode == "over":
                obs_b = (obs > t1o) & mask
            elif threshold_mode == "under":
                obs_b = (obs <= t1o) & mask
            elif threshold_mode == "between":
                obs_b = (obs > t1o) & (obs <= t2o) & mask
            elif threshold_mode == "tolerance":
                obs_b = (obs > (1.-tolerance) * t1o) & (obs <= (1.+tolerance)*t1o) & mask

        mod_bin = compute_integral_table(p_f)
        obs_bin = compute_integral_table(obs_b.astype(float))
        invalid_sat = _invalid_sat(mask)

        for jj, window in enumerate(windows):
            num_t[jj], den_t[jj], fss_t[jj] = fss(fcst, obs, window,
                                                    fcst_cache=mod_bin, obs_cache=obs_bin,
                                                    threshold_mode=threshold_mode,
                                                    invalid_cache=invalid_sat)
        ovest_val = (p_f.sum() - obs_b.sum()) / mask.sum()

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
                                  threshold_mode=threshold_mode, tolerance=tolerance, eps=eps)
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
                threshold_max_weight=2., window_max_weight=2., threshold_limiting="relative",
                threshold_mode="over", tolerance=0.1):
        self.wmin = int(window_limits[0])
        self.wmax = int(window_limits[1])
        # NaN-aware: obs (e.g. OPERA) may carry NaN where there is no coverage
        if threshold_limiting == "relative":
            obs_max = np.nanmax(obs)
            self.tmin = threshold_limits[0] * obs_max / 100.
            self.tmax = threshold_limits[1] * obs_max / 100.
        elif threshold_limiting == "absolute":
            self.tmin = threshold_limits[0]
            self.tmax = threshold_limits[1]
        elif threshold_limiting == "percentiles":
            self.tmin = np.nanpercentile(obs, threshold_limits[0])
            self.tmax = np.nanpercentile(obs, threshold_limits[1])
        self.nsamples = nsamples
        self.threshold_mode = threshold_mode
        self.tolerance = tolerance
        self.denominators = np.zeros(nsamples)
        self.numerators = np.zeros(nsamples)
        self.values = np.zeros(nsamples)
        self.windows = np.zeros(nsamples)
        self.thresholds = np.zeros(nsamples)
        self.__calc__(fcst, obs)
        self.__calc_cwfss()

    def _binarise(self, field, t):
        """Boolean exceedance mask at threshold t using the configured threshold_mode.

        NaN comparisons evaluate to False, so missing points never count as
        exceedances; the missing-data path additionally ANDs this with the
        validity mask.
        """
        if self.threshold_mode == "over":
            return field > t
        elif self.threshold_mode == "under":
            return field <= t
        elif self.threshold_mode == "tolerance":
            return (field > (1. - self.tolerance) * t) & (field <= (1. + self.tolerance) * t)
        else:
            return field > t

    def __calc__(self, fcst, obs):
        # points valid in both fields; missing data path uses per-window weighting
        mask = _validity_mask(fcst, obs)
        clean = mask.all()
        if not clean:
            invalid_sat = _invalid_sat(mask)
        for N in range(self.nsamples):
            x, y = R2(N)
            t = self.tmin + y * (self.tmax - self.tmin)
            w = int(self.wmin + x * (self.wmax - self.wmin))
            self.windows[N] = w
            self.thresholds[N] = t
            if clean:
                obs_bin = compute_integral_table(self._binarise(obs, t).astype(int))
                mod_bin = compute_integral_table(self._binarise(fcst, t).astype(int))
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
            else:
                with np.errstate(invalid='ignore'):  # NaN comparisons -> False
                    obs_bin = compute_integral_table((self._binarise(obs, t) & mask).astype(float))
                    mod_bin = compute_integral_table((self._binarise(fcst, t) & mask).astype(float))
                Sf = integral_filter(mod_bin, w)
                So = integral_filter(obs_bin, w)
                ww = w // 2
                area = (2.0 * ww + 1.0) ** 2
                C = area - integral_filter(invalid_sat, w)
                self.numerators[N], self.denominators[N], self.values[N] = \
                    _fss_score_masked(Sf, So, C)

    def __calc_cwfss(self):
        t_factor = (self.tmax + self.thresholds) / self.tmax
        w_factor = 2. * self.wmax / (self.wmax + self.windows)
        weighted_fss = (2. * (self.values - 0.5)).clip(0., 1.) * t_factor * w_factor
        # Samples with an undefined FSS (NaN -- e.g. no valid exceedance in the
        # window) drop out of nanmean(weighted_fss); the theoretical maximum must
        # carry the same mask so the achievable ceiling stays 1 and is not
        # lowered merely by the presence of missing data.
        weighted_fss_max = np.where(np.isnan(weighted_fss), np.nan, t_factor * w_factor)
        self.cwfss = np.nanmean(weighted_fss) / np.nanmean(weighted_fss_max)

    def bootstrap(self, N=500):
        self.bootstrap_info = np.zeros(N)
        t_factor = (self.tmax + self.thresholds) / self.tmax
        w_factor = 2. * self.wmax / (self.wmax + self.windows)
        weighted_fss = (2. * (self.values - 0.5)).clip(0., 1.) * t_factor * w_factor
        weighted_fss_max = np.where(np.isnan(weighted_fss), np.nan, t_factor * w_factor)
        for ii in range(N):
            idx = (self.nsamples * np.random.rand(self.nsamples)).astype(int)
            self.bootstrap_info[ii] = np.nanmean(weighted_fss[idx]) / np.nanmean(weighted_fss_max[idx])
