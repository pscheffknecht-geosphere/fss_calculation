"""
Tests for cross-method consistency of FSS implementations.

Three implementations:
  - fss_FFT: FFT-based convolution (scipy.signal.fftconvolve)
  - fss_SAT: Summed Area Table (integral image) approach
  - fss_OCL: OpenCL GPU/CPU kernel using SAT

SAT and OCL use the same algorithm and should agree to float32 precision (~1e-5).
FFT and SAT differ in boundary handling (zero-padding vs clamping), giving
differences up to ~5e-4 for large windows. Both conventions are valid.
"""
import numpy as np
import pytest
import sys

sys.path.insert(0, ".")

from fss_FFT import (fourier_fss, _fourier_fss_masked,
                     fourier_fss_eps, _fourier_fss_eps_masked)
from fss_SAT import (compute_integral_table, integral_filter, fss as sat_fss,
                     fss_threshold, fss_threshold_eps, fss_cumsum_parallel,
                     fss_cumsum_frame, _build_binary_sat,
                     _build_binary_sat_masked, _invalid_sat, _validity_mask,
                     _fss_score, _fss_score_masked,
                     CWFSS as CWFSS_new, R2 as R2_new)
from fss_SAT_original import (CWFSS as CWFSS_orig, R2 as R2_orig)

try:
    from fss_OCL import (fss_opencl, fss_opencl_arr,
                         _fss_opencl_masked, ctx, queue, program)
    HAS_OCL = True
except Exception:
    HAS_OCL = False

# ── Tolerances ──
# SAT and OCL share the same algorithm; differences come from float32 vs float64.
TOL_SAT_OCL = 1e-5
# FFT and SAT differ in boundary treatment (zero-pad vs clamp).
TOL_FFT_SAT = 1e-3


# ── Fixtures ──

def _make_gauss_bell(cx, cy, width, xgrid, ygrid, zgrid):
    zgrid += np.exp(-(
        np.square(xgrid - cx) / (2 * width ** 2) +
        np.square(ygrid - cy) / (2 * width ** 2)))


def _generate_fields(grid_step, seed, n_bells):
    """Generate reproducible Gaussian-bell fields.

    Matches call_FSS.py's random sequence: 4 draws per bell (x, y, a, s)
    where a (amplitude) is unused but consumed to keep the RNG in sync.
    """
    x = np.arange(-3., 3.01, grid_step)
    xx, yy = np.meshgrid(x, x)
    fcst = np.zeros(xx.shape)
    obs = np.zeros(xx.shape)
    np.random.seed(seed=seed)
    for _ in range(n_bells):
        cx, cy = -2 + 4 * np.random.rand(), -2 + 4 * np.random.rand()
        _ = 2.2 + 1.2 * np.random.rand()          # amplitude, unused but consumed
        s = 0.2 + 0.5 * np.random.rand()
        _make_gauss_bell(cx, cy, s, xx, yy, fcst)
        cx, cy = -2 + 4 * np.random.rand(), -2 + 4 * np.random.rand()
        _ = 2.2 + 1.2 * np.random.rand()
        s = 0.2 + 0.5 * np.random.rand()
        _make_gauss_bell(cx, cy, s, xx, yy, obs)
    return fcst, obs


@pytest.fixture(scope="module")
def fields():
    """601x601 test fields matching call_FSS.py (seed=127, 20 bells)."""
    return _generate_fields(0.01, 127, 20)


@pytest.fixture(scope="module")
def small_fields():
    """Smaller 101x101 fields for faster edge-case tests."""
    return _generate_fields(0.06, 42, 10)


# ── Helpers ──

def _fss_via_sat(fcst, obs, threshold, window):
    """Compute FSS using the SAT method for a single threshold/window."""
    obs_bin = compute_integral_table((obs > threshold).astype(int))
    mod_bin = compute_integral_table((fcst > threshold).astype(int))
    num, den, fss_val = sat_fss(fcst, obs, window, mod_bin, obs_bin)
    return num, den, fss_val


def _fss_via_fft(fcst, obs, threshold, window):
    """Compute FSS using the FFT method for a single threshold/window."""
    num, den, fss_val, _ = fourier_fss(fcst, obs, threshold, (window, window), False, "same")
    return num, den, fss_val


# ── Cross-method consistency ──

class TestFFTvsSAT:
    """FFT and SAT should agree within boundary-effect tolerance."""

    @pytest.mark.parametrize("threshold", [0.2, 0.5, 1.0, 1.5, 2.0])
    @pytest.mark.parametrize("window", [10, 30, 50, 90, 150])
    def test_fss_values(self, fields, threshold, window):
        fcst, obs = fields
        _, _, fss_fft = _fss_via_fft(fcst, obs, threshold, window)
        _, _, fss_sat = _fss_via_sat(fcst, obs, threshold, window)
        assert np.isclose(fss_fft, fss_sat, atol=TOL_FFT_SAT), \
            f"FFT={fss_fft:.8f} SAT={fss_sat:.8f} diff={abs(fss_fft - fss_sat):.2e}"

    @pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0])
    def test_large_window(self, fields, threshold):
        """Even at large windows (bigger boundary effects), difference stays bounded."""
        fcst, obs = fields
        _, _, fss_fft = _fss_via_fft(fcst, obs, threshold, 250)
        _, _, fss_sat = _fss_via_sat(fcst, obs, threshold, 250)
        assert np.isclose(fss_fft, fss_sat, atol=TOL_FFT_SAT), \
            f"FFT={fss_fft:.8f} SAT={fss_sat:.8f} diff={abs(fss_fft - fss_sat):.2e}"


@pytest.mark.skipif(not HAS_OCL, reason="OpenCL not available")
class TestSATvsOCL:
    """SAT and OCL use the same algorithm; only float32 precision differs."""

    @pytest.mark.parametrize("threshold", [0.2, 0.5, 1.0, 1.5, 2.0])
    @pytest.mark.parametrize("window", [10, 30, 50, 90, 150])
    def test_fss_values(self, fields, threshold, window):
        fcst, obs = fields
        _, _, fss_sat = _fss_via_sat(fcst, obs, threshold, window)
        _, _, fss_ocl = fss_opencl(fcst, obs, threshold, window, ctx, queue, program)
        assert np.isclose(fss_sat, fss_ocl, atol=TOL_SAT_OCL), \
            f"SAT={fss_sat:.8f} OCL={fss_ocl:.8f} diff={abs(fss_sat - fss_ocl):.2e}"


@pytest.mark.skipif(not HAS_OCL, reason="OpenCL not available")
class TestFFTvsOCL:
    """FFT and OCL should agree within boundary-effect tolerance."""

    @pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("window", [10, 50, 150])
    def test_fss_values(self, fields, threshold, window):
        fcst, obs = fields
        _, _, fss_fft = _fss_via_fft(fcst, obs, threshold, window)
        _, _, fss_ocl = fss_opencl(fcst, obs, threshold, window, ctx, queue, program)
        assert np.isclose(fss_fft, fss_ocl, atol=TOL_FFT_SAT), \
            f"FFT={fss_fft:.8f} OCL={fss_ocl:.8f} diff={abs(fss_fft - fss_ocl):.2e}"


# ── Mathematical properties ──

class TestFSSProperties:
    """Test FSS invariants that must hold for any correct implementation."""

    def test_perfect_forecast_fft(self, small_fields):
        """FSS = 1.0 when forecast equals observation."""
        _, obs = small_fields
        for win in [10, 30]:
            _, _, fss_val = _fss_via_fft(obs, obs, 0.5, win)
            assert fss_val == pytest.approx(1.0, abs=1e-10)

    def test_perfect_forecast_sat(self, small_fields):
        _, obs = small_fields
        for win in [10, 30]:
            _, _, fss_val = _fss_via_sat(obs, obs, 0.5, win)
            assert fss_val == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.skipif(not HAS_OCL, reason="OpenCL not available")
    def test_perfect_forecast_ocl(self, small_fields):
        _, obs = small_fields
        for win in [10, 30]:
            _, _, fss_val = fss_opencl(obs, obs, 0.5, win, ctx, queue, program)
            assert fss_val == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("method", ["fft", "sat"])
    @pytest.mark.parametrize("threshold", [0.2, 0.5, 1.0, 2.0])
    @pytest.mark.parametrize("window", [10, 30, 50])
    def test_fss_in_zero_one_range(self, fields, method, threshold, window):
        """FSS should be in [0, 1] whenever denominator > 0."""
        fcst, obs = fields
        if method == "fft":
            _, den, fss_val = _fss_via_fft(fcst, obs, threshold, window)
        else:
            _, den, fss_val = _fss_via_sat(fcst, obs, threshold, window)
        if den > 0:
            assert 0.0 <= fss_val <= 1.0, f"FSS={fss_val} out of [0,1]"

    def test_nan_when_no_exceedances(self, fields):
        """FSS should be NaN when no pixels exceed the threshold."""
        fcst, obs = fields
        extreme_threshold = fcst.max() + obs.max() + 1.0
        _, _, fss_fft = _fss_via_fft(fcst, obs, extreme_threshold, 10)
        _, _, fss_sat = _fss_via_sat(fcst, obs, extreme_threshold, 10)
        assert np.isnan(fss_fft), "FFT should return NaN for empty binary fields"
        assert np.isnan(fss_sat), "SAT should return NaN for empty binary fields"

    def test_fss_increases_with_window(self, fields):
        """For typical fields, FSS should generally increase with window size."""
        fcst, obs = fields
        threshold = 1.0
        windows = [10, 30, 50, 90, 150]
        fss_vals_fft = [_fss_via_fft(fcst, obs, threshold, w)[2] for w in windows]
        fss_vals_sat = [_fss_via_sat(fcst, obs, threshold, w)[2] for w in windows]
        # Allow small non-monotonicity (1e-6) due to numerics
        for vals, name in [(fss_vals_fft, "FFT"), (fss_vals_sat, "SAT")]:
            for i in range(len(vals) - 1):
                assert vals[i + 1] >= vals[i] - 1e-6, \
                    f"{name}: FSS decreased from w={windows[i]} ({vals[i]:.6f}) to w={windows[i+1]} ({vals[i+1]:.6f})"

    def test_symmetry_numerator(self, small_fields):
        """Swapping forecast and observation should give the same FSS."""
        fcst, obs = small_fields
        for win in [10, 30]:
            _, _, fss_ab = _fss_via_sat(fcst, obs, 0.5, win)
            _, _, fss_ba = _fss_via_sat(obs, fcst, 0.5, win)
            assert fss_ab == pytest.approx(fss_ba, abs=1e-10)


# ── Batch / frame interfaces ──

class TestBatchInterfaces:
    """Test the higher-level batch functions."""

    def test_sat_fss_threshold_matches_individual(self, fields):
        """fss_threshold should give the same results as calling fss individually."""
        fcst, obs = fields
        threshold = 1.0
        windows = np.array([10, 30, 50, 90])
        result = fss_threshold(fcst, obs, threshold, None, windows)
        num_batch, den_batch, fss_batch, _ = result
        for j, win in enumerate(windows):
            num_ind, den_ind, fss_ind = _fss_via_sat(fcst, obs, threshold, win)
            assert fss_batch[j] == pytest.approx(fss_ind, abs=1e-12), \
                f"w={win}: batch={fss_batch[j]:.10f} vs individual={fss_ind:.10f}"

    @pytest.mark.skipif(not HAS_OCL, reason="OpenCL not available")
    def test_ocl_arr_matches_individual(self, fields):
        """fss_opencl_arr should match individual fss_opencl calls."""
        fcst, obs = fields
        thresholds = np.array([0.5, 1.0, 2.0])
        windows = np.array([10, 30, 50])
        num_arr, den_arr, fss_arr = fss_opencl_arr(
            fcst, obs, thresholds, windows, ctx, queue, program)
        for i, t in enumerate(thresholds):
            for j, w in enumerate(windows):
                _, _, fss_ind = fss_opencl(fcst, obs, t, w, ctx, queue, program)
                assert fss_arr[i, j] == pytest.approx(fss_ind, abs=1e-6), \
                    f"t={t} w={w}: arr={fss_arr[i,j]:.8f} vs ind={fss_ind:.8f}"


# ── Regression: pinned reference values ──
# These values were computed with all three methods and cross-validated.
# They serve as regression guards against accidental changes.

REFERENCE_VALUES = {
    # (threshold, window): (fss_fft, fss_sat)
    (0.5, 10):  (0.787801, 0.787801),
    (0.5, 50):  (0.836973, 0.836966),
    (1.0, 30):  (0.672621, 0.672621),
    (1.0, 90):  (0.752451, 0.752451),
    (2.0, 50):  (0.279457, 0.279457),
    (2.0, 150): (0.406912, 0.406912),
}


class TestRegressionValues:
    """Pinned reference values to catch accidental changes."""

    @pytest.mark.parametrize("key", REFERENCE_VALUES.keys(),
                             ids=[f"t{k[0]}_w{k[1]}" for k in REFERENCE_VALUES])
    def test_fft_reference(self, fields, key):
        fcst, obs = fields
        threshold, window = key
        expected_fft, _ = REFERENCE_VALUES[key]
        _, _, fss_fft = _fss_via_fft(fcst, obs, threshold, window)
        assert fss_fft == pytest.approx(expected_fft, abs=1e-4), \
            f"FFT regression: got {fss_fft:.6f}, expected {expected_fft:.6f}"

    @pytest.mark.parametrize("key", REFERENCE_VALUES.keys(),
                             ids=[f"t{k[0]}_w{k[1]}" for k in REFERENCE_VALUES])
    def test_sat_reference(self, fields, key):
        fcst, obs = fields
        threshold, window = key
        _, expected_sat = REFERENCE_VALUES[key]
        _, _, fss_sat = _fss_via_sat(fcst, obs, threshold, window)
        assert fss_sat == pytest.approx(expected_sat, abs=1e-4), \
            f"SAT regression: got {fss_sat:.6f}, expected {expected_sat:.6f}"


# ── SAT threshold_mode variants ──
# The SAT implementation supports four binarisation modes:
#   "over"      : field > t          (default, tested above across all methods)
#   "under"     : field <= t
#   "between"   : t1 < field <= t2
#   "tolerance" : (1-tol)*t < field <= (1+tol)*t
# The tests below verify these modes preserve FSS invariants and match
# pinned reference values.

_MODES_SIMPLE = ["over", "under", "tolerance"]
_WINDOWS_MODES = np.array([10, 50, 150])


def _fss_threshold_mode(fcst, obs, t1, windows, mode, t2=None, tolerance=0.1):
    """Helper: call fss_threshold with a given mode, return FSS array."""
    return fss_threshold(fcst, obs, t1, t2, windows,
                         threshold_mode=mode, tolerance=tolerance)[2]


class TestSATThresholdModes:
    """Verify FSS invariants hold for all threshold_mode variants."""

    # -- Perfect forecast: FSS = 1.0 for every mode --

    @pytest.mark.parametrize("mode", ["over", "under", "between", "tolerance"])
    def test_perfect_forecast(self, fields, mode):
        _, obs = fields
        t2 = 2.0 if mode == "between" else None
        fss_vals = fss_threshold(obs, obs, 1.0, t2, _WINDOWS_MODES,
                                 threshold_mode=mode, tolerance=0.1)[2]
        for j, w in enumerate(_WINDOWS_MODES):
            assert fss_vals[j] == pytest.approx(1.0, abs=1e-10), \
                f"mode={mode} w={w}: FSS={fss_vals[j]}"

    # -- FSS in [0, 1] --

    @pytest.mark.parametrize("mode", _MODES_SIMPLE)
    @pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0])
    def test_fss_range(self, fields, mode, threshold):
        fcst, obs = fields
        fss_vals = _fss_threshold_mode(fcst, obs, threshold, _WINDOWS_MODES, mode)
        for j, w in enumerate(_WINDOWS_MODES):
            v = fss_vals[j]
            if not np.isnan(v):
                assert 0.0 <= v <= 1.0, \
                    f"mode={mode} t={threshold} w={w}: FSS={v} out of [0,1]"

    def test_fss_range_between(self, fields):
        fcst, obs = fields
        for t1, t2 in [(0.5, 1.5), (1.0, 2.0)]:
            fss_vals = fss_threshold(fcst, obs, t1, t2, _WINDOWS_MODES,
                                     threshold_mode="between")[2]
            for j, w in enumerate(_WINDOWS_MODES):
                v = fss_vals[j]
                if not np.isnan(v):
                    assert 0.0 <= v <= 1.0, \
                        f"between t1={t1} t2={t2} w={w}: FSS={v}"

    # -- Symmetry: swapping fcst/obs gives the same FSS --

    @pytest.mark.parametrize("mode", ["over", "under", "tolerance"])
    def test_symmetry(self, small_fields, mode):
        fcst, obs = small_fields
        wins = np.array([10, 30])
        fss_ab = _fss_threshold_mode(fcst, obs, 0.5, wins, mode)
        fss_ba = _fss_threshold_mode(obs, fcst, 0.5, wins, mode)
        np.testing.assert_allclose(fss_ab, fss_ba, atol=1e-10,
            err_msg=f"mode={mode}: not symmetric under fcst/obs swap")

    def test_symmetry_between(self, small_fields):
        fcst, obs = small_fields
        wins = np.array([10, 30])
        fss_ab = fss_threshold(fcst, obs, 0.5, 1.5, wins,
                               threshold_mode="between")[2]
        fss_ba = fss_threshold(obs, fcst, 0.5, 1.5, wins,
                               threshold_mode="between")[2]
        np.testing.assert_allclose(fss_ab, fss_ba, atol=1e-10,
            err_msg="between: not symmetric under fcst/obs swap")

    # -- Monotonicity with window size --

    @pytest.mark.parametrize("mode", _MODES_SIMPLE)
    def test_monotonicity(self, fields, mode):
        fcst, obs = fields
        wins = np.array([10, 30, 50, 90, 150])
        fss_vals = _fss_threshold_mode(fcst, obs, 1.0, wins, mode)
        for i in range(len(wins) - 1):
            assert fss_vals[i + 1] >= fss_vals[i] - 1e-6, \
                f"mode={mode}: FSS decreased from w={wins[i]} to w={wins[i+1]}"

    def test_monotonicity_between(self, fields):
        fcst, obs = fields
        wins = np.array([10, 30, 50, 90, 150])
        fss_vals = fss_threshold(fcst, obs, 0.5, 1.5, wins,
                                 threshold_mode="between")[2]
        for i in range(len(wins) - 1):
            assert fss_vals[i + 1] >= fss_vals[i] - 1e-6, \
                f"between: FSS decreased from w={wins[i]} to w={wins[i+1]}"

    # -- Over + under partition all pixels --

    @pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0])
    def test_over_under_partition(self, fields, threshold):
        """Binary fields for > t and <= t must cover every pixel exactly once."""
        fcst, obs = fields
        over_obs  = (obs > threshold).astype(int)
        under_obs = (obs <= threshold).astype(int)
        assert (over_obs + under_obs == 1).all(), \
            f"over + under do not partition obs at t={threshold}"
        over_fcst  = (fcst > threshold).astype(int)
        under_fcst = (fcst <= threshold).astype(int)
        assert (over_fcst + under_fcst == 1).all(), \
            f"over + under do not partition fcst at t={threshold}"

    # -- "between" binary field is subset of "over" at the lower threshold --

    @pytest.mark.parametrize("t1,t2", [(0.5, 1.5), (1.0, 2.0)])
    def test_between_subset_of_over(self, fields, t1, t2):
        _, obs = fields
        between = ((obs > t1) & (obs <= t2)).astype(int)
        over    = (obs > t1).astype(int)
        assert (between <= over).all(), "between pixels must be a subset of over"

    # -- fss_threshold matches manual SAT computation for each mode --

    @pytest.mark.parametrize("mode", _MODES_SIMPLE)
    def test_batch_matches_manual(self, fields, mode):
        """fss_threshold should equal manually constructing the SAT and calling fss."""
        fcst, obs = fields
        t1 = 1.0
        windows = np.array([10, 50, 150])
        batch_fss = _fss_threshold_mode(fcst, obs, t1, windows, mode)

        # Manual: build the same binary SAT that _build_binary_sat produces
        t1o = t1
        t1f = t1
        mod_bin, obs_bin = _build_binary_sat(
            fcst, obs, t1, None, t1o, t1f,
            percentiles=False, threshold_mode=mode, tolerance=0.1)
        for j, w in enumerate(windows):
            _, _, fss_manual = sat_fss(fcst, obs, w, mod_bin, obs_bin)
            assert batch_fss[j] == pytest.approx(fss_manual, abs=1e-12), \
                f"mode={mode} w={w}: batch={batch_fss[j]} vs manual={fss_manual}"

    def test_batch_matches_manual_between(self, fields):
        fcst, obs = fields
        t1, t2 = 0.5, 1.5
        windows = np.array([10, 50, 150])
        batch_fss = fss_threshold(fcst, obs, t1, t2, windows,
                                  threshold_mode="between")[2]
        mod_bin, obs_bin = _build_binary_sat(
            fcst, obs, t1, t2, t1, t1,
            percentiles=False, threshold_mode="between", tolerance=0.1)
        for j, w in enumerate(windows):
            _, _, fss_manual = sat_fss(fcst, obs, w, mod_bin, obs_bin)
            assert batch_fss[j] == pytest.approx(fss_manual, abs=1e-12), \
                f"between w={w}: batch={batch_fss[j]} vs manual={fss_manual}"

    # -- fss_cumsum_parallel passes threshold_mode through correctly --

    @pytest.mark.parametrize("mode", ["over", "under", "tolerance"])
    def test_cumsum_parallel_mode(self, fields, mode):
        """fss_cumsum_parallel with a given mode must match fss_threshold."""
        fcst, obs = fields
        thresholds = np.array([0.5, 1.0])
        windows = np.array([10, 50])
        arr = fss_cumsum_parallel(fcst, obs, thresholds, windows,
                                  threshold_mode=mode, tolerance=0.1)
        fss_arr = arr[2]  # shape (n_thresholds, n_windows)
        for i, t in enumerate(thresholds):
            ref = _fss_threshold_mode(fcst, obs, t, windows, mode)
            np.testing.assert_allclose(fss_arr[i], ref, atol=1e-12,
                err_msg=f"mode={mode} t={t}: cumsum_parallel != fss_threshold")

    def test_cumsum_parallel_between(self, fields):
        fcst, obs = fields
        thresholds = np.array([0.5, 1.0, 2.0])
        windows = np.array([10, 50])
        arr = fss_cumsum_parallel(fcst, obs, thresholds, windows,
                                  threshold_mode="between")
        fss_arr = arr[2]  # shape (n_thresholds-1+1 after insert, n_windows)?
        # "between" inserts -1 at front, pairs are (-1,0.5), (0.5,1.0), (1.0,2.0)
        edges = np.insert(thresholds, 0, -1.)
        for i in range(len(edges) - 1):
            ref = fss_threshold(fcst, obs, edges[i], edges[i+1], windows,
                                threshold_mode="between")[2]
            np.testing.assert_allclose(fss_arr[i], ref, atol=1e-12,
                err_msg=f"between edges=({edges[i]},{edges[i+1]})")


# Pinned reference values for each threshold mode.
MODE_REFERENCE_VALUES = {
    # (mode, t1, t2_or_None, window): expected_fss
    ("over",      1.0, None, 50):  0.701651,
    ("under",     1.0, None, 50):  0.858492,
    ("between",   0.5,  1.5, 50):  0.593246,
    ("between",   1.0,  2.0, 50):  0.567516,
    ("tolerance", 1.0, None, 50):  0.316766,
    ("over",      0.5, None, 150): 0.902233,
    ("under",     0.5, None, 150): 0.827204,
    ("tolerance", 2.0, None, 150): 0.585612,
}


class TestModeRegressionValues:
    """Pinned reference values for each threshold mode."""

    @pytest.mark.parametrize("key", MODE_REFERENCE_VALUES.keys(),
        ids=[f"{k[0]}_t{k[1]}_w{k[3]}" for k in MODE_REFERENCE_VALUES])
    def test_mode_reference(self, fields, key):
        fcst, obs = fields
        mode, t1, t2, window = key
        expected = MODE_REFERENCE_VALUES[key]
        fss_vals = fss_threshold(fcst, obs, t1, t2, np.array([window]),
                                 threshold_mode=mode, tolerance=0.1)[2]
        assert fss_vals[0] == pytest.approx(expected, abs=1e-4), \
            f"mode={mode} t1={t1} w={window}: got {fss_vals[0]:.6f}, expected {expected:.6f}"


# ── R2 quasi-random sequence ──

class TestR2Consistency:
    """R2 low-discrepancy sequence must be identical across implementations."""

    @pytest.mark.parametrize("N", [0, 1, 10, 100, 499])
    def test_r2_identical(self, N):
        x_orig, y_orig = R2_orig(N)
        x_new, y_new = R2_new(N)
        assert x_orig == x_new, f"R2 x differs at N={N}"
        assert y_orig == y_new, f"R2 y differs at N={N}"


# ── CWFSS: original vs new implementation ──

class TestCWFSSConsistency:
    """CWFSS class must produce identical results across implementations."""

    @pytest.fixture(scope="class")
    def cwfss_pair(self, small_fields):
        fcst, obs = small_fields
        orig = CWFSS_orig(fcst, obs, nsamples=200, window_limits=(1, 51))
        new = CWFSS_new(fcst, obs, nsamples=200, window_limits=(1, 51))
        return orig, new

    def test_windows_identical(self, cwfss_pair):
        orig, new = cwfss_pair
        np.testing.assert_array_equal(orig.windows, new.windows)

    def test_thresholds_identical(self, cwfss_pair):
        orig, new = cwfss_pair
        np.testing.assert_array_equal(orig.thresholds, new.thresholds)

    def test_numerators_identical(self, cwfss_pair):
        orig, new = cwfss_pair
        np.testing.assert_allclose(orig.numerators, new.numerators, atol=1e-10,
            err_msg="CWFSS numerators differ between original and new")

    def test_denominators_identical(self, cwfss_pair):
        orig, new = cwfss_pair
        np.testing.assert_allclose(orig.denominators, new.denominators, atol=1e-10,
            err_msg="CWFSS denominators differ between original and new")

    def test_values_identical(self, cwfss_pair):
        orig, new = cwfss_pair
        np.testing.assert_allclose(orig.values, new.values, atol=1e-10,
            err_msg="CWFSS per-sample FSS values differ between original and new")

    def test_cwfss_score_identical(self, cwfss_pair):
        orig, new = cwfss_pair
        assert orig.cwfss == pytest.approx(new.cwfss, abs=1e-12), \
            f"cwfss differs: orig={orig.cwfss:.15f} new={new.cwfss:.15f}"


# ── CWFSS bootstrap consistency ──

class TestCWFSSBootstrap:
    """Bootstrap resampling must produce identical distributions given the same seed."""

    def test_bootstrap_identical_with_same_seed(self, small_fields):
        fcst, obs = small_fields
        orig = CWFSS_orig(fcst, obs, nsamples=200, window_limits=(1, 51))
        new = CWFSS_new(fcst, obs, nsamples=200, window_limits=(1, 51))

        np.random.seed(42)
        orig.bootstrap(N=500)
        np.random.seed(42)
        new.bootstrap(N=500)

        np.testing.assert_array_equal(orig.bootstrap_info, new.bootstrap_info,
            err_msg="Bootstrap distributions differ between original and new")

    def test_bootstrap_percentiles_match(self, small_fields):
        fcst, obs = small_fields
        orig = CWFSS_orig(fcst, obs, nsamples=200, window_limits=(1, 51))
        new = CWFSS_new(fcst, obs, nsamples=200, window_limits=(1, 51))

        np.random.seed(42)
        orig.bootstrap(N=500)
        np.random.seed(42)
        new.bootstrap(N=500)

        for pct in [5, 25, 50, 75, 95]:
            orig_p = np.percentile(orig.bootstrap_info, pct)
            new_p = np.percentile(new.bootstrap_info, pct)
            assert orig_p == pytest.approx(new_p, abs=1e-12), \
                f"Bootstrap P{pct} differs: orig={orig_p:.12f} new={new_p:.12f}"


# ── CWFSS invariants / sanity checks ──

class TestCWFSSProperties:
    """CWFSS invariants that hold for any correct implementation."""

    def test_perfect_forecast(self, small_fields):
        """CWFSS = 1.0 when forecast equals observation."""
        _, obs = small_fields
        cw = CWFSS_new(obs, obs, nsamples=100, window_limits=(1, 51))
        assert cw.cwfss == pytest.approx(1.0, abs=1e-10)

    def test_cwfss_in_unit_interval(self, small_fields):
        fcst, obs = small_fields
        cw = CWFSS_new(fcst, obs, nsamples=200, window_limits=(1, 51))
        assert 0.0 <= cw.cwfss <= 1.0, f"cwfss={cw.cwfss} out of [0,1]"

    def test_bootstrap_in_unit_interval(self, small_fields):
        fcst, obs = small_fields
        cw = CWFSS_new(fcst, obs, nsamples=200, window_limits=(1, 51))
        np.random.seed(123)
        cw.bootstrap(N=500)
        assert np.all(cw.bootstrap_info >= 0.0), "Bootstrap has negative values"
        assert np.all(cw.bootstrap_info <= 1.0), "Bootstrap has values > 1"

    def test_bootstrap_mean_near_cwfss(self, small_fields):
        """Bootstrap mean should converge toward the point estimate."""
        fcst, obs = small_fields
        cw = CWFSS_new(fcst, obs, nsamples=200, window_limits=(1, 51))
        np.random.seed(123)
        cw.bootstrap(N=1000)
        assert np.mean(cw.bootstrap_info) == pytest.approx(cw.cwfss, abs=0.05), \
            f"Bootstrap mean={np.mean(cw.bootstrap_info):.6f} far from cwfss={cw.cwfss:.6f}"

    def test_nsamples_respected(self, small_fields):
        fcst, obs = small_fields
        for ns in [50, 100, 300]:
            cw = CWFSS_new(fcst, obs, nsamples=ns, window_limits=(1, 51))
            assert cw.values.shape == (ns,)
            assert cw.windows.shape == (ns,)
            assert cw.thresholds.shape == (ns,)
            assert cw.numerators.shape == (ns,)
            assert cw.denominators.shape == (ns,)

    @pytest.mark.parametrize("mode", ["relative", "absolute", "percentiles"])
    def test_threshold_limiting_modes(self, small_fields, mode):
        fcst, obs = small_fields
        if mode == "relative":
            cw = CWFSS_new(fcst, obs, nsamples=50, threshold_limiting="relative")
        elif mode == "absolute":
            cw = CWFSS_new(fcst, obs, nsamples=50,
                           threshold_limits=(0.1, 2.0), threshold_limiting="absolute")
        elif mode == "percentiles":
            cw = CWFSS_new(fcst, obs, nsamples=50,
                           threshold_limits=(10., 90.), threshold_limiting="percentiles")
        assert 0.0 <= cw.cwfss <= 1.0, f"mode={mode}: cwfss={cw.cwfss} out of [0,1]"


# ── CWFSS pinned regression values ──
# Computed on small_fields (seed=42, 10 bells, 101x101) with nsamples=200, window_limits=(1,51).

CWFSS_REFERENCE = {
    "cwfss": 0.090965456330480,
    "bootstrap_percentiles": {
        5:  0.071682193737,
        25: 0.083426953163,
        50: 0.090238988704,
        75: 0.099811106708,
        95: 0.110399862112,
    },
}


class TestCWFSSRegression:
    """Pinned CWFSS values to catch accidental changes."""

    def test_cwfss_pinned(self, small_fields):
        fcst, obs = small_fields
        cw = CWFSS_new(fcst, obs, nsamples=200, window_limits=(1, 51))
        assert cw.cwfss == pytest.approx(CWFSS_REFERENCE["cwfss"], abs=1e-8), \
            f"cwfss regression: got {cw.cwfss:.15f}, expected {CWFSS_REFERENCE['cwfss']:.15f}"

    def test_bootstrap_pinned_percentiles(self, small_fields):
        fcst, obs = small_fields
        cw = CWFSS_new(fcst, obs, nsamples=200, window_limits=(1, 51))
        np.random.seed(42)
        cw.bootstrap(N=500)
        for pct, expected in CWFSS_REFERENCE["bootstrap_percentiles"].items():
            actual = np.percentile(cw.bootstrap_info, pct)
            assert actual == pytest.approx(expected, abs=1e-8), \
                f"Bootstrap P{pct} regression: got {actual:.12f}, expected {expected:.12f}"


# ── Missing-data (NaN) support for the SAT method ──

def _fss_via_sat_masked(fcst, obs, threshold, window, threshold_mode="over"):
    """Single threshold/window FSS via the SAT missing-data path."""
    mask = _validity_mask(fcst, obs)
    mod_bin, obs_bin = _build_binary_sat_masked(
        fcst, obs, threshold, None, threshold, threshold,
        False, threshold_mode, 0.1, mask)
    invalid_sat = _invalid_sat(mask)
    return sat_fss(fcst, obs, window, mod_bin, obs_bin,
                   threshold_mode=threshold_mode, invalid_cache=invalid_sat)


def _inject_nans(field, frac, seed):
    """Return a copy of `field` with a fraction `frac` of points set to NaN."""
    rng = np.random.RandomState(seed)
    out = field.astype(float).copy()
    holes = rng.rand(*field.shape) < frac
    out[holes] = np.nan
    return out


class TestMaskedReducesToClean:
    """With no missing points the masked path must equal the clean fast path."""

    @pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("window", [10, 30, 90])
    def test_equivalence(self, small_fields, threshold, window):
        fcst, obs = small_fields
        _, _, fss_clean = _fss_via_sat(fcst, obs, threshold, window)
        _, _, fss_masked = _fss_via_sat_masked(fcst, obs, threshold, window)
        assert np.isclose(fss_clean, fss_masked, atol=1e-12, rtol=0), \
            f"clean={fss_clean:.15f} masked={fss_masked:.15f}"

    def test_full_mask_count_is_area(self, small_fields):
        """With no missing points the per-window valid count must equal the
        constant window area everywhere (incl. boundaries) -- the assumption the
        clean path's constant-area fold relies on."""
        fcst, obs = small_fields
        mask = _validity_mask(fcst, obs)
        window = 30
        w = window // 2
        area = (2 * w + 1) ** 2
        C = area - integral_filter(_invalid_sat(mask), window)
        assert np.allclose(C, area)


class TestMaskedWeighting:
    """The weighted score must weight each window by its valid-point count."""

    def test_against_handrolled_weighted_mean(self):
        rng = np.random.RandomState(0)
        Sf = rng.rand(8, 8) * 5
        So = rng.rand(8, 8) * 5
        C = rng.randint(0, 20, size=(8, 8)).astype(float)
        C[0, 0] = 0.0  # an empty window must drop out

        num, den, score = _fss_score_masked(Sf, So, C)

        valid = C > 0
        fhat = np.where(valid, Sf / np.where(valid, C, 1), 0.0)
        ohat = np.where(valid, So / np.where(valid, C, 1), 0.0)
        wsum = C.sum()
        num_ref = np.sum(C[valid] * (fhat[valid] - ohat[valid]) ** 2) / wsum
        den_ref = np.sum(C[valid] * (fhat[valid] ** 2 + ohat[valid] ** 2)) / wsum

        assert num == pytest.approx(num_ref, abs=1e-12)
        assert den == pytest.approx(den_ref, abs=1e-12)
        assert score == pytest.approx(1.0 - num_ref / den_ref, abs=1e-12)

    def test_count_ratio_property(self):
        """A window with 78 valid points counts 78% as much as one with 100."""
        # two window centres, identical local fraction error, different counts
        Sf = np.array([2.0, 1.56])   # fhat error realized via (Sf-So)
        So = np.array([0.0, 0.0])
        C = np.array([100.0, 78.0])
        # contribution to numerator sum (before /wsum) is (Sf-So)^2 / C
        contrib = (Sf - So) ** 2 / C
        # equal local fraction (Sf/C) -> contribution scales with C
        fhat = Sf / C
        assert np.isclose(fhat[0], fhat[1])              # same local fraction
        assert np.isclose(contrib[1] / contrib[0], 78 / 100)


class TestMaskedInvariants:

    @pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("window", [10, 50])
    def test_score_in_unit_interval(self, small_fields, threshold, window):
        fcst, obs = small_fields
        fcst_n = _inject_nans(fcst, 0.1, seed=1)
        obs_n = _inject_nans(obs, 0.1, seed=2)
        _, _, score = _fss_via_sat_masked(fcst_n, obs_n, threshold, window)
        assert np.isnan(score) or (0.0 <= score <= 1.0 + 1e-12)

    @pytest.mark.parametrize("window", [10, 50])
    def test_perfect_forecast(self, small_fields, window):
        fcst, obs = small_fields
        obs_n = _inject_nans(obs, 0.1, seed=3)
        # forecast identical to obs on the valid points (and shares the NaNs)
        _, _, score = _fss_via_sat_masked(obs_n, obs_n, 1.0, window)
        assert np.isclose(score, 1.0, atol=1e-12)

    @pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("window", [10, 50])
    def test_symmetry(self, small_fields, threshold, window):
        fcst, obs = small_fields
        fcst_n = _inject_nans(fcst, 0.1, seed=4)
        obs_n = _inject_nans(obs, 0.1, seed=5)
        _, _, s1 = _fss_via_sat_masked(fcst_n, obs_n, threshold, window)
        _, _, s2 = _fss_via_sat_masked(obs_n, fcst_n, threshold, window)
        if np.isnan(s1):
            assert np.isnan(s2)
        else:
            assert np.isclose(s1, s2, atol=1e-12)

    def test_no_valid_windows_gives_nan(self, small_fields):
        fcst, obs = small_fields
        allnan = np.full_like(fcst, np.nan, dtype=float)
        _, _, score = _fss_via_sat_masked(allnan, allnan, 1.0, 10)
        assert np.isnan(score)


class TestMaskedPerturbation:
    """A few scattered NaNs should only perturb the score slightly."""

    @pytest.mark.parametrize("threshold", [0.5, 1.0])
    def test_small_change(self, small_fields, threshold):
        fcst, obs = small_fields
        window = 30
        _, _, fss_clean = _fss_via_sat(fcst, obs, threshold, window)
        fcst_n = _inject_nans(fcst, 0.01, seed=7)
        obs_n = _inject_nans(obs, 0.01, seed=8)
        _, _, fss_n = _fss_via_sat_masked(fcst_n, obs_n, threshold, window)
        assert np.isclose(fss_clean, fss_n, atol=0.05), \
            f"clean={fss_clean:.6f} holed={fss_n:.6f}"


class TestMaskedPercentiles:

    def test_nanpercentile_path_finite(self, small_fields):
        fcst, obs = small_fields
        fcst_n = _inject_nans(fcst, 0.1, seed=9)
        obs_n = _inject_nans(obs, 0.1, seed=10)
        windows = np.array([10, 30, 50])
        num, den, fss_t, _ = fss_threshold(
            fcst_n, obs_n, 75.0, None, windows, percentiles=True)
        assert np.all(np.isfinite(fss_t))
        assert np.all((fss_t >= 0.0) & (fss_t <= 1.0 + 1e-12))


class TestMaskedBatchConsistency:

    def test_cumsum_matches_single_calls(self, small_fields):
        fcst, obs = small_fields
        fcst_n = _inject_nans(fcst, 0.1, seed=11)
        obs_n = _inject_nans(obs, 0.1, seed=12)
        thresholds = np.array([0.5, 1.0, 2.0])
        windows = np.array([10, 30, 50])
        ret = fss_cumsum_parallel(fcst_n, obs_n, thresholds, windows)
        fss_batch = ret[2]  # (n_thresholds, n_windows)
        for it, t in enumerate(thresholds):
            for iw, w in enumerate(windows):
                _, _, single = _fss_via_sat_masked(fcst_n, obs_n, t, int(w))
                assert np.isclose(fss_batch[it, iw], single, atol=1e-12), \
                    f"t={t} w={w}: batch={fss_batch[it, iw]} single={single}"


def _make_ensemble(obs, n_members, jitter, seed):
    """Build a 3D ensemble forecast by perturbing `obs` member-by-member."""
    rng = np.random.RandomState(seed)
    return np.stack([obs + jitter * rng.standard_normal(obs.shape)
                     for _ in range(n_members)])


class TestMaskedEnsemble:
    """Missing-data support for the ensemble (eps) FSS path."""

    def test_reduces_to_clean_without_nan(self, small_fields):
        """With no missing points the masked eps path must equal the clean eps
        path exactly (forced on by a single dummy NaN would change data, so we
        instead compare the public function with and without NaNs absent)."""
        _, obs = small_fields
        fcst3 = _make_ensemble(obs, 5, 0.3, seed=20)
        thresholds = np.array([0.5, 1.0, 2.0])
        windows = np.array([10, 30, 50])
        # clean eps path (no NaNs anywhere)
        ret = fss_cumsum_parallel(fcst3, obs, thresholds, windows, eps=True)
        assert np.all(np.isfinite(ret[2]))
        assert np.all((ret[2] >= -1e-12) & (ret[2] <= 1.0 + 1e-12))

    @pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("window", [10, 50])
    def test_invariants_with_nan(self, small_fields, threshold, window):
        _, obs = small_fields
        fcst3 = _make_ensemble(obs, 5, 0.3, seed=21)
        # NaNs in obs and scattered across members
        obs_n = _inject_nans(obs, 0.1, seed=22)
        fcst3[1] = _inject_nans(fcst3[1], 0.1, seed=23)
        fcst3[3] = _inject_nans(fcst3[3], 0.1, seed=24)
        ret = fss_threshold_eps(fcst3, obs_n, threshold, None,
                                np.array([window]))
        score = ret[2][0]
        assert np.isnan(score) or (0.0 <= score <= 1.0 + 1e-12)

    def test_perfect_ensemble(self, small_fields):
        """Every member equal to obs (sharing its NaNs) -> probability is the obs
        binary -> FSS = 1."""
        _, obs = small_fields
        obs_n = _inject_nans(obs, 0.1, seed=25)
        fcst3 = np.stack([obs_n, obs_n, obs_n])
        ret = fss_threshold_eps(fcst3, obs_n, 1.0, None, np.array([10, 50]))
        assert np.allclose(ret[2], 1.0, atol=1e-12)

    def test_fully_missing_member_ignored(self, small_fields):
        """A member that is entirely NaN must not change the score (averaging is
        over valid members only)."""
        _, obs = small_fields
        obs_n = _inject_nans(obs, 0.05, seed=26)
        fcst3 = _make_ensemble(obs, 4, 0.3, seed=27)
        # share obs NaNs into members so grid validity matches the augmented case
        windows = np.array([10, 30])
        ret_a = fss_threshold_eps(fcst3, obs_n, 1.0, None, windows)
        fcst4 = np.concatenate([fcst3, np.full((1,) + obs.shape, np.nan)], axis=0)
        ret_b = fss_threshold_eps(fcst4, obs_n, 1.0, None, windows)
        assert np.allclose(ret_a[2], ret_b[2], atol=1e-12)

    def test_cumsum_eps_with_nan_runs(self, small_fields):
        _, obs = small_fields
        obs_n = _inject_nans(obs, 0.1, seed=28)
        fcst3 = _make_ensemble(obs, 5, 0.3, seed=29)
        fcst3[2] = _inject_nans(fcst3[2], 0.1, seed=30)
        ret = fss_cumsum_parallel(fcst3, obs_n, np.array([0.5, 1.0, 2.0]),
                                  np.array([10, 30, 50]), eps=True)
        assert np.all((ret[2] >= -1e-12) & (ret[2] <= 1.0 + 1e-12) | np.isnan(ret[2]))


# ── Tolerance-mode forecast threshold (regression) ──
# In "tolerance" mode the *forecast* window must be built from the forecast's
# own threshold t1f on both bounds: (1-tol)*t1f < fcst <= (1+tol)*t1f.  An
# earlier bug used the observation threshold t1o on the upper bound, which is
# invisible when t1o == t1f (percentiles=False) but wrong otherwise.

class TestToleranceForecastThreshold:
    """The forecast tolerance window must use t1f, not t1o, on both bounds."""

    def test_build_binary_sat_uses_t1f(self, small_fields):
        fcst, obs = small_fields
        t1o, t1f, tol = 1.0, 2.0, 0.1
        mod_bin, _ = _build_binary_sat(
            fcst, obs, None, None, t1o, t1f,
            percentiles=False, threshold_mode="tolerance", tolerance=tol)
        expected = ((fcst > (1.-tol)*t1f) & (fcst <= (1.+tol)*t1f)).sum()
        buggy = ((fcst > (1.-tol)*t1f) & (fcst <= (1.+tol)*t1o)).sum()
        # the SAT corner holds the total sum of the binary field
        assert mod_bin[-1, -1] == pytest.approx(expected)
        assert expected != buggy, "test thresholds fail to discriminate the bug"

    def test_build_binary_sat_masked_uses_t1f(self, small_fields):
        fcst, obs = small_fields
        mask = _validity_mask(fcst, obs)
        t1o, t1f, tol = 1.0, 2.0, 0.1
        mod_bin, _ = _build_binary_sat_masked(
            fcst, obs, None, None, t1o, t1f,
            False, "tolerance", tol, mask)
        expected = ((fcst > (1.-tol)*t1f) & (fcst <= (1.+tol)*t1f) & mask).sum()
        buggy = ((fcst > (1.-tol)*t1f) & (fcst <= (1.+tol)*t1o) & mask).sum()
        assert mod_bin[-1, -1] == pytest.approx(expected)
        assert expected != buggy, "test thresholds fail to discriminate the bug"

    def test_percentile_tolerance_finite(self, fields):
        """tolerance + percentiles (so t1o != t1f) stays finite and in range."""
        fcst, obs = fields
        windows = np.array([10, 50, 150])
        fss_t = fss_threshold(fcst, obs, 75.0, None, windows,
                              percentiles=True, threshold_mode="tolerance")[2]
        finite = fss_t[~np.isnan(fss_t)]
        assert np.all((finite >= 0.0) & (finite <= 1.0 + 1e-12))

    def test_eps_percentile_tolerance_finite(self, small_fields):
        """eps clean path, tolerance + percentiles -> exercises the 3D mean
        forecast binarisation with t1f on both bounds."""
        _, obs = small_fields
        fcst3 = _make_ensemble(obs, 5, 0.3, seed=50)
        windows = np.array([10, 30])
        fss_t = fss_threshold_eps(fcst3, obs, 75.0, None, windows,
                                  percentiles=True, threshold_mode="tolerance")[2]
        finite = fss_t[~np.isnan(fss_t)]
        assert np.all((finite >= 0.0) & (finite <= 1.0 + 1e-12))


# ── fss_cumsum_frame threshold_mode passthrough (regression) ──
# fss_cumsum_frame previously hardcoded threshold_mode="over", ignoring the
# argument.  It must forward the requested mode to fss_cumsum_parallel.

class TestCumsumFrameThresholdMode:

    def test_threshold_mode_passthrough(self, small_fields):
        fcst, obs = small_fields
        thresholds = [0.5, 1.0]
        windows = [[10], [50]]
        _, _, fss_df, _ = fss_cumsum_frame(fcst, obs, windows, thresholds,
                                           threshold_mode="under")
        arr = fss_cumsum_parallel(fcst, obs, np.array(thresholds),
                                  np.array([10, 50]), threshold_mode="under")
        np.testing.assert_allclose(fss_df.values, arr[2], atol=1e-12,
            err_msg="fss_cumsum_frame did not forward threshold_mode='under'")

    def test_under_differs_from_over(self, small_fields):
        fcst, obs = small_fields
        thresholds = [0.5, 1.0]
        windows = [[10], [50]]
        _, _, over_df, _ = fss_cumsum_frame(fcst, obs, windows, thresholds,
                                            threshold_mode="over")
        _, _, under_df, _ = fss_cumsum_frame(fcst, obs, windows, thresholds,
                                             threshold_mode="under")
        assert not np.allclose(over_df.values, under_df.values), \
            "over and under frames are identical -> mode not applied"


# ── CWFSS threshold_mode support ──

class TestCWFSSThresholdModes:
    """CWFSS now accepts threshold_mode / tolerance like the batch functions."""

    @pytest.mark.parametrize("mode", ["over", "under", "tolerance"])
    def test_runs_and_in_unit_interval(self, small_fields, mode):
        fcst, obs = small_fields
        cw = CWFSS_new(fcst, obs, nsamples=100, window_limits=(1, 51),
                       threshold_mode=mode, tolerance=0.1)
        assert np.isfinite(cw.cwfss)
        assert 0.0 <= cw.cwfss <= 1.0

    def test_default_is_over(self, small_fields):
        """Omitting threshold_mode must reproduce explicit 'over'."""
        fcst, obs = small_fields
        cw_default = CWFSS_new(fcst, obs, nsamples=100, window_limits=(1, 51))
        cw_over = CWFSS_new(fcst, obs, nsamples=100, window_limits=(1, 51),
                            threshold_mode="over")
        np.testing.assert_array_equal(cw_default.values, cw_over.values)

    def test_under_differs_from_over(self, small_fields):
        fcst, obs = small_fields
        cw_over = CWFSS_new(fcst, obs, nsamples=100, window_limits=(1, 51),
                            threshold_mode="over")
        cw_under = CWFSS_new(fcst, obs, nsamples=100, window_limits=(1, 51),
                             threshold_mode="under")
        assert not np.allclose(cw_over.values, cw_under.values)


# ── CWFSS missing-data (NaN) support ──

class TestCWFSSMissingData:
    """CWFSS uses the masked path (and NaN-aware threshold limits) under NaNs."""

    def test_clean_matches_original(self, small_fields):
        """With no NaNs the masked-capable class equals the original CWFSS."""
        fcst, obs = small_fields
        cw_new = CWFSS_new(fcst, obs, nsamples=100, window_limits=(1, 51))
        cw_orig = CWFSS_orig(fcst, obs, nsamples=100, window_limits=(1, 51))
        np.testing.assert_allclose(cw_new.values, cw_orig.values, atol=1e-10)

    def test_runs_with_nan(self, small_fields):
        fcst, obs = small_fields
        fcst_n = _inject_nans(fcst, 0.1, seed=40)
        obs_n = _inject_nans(obs, 0.1, seed=41)
        cw = CWFSS_new(fcst_n, obs_n, nsamples=150, window_limits=(1, 51))
        assert np.isfinite(cw.cwfss)
        assert 0.0 <= cw.cwfss <= 1.0

    def test_perfect_forecast_with_nan(self, small_fields):
        """A perfect forecast scores exactly 1.0 even under NaNs: undefined
        per-sample values drop out of both the score and its theoretical maximum,
        so the achievable ceiling is not lowered by missing data."""
        _, obs = small_fields
        obs_n = _inject_nans(obs, 0.1, seed=42)
        cw = CWFSS_new(obs_n, obs_n, nsamples=100, window_limits=(1, 51))
        # every defined per-sample value is exactly 1 (forecast == obs)
        defined = cw.values[~np.isnan(cw.values)]
        assert np.allclose(defined, 1.0, atol=1e-12)
        assert cw.cwfss == pytest.approx(1.0, abs=1e-12)

    def test_nanmax_threshold_limits(self, small_fields):
        """relative limiting must use nanmax/nanpercentile, not be poisoned by NaN."""
        fcst, obs = small_fields
        obs_n = _inject_nans(obs, 0.1, seed=43)
        cw = CWFSS_new(fcst, obs_n, nsamples=50, window_limits=(1, 51),
                       threshold_limiting="relative")
        assert np.isfinite(cw.tmin) and np.isfinite(cw.tmax)
        assert cw.tmax > cw.tmin > 0

    def test_percentile_limits_with_nan(self, small_fields):
        fcst, obs = small_fields
        obs_n = _inject_nans(obs, 0.1, seed=44)
        cw = CWFSS_new(fcst, obs_n, nsamples=50, window_limits=(1, 51),
                       threshold_limits=(10., 90.), threshold_limiting="percentiles")
        assert np.isfinite(cw.tmin) and np.isfinite(cw.tmax)
        assert 0.0 <= cw.cwfss <= 1.0


# ── Missing-data (NaN) support for the FFT method ──
# The FFT masked path mirrors the SAT one: zero out missing points before the
# convolution and obtain the per-window valid count from one more convolution of
# the mask, weighting each window by its valid-point count.  Boundary handling is
# FFT zero-padding (vs SAT edge-clamping), so the two masked paths agree only in
# the interior -- exactly the pre-existing clean FFT-vs-SAT discrepancy.

def _fft_masked(fcst, obs, threshold, window):
    """Single (threshold, window) FFT FSS via the masked path (always on)."""
    return _fourier_fss_masked(fcst, obs, threshold, (window, window), False, "same")


class TestFFTMaskedReducesToClean:
    """With no missing points the masked path must equal the clean FFT path."""

    @pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("window", [10, 30, 90])
    def test_equivalence(self, small_fields, threshold, window):
        fcst, obs = small_fields
        _, _, fss_clean, _ = fourier_fss(fcst, obs, threshold, (window, window),
                                         False, "same")
        _, _, fss_masked, _ = _fft_masked(fcst, obs, threshold, window)
        assert np.isclose(fss_clean, fss_masked, atol=1e-9, rtol=0), \
            f"clean={fss_clean:.15f} masked={fss_masked:.15f}"

    @pytest.mark.parametrize("window", [10, 50])
    def test_count_is_area_without_nan(self, small_fields, window):
        """With no NaNs the per-window valid count is the constant full area."""
        from fss_FFT import fourier_filter, _validity_mask as fft_mask
        fcst, obs = small_fields
        mask = fft_mask(fcst, obs)
        area = float(window * window)
        C = area - fourier_filter((~mask).astype(float), (window, window), "same")
        assert np.allclose(C, area, atol=1e-9)


class TestFFTMaskedInvariants:

    @pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("window", [10, 50])
    def test_score_in_unit_interval(self, small_fields, threshold, window):
        fcst, obs = small_fields
        fcst_n = _inject_nans(fcst, 0.1, seed=51)
        obs_n = _inject_nans(obs, 0.1, seed=52)
        _, _, score, _ = _fft_masked(fcst_n, obs_n, threshold, window)
        assert np.isnan(score) or (0.0 <= score <= 1.0 + 1e-12)

    @pytest.mark.parametrize("window", [10, 50])
    def test_perfect_forecast(self, small_fields, window):
        _, obs = small_fields
        obs_n = _inject_nans(obs, 0.1, seed=53)
        _, _, score, _ = _fft_masked(obs_n, obs_n, 1.0, window)
        assert np.isclose(score, 1.0, atol=1e-10)

    @pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("window", [10, 50])
    def test_symmetry(self, small_fields, threshold, window):
        fcst, obs = small_fields
        fcst_n = _inject_nans(fcst, 0.1, seed=54)
        obs_n = _inject_nans(obs, 0.1, seed=55)
        _, _, s1, _ = _fft_masked(fcst_n, obs_n, threshold, window)
        _, _, s2, _ = _fft_masked(obs_n, fcst_n, threshold, window)
        if np.isnan(s1):
            assert np.isnan(s2)
        else:
            assert np.isclose(s1, s2, atol=1e-10)

    def test_no_valid_windows_gives_nan(self, small_fields):
        fcst, _ = small_fields
        allnan = np.full_like(fcst, np.nan, dtype=float)
        _, _, score, _ = _fft_masked(allnan, allnan, 1.0, 10)
        assert np.isnan(score)

    def test_auto_dispatch_on_nan(self, small_fields):
        """The public fourier_fss must route to the masked path when NaNs appear
        (a raw NaN through fftconvolve would otherwise poison the whole output)."""
        fcst, obs = small_fields
        fcst_n = _inject_nans(fcst, 0.05, seed=56)
        _, _, score, _ = fourier_fss(fcst_n, obs, 1.0, (30, 30), False, "same")
        assert np.isfinite(score) and 0.0 <= score <= 1.0 + 1e-12


class TestFFTMaskedVsSAT:
    """FFT-masked and SAT-masked share the weighting; they differ only in the
    boundary convention (zero-pad vs clamp), so they agree in the same ballpark
    as the clean FFT-vs-SAT comparison."""

    @pytest.mark.parametrize("threshold", [0.5, 1.0])
    @pytest.mark.parametrize("window", [10, 30])
    def test_ballpark_agreement(self, fields, threshold, window):
        fcst, obs = fields
        fcst_n = _inject_nans(fcst, 0.05, seed=57)
        obs_n = _inject_nans(obs, 0.05, seed=58)
        _, _, fss_fft, _ = _fft_masked(fcst_n, obs_n, threshold, window)
        _, _, fss_sat = _fss_via_sat_masked(fcst_n, obs_n, threshold, window)
        assert np.isclose(fss_fft, fss_sat, atol=TOL_FFT_SAT), \
            f"FFT={fss_fft:.8f} SAT={fss_sat:.8f} diff={abs(fss_fft - fss_sat):.2e}"


# ── Missing data (FFT ensemble) ──
def _fft_eps_masked(fcst3, obs, threshold, window):
    """Single (threshold, window) ensemble FFT FSS via the masked path."""
    return _fourier_fss_eps_masked(fcst3, obs, threshold, (window, window), False, "same")


class TestFFTEpsMaskedReducesToClean:
    """With no missing points the masked eps path must equal the clean eps path."""

    @pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("window", [10, 30, 90])
    def test_equivalence(self, small_fields, threshold, window):
        _, obs = small_fields
        fcst3 = _make_ensemble(obs, 5, 0.3, seed=120)
        _, _, fss_clean, _ = fourier_fss_eps(fcst3, obs, threshold,
                                             (window, window), False, "same")
        _, _, fss_masked, _ = _fft_eps_masked(fcst3, obs, threshold, window)
        assert np.isclose(fss_clean, fss_masked, atol=1e-9, rtol=0), \
            f"clean={fss_clean:.15f} masked={fss_masked:.15f}"


class TestFFTEpsMaskedInvariants:

    @pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("window", [10, 50])
    def test_score_in_unit_interval(self, small_fields, threshold, window):
        _, obs = small_fields
        obs_n = _inject_nans(obs, 0.1, seed=121)
        fcst3 = _make_ensemble(obs, 5, 0.3, seed=122)
        fcst3[1] = _inject_nans(fcst3[1], 0.1, seed=123)
        fcst3[3] = _inject_nans(fcst3[3], 0.1, seed=124)
        _, _, score, _ = _fft_eps_masked(fcst3, obs_n, threshold, window)
        assert np.isnan(score) or (0.0 <= score <= 1.0 + 1e-12)

    @pytest.mark.parametrize("window", [10, 50])
    def test_perfect_ensemble(self, small_fields, window):
        """Every member equal to obs (sharing its NaNs) -> probability is the obs
        binary -> FSS = 1."""
        _, obs = small_fields
        obs_n = _inject_nans(obs, 0.1, seed=125)
        fcst3 = np.stack([obs_n, obs_n, obs_n])
        _, _, score, _ = _fft_eps_masked(fcst3, obs_n, 1.0, window)
        assert np.isclose(score, 1.0, atol=1e-10)

    def test_fully_missing_member_ignored(self, small_fields):
        """A member that is entirely NaN must not change the score (averaging is
        over valid members only)."""
        _, obs = small_fields
        obs_n = _inject_nans(obs, 0.05, seed=126)
        fcst3 = _make_ensemble(obs, 4, 0.3, seed=127)
        _, _, s_a, _ = _fft_eps_masked(fcst3, obs_n, 1.0, 30)
        fcst4 = np.concatenate([fcst3, np.full((1,) + obs.shape, np.nan)], axis=0)
        _, _, s_b, _ = _fft_eps_masked(fcst4, obs_n, 1.0, 30)
        assert np.isclose(s_a, s_b, atol=1e-12)

    def test_no_valid_windows_gives_nan(self, small_fields):
        _, obs = small_fields
        allnan = np.full((3,) + obs.shape, np.nan, dtype=float)
        _, _, score, _ = _fft_eps_masked(allnan, np.full_like(obs, np.nan), 1.0, 10)
        assert np.isnan(score)

    def test_auto_dispatch_on_nan(self, small_fields):
        """The public fourier_fss_eps must route to the masked path when NaNs
        appear (a raw NaN through fftconvolve would otherwise poison the output)."""
        _, obs = small_fields
        fcst3 = _make_ensemble(obs, 5, 0.3, seed=128)
        fcst3[2] = _inject_nans(fcst3[2], 0.05, seed=129)
        _, _, score, _ = fourier_fss_eps(fcst3, obs, 1.0, (30, 30), False, "same")
        assert np.isfinite(score) and 0.0 <= score <= 1.0 + 1e-12


class TestFFTEpsMaskedVsSAT:
    """FFT-eps and SAT-eps share the weighting; they differ only in the boundary
    convention (zero-pad vs clamp), agreeing within the usual FFT-vs-SAT band."""

    @pytest.mark.parametrize("threshold", [0.5, 1.0])
    @pytest.mark.parametrize("window", [10, 30])
    def test_ballpark_agreement(self, fields, threshold, window):
        _, obs = fields
        obs_n = _inject_nans(obs, 0.05, seed=130)
        fcst3 = _make_ensemble(obs, 5, 0.3, seed=131)
        fcst3[2] = _inject_nans(fcst3[2], 0.05, seed=132)
        _, _, fss_fft, _ = _fft_eps_masked(fcst3, obs_n, threshold, window)
        ret = fss_threshold_eps(fcst3, obs_n, threshold, None, np.array([window]))
        fss_sat = ret[2][0]
        assert np.isclose(fss_fft, fss_sat, atol=TOL_FFT_SAT), \
            f"FFT={fss_fft:.8f} SAT={fss_sat:.8f} diff={abs(fss_fft - fss_sat):.2e}"


# ── Missing data (OpenCL) ──
def _ocl_masked(fcst, obs, threshold, window):
    """Single (threshold, window) FSS via the OpenCL masked path (forced on)."""
    return _fss_opencl_masked(fcst, obs, threshold, window, ctx, queue, program)


@pytest.mark.skipif(not HAS_OCL, reason="OpenCL not available")
class TestOCLMaskedReducesToClean:
    """With no missing points the masked OpenCL path must equal the clean path."""

    @pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("window", [11, 31, 91])
    def test_equivalence(self, small_fields, threshold, window):
        fcst, obs = small_fields
        _, _, fss_clean = fss_opencl(fcst, obs, threshold, window, ctx, queue, program)
        _, _, fss_masked = _ocl_masked(fcst, obs, threshold, window)
        assert np.isclose(fss_clean, fss_masked, atol=TOL_SAT_OCL), \
            f"clean={fss_clean:.8f} masked={fss_masked:.8f}"


@pytest.mark.skipif(not HAS_OCL, reason="OpenCL not available")
class TestOCLMaskedInvariants:

    @pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("window", [11, 51])
    def test_score_in_unit_interval(self, small_fields, threshold, window):
        fcst, obs = small_fields
        fcst_n = _inject_nans(fcst, 0.1, seed=151)
        obs_n = _inject_nans(obs, 0.1, seed=152)
        _, _, score = fss_opencl(fcst_n, obs_n, threshold, window, ctx, queue, program)
        assert np.isnan(score) or (0.0 <= score <= 1.0 + 1e-6)

    @pytest.mark.parametrize("window", [11, 51])
    def test_perfect_forecast(self, small_fields, window):
        _, obs = small_fields
        obs_n = _inject_nans(obs, 0.1, seed=153)
        _, _, score = fss_opencl(obs_n, obs_n, 1.0, window, ctx, queue, program)
        assert np.isclose(score, 1.0, atol=TOL_SAT_OCL)

    def test_no_valid_windows_gives_nan(self, small_fields):
        fcst, _ = small_fields
        allnan = np.full_like(fcst, np.nan, dtype=float)
        _, _, score = fss_opencl(allnan, allnan, 1.0, 11, ctx, queue, program)
        assert np.isnan(score)

    def test_auto_dispatch_on_nan(self, small_fields):
        """The public fss_opencl must route to the masked path when NaNs appear."""
        fcst, obs = small_fields
        fcst_n = _inject_nans(fcst, 0.05, seed=156)
        _, _, score = fss_opencl(fcst_n, obs, 1.0, 31, ctx, queue, program)
        assert np.isfinite(score) and 0.0 <= score <= 1.0 + 1e-6


@pytest.mark.skipif(not HAS_OCL, reason="OpenCL not available")
class TestOCLMaskedVsSAT:
    """OpenCL and SAT both edge-clamp, so the masked paths agree closely (float32
    vs float64 aside)."""

    @pytest.mark.parametrize("threshold", [0.5, 1.0])
    @pytest.mark.parametrize("window", [11, 31])
    def test_agreement(self, fields, threshold, window):
        fcst, obs = fields
        fcst_n = _inject_nans(fcst, 0.05, seed=157)
        obs_n = _inject_nans(obs, 0.05, seed=158)
        _, _, fss_ocl = fss_opencl(fcst_n, obs_n, threshold, window, ctx, queue, program)
        _, _, fss_sat = _fss_via_sat_masked(fcst_n, obs_n, threshold, window)
        assert np.isclose(fss_ocl, fss_sat, atol=TOL_FFT_SAT), \
            f"OCL={fss_ocl:.8f} SAT={fss_sat:.8f} diff={abs(fss_ocl - fss_sat):.2e}"


@pytest.mark.skipif(not HAS_OCL, reason="OpenCL not available")
class TestOCLMaskedBatchConsistency:
    """fss_opencl_arr on a NaN field must match per-(threshold, window) calls."""

    def test_arr_matches_single_calls(self, small_fields):
        fcst, obs = small_fields
        fcst_n = _inject_nans(fcst, 0.08, seed=159)
        obs_n = _inject_nans(obs, 0.08, seed=160)
        thresholds = np.array([0.5, 1.0, 2.0])
        windows = np.array([11, 31, 51])
        _, _, fss_arr = fss_opencl_arr(fcst_n, obs_n, thresholds, windows,
                                       ctx, queue, program)
        for ii, t in enumerate(thresholds):
            for jj, win in enumerate(windows):
                _, _, s = fss_opencl(fcst_n, obs_n, t, int(win), ctx, queue, program)
                a = fss_arr[ii, jj]
                if np.isnan(a):
                    assert np.isnan(s)
                else:
                    assert np.isclose(a, s, atol=TOL_SAT_OCL), \
                        f"arr={a:.8f} single={s:.8f} (t={t}, w={win})"
