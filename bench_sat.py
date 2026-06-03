"""
Benchmark: original vs optimized SAT-based FSS.

Compares wall-clock time for computing FSS over different numbers of
(threshold x window) combinations on a 601x601 field.
"""
import numpy as np
import time
import sys, importlib

# ── Generate test fields ──
def make_gauss_bell(cx, cy, width, xgrid, ygrid, zgrid):
    zgrid += np.exp(-(np.square(xgrid - cx) / (2 * width**2) +
                      np.square(ygrid - cy) / (2 * width**2)))

x = np.arange(-3., 3.01, 0.01)
xx, yy = np.meshgrid(x, x)
fcst = np.zeros(xx.shape)
obs  = np.zeros(xx.shape)
np.random.seed(seed=127)
for _ in range(20):
    cx, cy = -2 + 4*np.random.rand(), -2 + 4*np.random.rand()
    _ = 2.2 + 1.2*np.random.rand(); s = 0.2 + 0.5*np.random.rand()
    make_gauss_bell(cx, cy, s, xx, yy, fcst)
    cx, cy = -2 + 4*np.random.rand(), -2 + 4*np.random.rand()
    _ = 2.2 + 1.2*np.random.rand(); s = 0.2 + 0.5*np.random.rand()
    make_gauss_bell(cx, cy, s, xx, yy, obs)

print(f"Field: {fcst.shape[0]}x{fcst.shape[1]}\n")

# ── Import original (from backup) and optimized (from fss_SAT) ──
sys.path.insert(0, '.')
import fss_SAT_original as orig
import fss_SAT as opt

# ── Benchmark configs ──
configs = [
    ("Small  (3t x 5w  =  15)", np.array([0.5, 1.0, 2.0]),        np.array([10, 30, 50, 90, 150])),
    ("Medium (7t x 10w =  70)", np.array([0.2, 0.5, 1., 1.5, 2., 3., 4.]),
                                np.arange(10, 200, 20, dtype=int)),
    ("Large  (7t x 20w = 140)", np.array([0.2, 0.5, 1., 1.5, 2., 3., 4.]),
                                np.arange(10, 400, 20, dtype=int)),
    ("XL    (15t x 30w = 450)", np.linspace(0.1, 3.0, 15),
                                np.arange(10, 600, 20, dtype=int)),
]

N_REPEATS = 3

def bench_cumsum_parallel(module, fcst, obs, thresholds, windows):
    """Time fss_cumsum_parallel and return (seconds, result)."""
    # Warm-up
    module.fss_cumsum_parallel(fcst, obs, thresholds[:1], windows[:1])
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        ret = module.fss_cumsum_parallel(fcst, obs, thresholds, windows)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times), ret

print(f"{'Config':<30} {'Original':>10} {'Optimized':>10} {'Speedup':>8}")
print("-" * 62)

for label, thresholds, windows in configs:
    t_orig, ret_orig = bench_cumsum_parallel(orig, fcst, obs, thresholds, windows)
    t_opt,  ret_opt  = bench_cumsum_parallel(opt,  fcst, obs, thresholds, windows)
    speedup = t_orig / t_opt

    # Verify results match
    fss_orig = ret_orig[2]
    fss_opt  = ret_opt[2]
    max_diff = np.nanmax(np.abs(fss_orig - fss_opt))

    print(f"{label:<30} {t_orig:>9.3f}s {t_opt:>9.3f}s {speedup:>7.2f}x  (max FSS diff: {max_diff:.1e})")

# ── Component-level timing ──
print(f"\n{'Component-level (single call, window=50)':}")

threshold = 1.0
window = 50
N = 50

# integral_filter
obs_bin_orig = orig.compute_integral_table((obs > threshold).astype(int))
obs_bin_opt  = opt.compute_integral_table((obs > threshold).astype(int))
mod_bin_opt  = opt.compute_integral_table((fcst > threshold).astype(int))

t0 = time.perf_counter()
for _ in range(N):
    orig.integral_filter(obs_bin_orig, window)
dt_filter_orig = (time.perf_counter() - t0) / N

t0 = time.perf_counter()
for _ in range(N):
    opt.integral_filter(obs_bin_opt, window)
dt_filter_opt = (time.perf_counter() - t0) / N

print(f"  integral_filter:  orig {dt_filter_orig*1000:.2f} ms  ->  opt {dt_filter_opt*1000:.2f} ms  ({dt_filter_orig/dt_filter_opt:.2f}x)")

# fss (filter + scoring)
t0 = time.perf_counter()
for _ in range(N):
    orig.fss(fcst, obs, window, mod_bin_opt, obs_bin_opt)
dt_fss_orig = (time.perf_counter() - t0) / N

t0 = time.perf_counter()
for _ in range(N):
    opt.fss(fcst, obs, window, mod_bin_opt, obs_bin_opt)
dt_fss_opt = (time.perf_counter() - t0) / N

print(f"  fss (full call):  orig {dt_fss_orig*1000:.2f} ms  ->  opt {dt_fss_opt*1000:.2f} ms  ({dt_fss_orig/dt_fss_opt:.2f}x)")
