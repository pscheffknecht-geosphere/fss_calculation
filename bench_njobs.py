"""Benchmark: effect of joblib n_jobs on SAT FSS performance.

Tests serial (n_jobs=1) vs parallel with various worker counts,
and also checks the effect of restricting BLAS threads to avoid
oversubscription.
"""
import numpy as np
import time
import os
import sys
sys.path.insert(0, '.')

# ── Generate test fields (601x601, seed=127) ──
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

print(f"Field: {fcst.shape[0]}x{fcst.shape[1]}")

# ── Detect CPU count ──
n_cpus = os.cpu_count()
print(f"CPUs: {n_cpus}")

# ── Show BLAS config ──
np.show_config()
print()

# ── Configs ──
configs = [
    ( "7t x 10w =    70",  7,  10),
    ("15t x 30w =   450", 15,  30),
    ("50t x 70w =  3500", 50,  70),
    ("100t x100w= 10000",100, 100),
]

# n_jobs values to test: 1, 2, 4, 8, ... up to 2*n_cpus, plus -1 (all CPUs)
njobs_list = [1]
v = 2
while v <= min(n_cpus, 64):
    njobs_list.append(v)
    v *= 2
if n_cpus not in njobs_list:
    njobs_list.append(n_cpus)
njobs_list.append(-1)  # joblib "all CPUs"

N_REPEATS = 3

def make_grid(n_thresh, n_win):
    thresholds = np.linspace(0.1, 3.0, n_thresh)
    windows = np.linspace(10, 500, n_win, dtype=int)
    windows = (windows // 2) * 2
    windows = np.unique(windows)
    return thresholds, windows


def bench(fcst, obs, thresholds, windows, n_jobs):
    from fss_SAT import fss_cumsum_parallel
    # warm-up (also triggers joblib pool creation)
    fss_cumsum_parallel(fcst, obs, thresholds[:1], windows[:1], n_jobs=n_jobs)
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        ret = fss_cumsum_parallel(fcst, obs, thresholds, windows, n_jobs=n_jobs)
        times.append(time.perf_counter() - t0)
    return min(times), ret[2]


# ── Run with default BLAS threads ──
blas_threads = os.environ.get('OMP_NUM_THREADS', 'default')
print(f"=== BLAS threads: {blas_threads} ===\n")

header = f"{'Config':<24} {'combos':>6}"
for nj in njobs_list:
    header += f" | {'n=' + str(nj):>8}"
header += " | best"
print(header)
print("-" * len(header))

for label, nt, nw in configs:
    thresholds, windows = make_grid(nt, nw)
    actual_combos = len(thresholds) * len(windows)

    row = f"{label:<24} {actual_combos:>6}"
    times = {}
    ref_fss = None
    for nj in njobs_list:
        t, fss_val = bench(fcst, obs, thresholds, windows, nj)
        times[nj] = t
        if ref_fss is None:
            ref_fss = fss_val
        else:
            diff = np.nanmax(np.abs(fss_val - ref_fss))
            if diff > 1e-10:
                row += f" DIFF={diff:.1e}"

    t_serial = times[1]
    for nj in njobs_list:
        t = times[nj]
        ratio = t_serial / t
        if ratio >= 1:
            row += f" | {t:>5.3f}s {ratio:>4.1f}x"
        else:
            row += f" | {t:>5.3f}s {ratio:>4.2f}"

    best_nj = min(times, key=times.get)
    best_t = times[best_nj]
    row += f" | n={best_nj} ({t_serial/best_t:.1f}x)"
    print(row)

# ── Now test with OMP_NUM_THREADS=1 ──
print(f"\n=== Retesting with OMP_NUM_THREADS=1 (single-threaded BLAS) ===\n")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Need to reimport numpy internals won't change, but joblib workers
# will inherit the env vars
header = f"{'Config':<24} {'combos':>6}"
for nj in njobs_list:
    header += f" | {'n=' + str(nj):>8}"
header += " | best"
print(header)
print("-" * len(header))

for label, nt, nw in configs:
    thresholds, windows = make_grid(nt, nw)
    actual_combos = len(thresholds) * len(windows)

    row = f"{label:<24} {actual_combos:>6}"
    times = {}
    for nj in njobs_list:
        t, _ = bench(fcst, obs, thresholds, windows, nj)
        times[nj] = t

    t_serial = times[1]
    for nj in njobs_list:
        t = times[nj]
        ratio = t_serial / t
        if ratio >= 1:
            row += f" | {t:>5.3f}s {ratio:>4.1f}x"
        else:
            row += f" | {t:>5.3f}s {ratio:>4.2f}"

    best_nj = min(times, key=times.get)
    best_t = times[best_nj]
    row += f" | n={best_nj} ({t_serial/best_t:.1f}x)"
    print(row)
