"""Benchmark: original vs optimized OpenCL FSS."""
import numpy as np
import time
import sys
sys.path.insert(0, '.')

# Generate test fields
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

import fss_OCL_original as orig
import fss_OCL as opt

configs = [
    ("Small  (3t x 5w  =  15)", np.array([0.5, 1.0, 2.0]),
                                 np.array([10, 30, 50, 90, 150])),
    ("Medium (7t x 10w =  70)", np.array([0.2, 0.5, 1., 1.5, 2., 3., 4.]),
                                 np.arange(10, 200, 20, dtype=int)),
    ("Large  (7t x 20w = 140)", np.array([0.2, 0.5, 1., 1.5, 2., 3., 4.]),
                                 np.arange(10, 400, 20, dtype=int)),
    ("XL    (15t x 30w = 450)", np.linspace(0.1, 3.0, 15),
                                 np.arange(10, 600, 20, dtype=int)),
]

N_REPEATS = 3

def bench(mod, fcst, obs, thresholds, windows):
    # warm-up
    mod.fss_opencl_arr(fcst, obs, thresholds[:1], windows[:1],
                       mod.ctx, mod.queue, mod.program)
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        ret = mod.fss_opencl_arr(fcst, obs, thresholds, windows,
                                  mod.ctx, mod.queue, mod.program)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times), ret

print(f"{'Config':<30} {'Original':>10} {'Optimized':>10} {'Speedup':>8}")
print("-" * 62)

for label, thresholds, windows in configs:
    t_orig, ret_orig = bench(orig, fcst, obs, thresholds, windows)
    t_opt,  ret_opt  = bench(opt,  fcst, obs, thresholds, windows)
    speedup = t_orig / t_opt

    fss_o = ret_orig[2]
    fss_n = ret_opt[2]
    max_diff = np.nanmax(np.abs(fss_o - fss_n))

    print(f"{label:<30} {t_orig:>9.3f}s {t_opt:>9.3f}s {speedup:>7.2f}x  (max FSS diff: {max_diff:.1e})")

# Single-call comparison
print(f"\nSingle-call fss_opencl (window=50, threshold=1.0):")
N = 50
# warm-up
orig.fss_opencl(fcst, obs, 1.0, 50, orig.ctx, orig.queue, orig.program)
opt.fss_opencl(fcst, obs, 1.0, 50, opt.ctx, opt.queue, opt.program)
t0 = time.perf_counter()
for _ in range(N):
    orig.fss_opencl(fcst, obs, 1.0, 50, orig.ctx, orig.queue, orig.program)
dt_orig = (time.perf_counter() - t0) / N
t0 = time.perf_counter()
for _ in range(N):
    opt.fss_opencl(fcst, obs, 1.0, 50, opt.ctx, opt.queue, opt.program)
dt_opt = (time.perf_counter() - t0) / N
print(f"  orig: {dt_orig*1000:.2f} ms  ->  opt: {dt_opt*1000:.2f} ms  ({dt_orig/dt_opt:.2f}x)")
