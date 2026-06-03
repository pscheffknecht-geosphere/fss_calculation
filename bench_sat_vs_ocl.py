"""Benchmark: optimized SAT (numpy) vs optimized OpenCL FSS across grid sizes.

Also reports per-component OCL timings to diagnose scaling behaviour
on different hardware (few CUs vs many CUs).
"""
import numpy as np
import time
import os
import sys
sys.path.insert(0, '.')

# Suppress POCL interactive prompt
if 'PYOPENCL_CTX' not in os.environ:
    os.environ['PYOPENCL_CTX'] = '0'

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

# ── Show BLAS config ──
np.show_config()
print()

# ── Import both implementations ──
import pyopencl as cl
from fss_SAT import fss_cumsum_parallel
from fss_OCL import fss_opencl_arr, ctx, queue, program, _LOCAL_SIZE

dev = ctx.devices[0]
print(f"OpenCL device: {dev.name}  ({dev.max_compute_units} CUs, local_size={_LOCAL_SIZE})")
print()

# ── Configs ──
configs = [
    ( "3t x  5w =    15",  3,   5),
    ( "7t x 10w =    70",  7,  10),
    ( "7t x 20w =   140",  7,  20),
    ("15t x 30w =   450", 15,  30),
    ("30t x 50w =  1500", 30,  50),
    ("50t x 70w =  3500", 50,  70),
    ("100t x100w= 10000",100, 100),
]

N_REPEATS = 3

def make_grid(n_thresh, n_win):
    thresholds = np.linspace(0.1, 3.0, n_thresh)
    windows = np.linspace(10, 500, n_win, dtype=int)
    windows = (windows // 2) * 2
    windows = np.unique(windows)
    return thresholds, windows

def bench_sat(fcst, obs, thresholds, windows):
    fss_cumsum_parallel(fcst, obs, thresholds[:1], windows[:1])
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        ret = fss_cumsum_parallel(fcst, obs, thresholds, windows)
        times.append(time.perf_counter() - t0)
    return min(times), ret[2]

def bench_ocl(fcst, obs, thresholds, windows):
    fss_opencl_arr(fcst, obs, thresholds[:1], windows[:1], ctx, queue, program)
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        ret = fss_opencl_arr(fcst, obs, thresholds, windows, ctx, queue, program)
        times.append(time.perf_counter() - t0)
    return min(times), ret[2]


# ── Component-level OCL timing ──
print("OCL component timing (single threshold, 10 windows):")
h, w = fcst.shape
size = h * w
mf = cl.mem_flags
fcst_f32 = fcst.astype(np.float32)
obs_f32  = obs.astype(np.float32)
fcst_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fcst_f32)
obs_buf  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=obs_f32)
bin_f = cl.Buffer(ctx, mf.READ_WRITE, size * 4)
bin_o = cl.Buffer(ctx, mf.READ_WRITE, size * 4)
sat_f = cl.Buffer(ctx, mf.READ_WRITE, size * 8)
sat_o = cl.Buffer(ctx, mf.READ_WRITE, size * 8)
ls = _LOCAL_SIZE
n_groups = (size + ls - 1) // ls
gs = n_groups * ls
partial_buf  = cl.Buffer(ctx, mf.WRITE_ONLY, n_groups * 2 * 4)
partial_host = np.empty(n_groups * 2, dtype=np.float32)

N = 30
threshold = np.float32(1.0)

# Threshold kernels
queue.finish()
t0 = time.perf_counter()
for _ in range(N):
    program.threshold_binary(queue, (size,), None, fcst_buf, threshold, bin_f, np.int32(w), np.int32(h))
    program.threshold_binary(queue, (size,), None, obs_buf,  threshold, bin_o, np.int32(w), np.int32(h))
    queue.finish()
dt_thresh = (time.perf_counter() - t0) / N

# SAT kernels
queue.finish()
t0 = time.perf_counter()
for _ in range(N):
    program.sat_row_scan(queue, (h,), None, bin_f, sat_f, np.int32(w), np.int32(h))
    program.sat_col_scan(queue, (w,), None, sat_f, np.int32(w), np.int32(h))
    program.sat_row_scan(queue, (h,), None, bin_o, sat_o, np.int32(w), np.int32(h))
    program.sat_col_scan(queue, (w,), None, sat_o, np.int32(w), np.int32(h))
    queue.finish()
dt_sat = (time.perf_counter() - t0) / N

# Fused box-filter + reduce
queue.finish()
t0 = time.perf_counter()
for _ in range(N):
    program.fss_box_reduce(
        queue, (gs,), (ls,), sat_f, sat_o, partial_buf,
        np.int32(w), np.int32(h), np.int32(25),
        cl.LocalMemory(ls * 2 * 4))
    cl.enqueue_copy(queue, partial_host, partial_buf)
    queue.finish()
dt_reduce = (time.perf_counter() - t0) / N

print(f"  threshold (x2):          {dt_thresh*1000:6.2f} ms")
print(f"  SAT row+col (x2):       {dt_sat*1000:6.2f} ms")
print(f"  fss_box_reduce + read:   {dt_reduce*1000:6.2f} ms")
print(f"  per-threshold total:     {(dt_thresh+dt_sat)*1000:6.2f} ms  (amortised over all windows)")
print(f"  per-window total:        {dt_reduce*1000:6.2f} ms")
print()

# ── Main comparison ──
print(f"{'Config':<24} {'combos':>6} | {'SAT':>9} {'OCL':>9} {'ratio':>9} | {'ms/combo (SAT/OCL)':>20} {'max|Δ|':>9}")
print("-" * 100)

for label, nt, nw in configs:
    thresholds, windows = make_grid(nt, nw)
    actual_combos = len(thresholds) * len(windows)

    t_sat, fss_sat = bench_sat(fcst, obs, thresholds, windows)
    t_ocl, fss_ocl = bench_ocl(fcst, obs, thresholds, windows)

    ratio = t_sat / t_ocl
    if ratio >= 1:
        ratio_str = f"OCL {ratio:.2f}x"
    else:
        ratio_str = f"SAT {1/ratio:.2f}x"

    max_diff = np.nanmax(np.abs(fss_sat - fss_ocl))
    ms_sat = t_sat * 1000 / actual_combos
    ms_ocl = t_ocl * 1000 / actual_combos

    print(f"{label:<24} {actual_combos:>6} | {t_sat:>8.3f}s {t_ocl:>8.3f}s  {ratio_str:>8} | {ms_sat:>7.3f} / {ms_ocl:>7.3f}    {max_diff:>9.1e}")
