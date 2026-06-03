import pyopencl as cl
import numpy as np

kernel_code = """
// Kernel 1: Threshold to binary
__kernel void threshold_binary(
    __global const float *input, float threshold,
    __global int *output, int width, int height
) {
    int gid = get_global_id(0);
    if (gid >= width * height) return;
    output[gid] = input[gid] > threshold ? 1 : 0;
}

// Kernel 2: SAT row prefix sum
__kernel void sat_row_scan(
    __global int *image, __global long *sat, int width, int height
) {
    int y = get_global_id(0);
    if (y >= height) return;
    long sum = 0;
    for (int x = 0; x < width; ++x) {
        int idx = y * width + x;
        sum += image[idx];
        sat[idx] = sum;
    }
}

// Kernel 3: SAT column prefix sum
__kernel void sat_col_scan(__global long *sat, int width, int height) {
    int x = get_global_id(0);
    if (x >= width) return;
    long sum = 0;
    for (int y = 0; y < height; ++y) {
        int idx = y * width + x;
        sum += sat[idx];
        sat[idx] = sum;
    }
}

// Kernel 4: Fused box-filter + parallel reduction.
// Reads directly from the two SATs, computes the per-pixel FSS
// numerator and denominator contributions, and accumulates
// work-group partial sums into partial_out.
// Output layout:  partial_out[group_id * 2]     = partial sum of numerator
//                 partial_out[group_id * 2 + 1] = partial sum of denominator
__kernel void fss_box_reduce(
    __global const long *sat_fcst, __global const long *sat_obs,
    __global float *partial_out,
    int width, int height, int win_half,
    __local float *scratch            // size = 2 * local_size floats
) {
    int lid       = get_local_id(0);
    int gs        = get_local_size(0);
    int size      = width * height;

    float l_num = 0.0f, l_den = 0.0f;

    // Grid-stride loop: each work-item may process several pixels.
    for (int gid = get_global_id(0); gid < size; gid += get_global_size(0)) {
        int x = gid % width;
        int y = gid / width;
        int x0 = clamp(x - win_half, 0, width  - 1);
        int x1 = clamp(x + win_half, 0, width  - 1);
        int y0 = clamp(y - win_half, 0, height - 1);
        int y1 = clamp(y + win_half, 0, height - 1);

        float fhat = (float)(
            sat_fcst[y1*width+x1] + sat_fcst[y0*width+x0]
          - sat_fcst[y0*width+x1] - sat_fcst[y1*width+x0]);
        float ohat = (float)(
            sat_obs[y1*width+x1] + sat_obs[y0*width+x0]
          - sat_obs[y0*width+x1] - sat_obs[y1*width+x0]);

        float d = fhat - ohat;
        l_num += d * d;
        l_den += fhat * fhat + ohat * ohat;
    }

    // Work-group tree reduction
    scratch[lid]      = l_num;
    scratch[lid + gs] = l_den;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = gs >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            scratch[lid]      += scratch[lid + s];
            scratch[lid + gs] += scratch[lid + gs + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        int grp = get_group_id(0);
        partial_out[grp * 2]     = scratch[0];
        partial_out[grp * 2 + 1] = scratch[gs];
    }
}

// Legacy kernel kept for backward compatibility
__kernel void sliding_sum_from_sat(
    __global const long *sat, __global float *output,
    int width, int height, int win_half
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int x0 = clamp(x - win_half, 0, width - 1);
    int x1 = clamp(x + win_half, 0, width - 1);
    int y0 = clamp(y - win_half, 0, height - 1);
    int y1 = clamp(y + win_half, 0, height - 1);

    long A = sat[y0 * width + x0];
    long B = sat[y0 * width + x1];
    long C = sat[y1 * width + x0];
    long D = sat[y1 * width + x1];

    output[y * width + x] = (float)(D + A - B - C);
}

// Legacy kernel kept for backward compatibility
__kernel void compute_fss_scores(
    __global const float *a, __global const float *b,
    __global float *numerator, __global float *denominator,
    int size
) {
    int gid = get_global_id(0);
    if (gid >= size) return;
    float diff = a[gid] - b[gid];
    numerator[gid]   = diff * diff;
    denominator[gid] = a[gid] * a[gid] + b[gid] * b[gid];
}
"""

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
program = cl.Program(ctx, kernel_code).build()

# Pick a power-of-2 local size that fits the device
_dev = ctx.devices[0]
_LOCAL_SIZE = 256
while _LOCAL_SIZE > _dev.max_work_group_size:
    _LOCAL_SIZE //= 2


def _reduce_partials(partial_host):
    """Sum work-group partial sums (float32) in float64 for precision."""
    p64 = partial_host.astype(np.float64)
    return p64[0::2].sum(), p64[1::2].sum()


def fss_opencl(fcst, obs, threshold, window, ctx, queue, program):
    h, w = fcst.shape
    size = h * w
    win_half = window // 2

    mf = cl.mem_flags
    fcst_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fcst.astype(np.float32))
    obs_buf  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=obs.astype(np.float32))
    bin_fcst = cl.Buffer(ctx, mf.READ_WRITE, size * 4)
    bin_obs  = cl.Buffer(ctx, mf.READ_WRITE, size * 4)
    sat_fcst = cl.Buffer(ctx, mf.READ_WRITE, size * 8)
    sat_obs  = cl.Buffer(ctx, mf.READ_WRITE, size * 8)

    # Threshold
    program.threshold_binary(queue, (size,), None, fcst_buf, np.float32(threshold), bin_fcst, np.int32(w), np.int32(h))
    program.threshold_binary(queue, (size,), None, obs_buf,  np.float32(threshold), bin_obs,  np.int32(w), np.int32(h))

    # SAT
    program.sat_row_scan(queue, (h,), None, bin_fcst, sat_fcst, np.int32(w), np.int32(h))
    program.sat_col_scan(queue, (w,), None, sat_fcst, np.int32(w), np.int32(h))
    program.sat_row_scan(queue, (h,), None, bin_obs,  sat_obs,  np.int32(w), np.int32(h))
    program.sat_col_scan(queue, (w,), None, sat_obs,  np.int32(w), np.int32(h))

    # Fused box-filter + reduction
    ls = _LOCAL_SIZE
    n_groups = (size + ls - 1) // ls
    gs = n_groups * ls
    partial_buf  = cl.Buffer(ctx, mf.WRITE_ONLY, n_groups * 2 * 4)
    partial_host = np.empty(n_groups * 2, dtype=np.float32)

    program.fss_box_reduce(
        queue, (gs,), (ls,),
        sat_fcst, sat_obs, partial_buf,
        np.int32(w), np.int32(h), np.int32(win_half),
        cl.LocalMemory(ls * 2 * 4))
    cl.enqueue_copy(queue, partial_host, partial_buf)

    num_sum, den_sum = _reduce_partials(partial_host)
    num_mean = num_sum / size
    den_mean = den_sum / size
    if den_sum == 0.0:
        return num_mean, den_mean, np.nan
    return num_mean, den_mean, 1.0 - num_sum / den_sum


def fss_opencl_arr(fcst, obs, thresholds, windows, ctx, queue, program):
    if not isinstance(thresholds, np.ndarray):
        thresholds = np.array(thresholds)
    if not isinstance(windows, np.ndarray):
        windows = np.array(windows)

    h, w = fcst.shape
    size = h * w

    mf = cl.mem_flags

    # Allocate all device buffers once
    fcst_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fcst.astype(np.float32))
    obs_buf  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=obs.astype(np.float32))
    bin_fcst = cl.Buffer(ctx, mf.READ_WRITE, size * 4)
    bin_obs  = cl.Buffer(ctx, mf.READ_WRITE, size * 4)
    sat_fcst = cl.Buffer(ctx, mf.READ_WRITE, size * 8)
    sat_obs  = cl.Buffer(ctx, mf.READ_WRITE, size * 8)

    ls = _LOCAL_SIZE
    n_groups = (size + ls - 1) // ls
    gs = n_groups * ls
    partial_buf  = cl.Buffer(ctx, mf.WRITE_ONLY, n_groups * 2 * 4)
    partial_host = np.empty(n_groups * 2, dtype=np.float32)

    num = np.zeros((thresholds.size, windows.size))
    den = np.zeros_like(num)
    fss_val = np.zeros_like(num)

    for ii, threshold in enumerate(thresholds):
        # Threshold + SAT: once per threshold, reused across all windows
        program.threshold_binary(queue, (size,), None,
            fcst_buf, np.float32(threshold), bin_fcst, np.int32(w), np.int32(h))
        program.threshold_binary(queue, (size,), None,
            obs_buf,  np.float32(threshold), bin_obs,  np.int32(w), np.int32(h))

        program.sat_row_scan(queue, (h,), None, bin_fcst, sat_fcst, np.int32(w), np.int32(h))
        program.sat_col_scan(queue, (w,), None, sat_fcst, np.int32(w), np.int32(h))
        program.sat_row_scan(queue, (h,), None, bin_obs,  sat_obs,  np.int32(w), np.int32(h))
        program.sat_col_scan(queue, (w,), None, sat_obs,  np.int32(w), np.int32(h))

        for jj, window in enumerate(windows):
            win_half = int(window) // 2

            program.fss_box_reduce(
                queue, (gs,), (ls,),
                sat_fcst, sat_obs, partial_buf,
                np.int32(w), np.int32(h), np.int32(win_half),
                cl.LocalMemory(ls * 2 * 4))
            cl.enqueue_copy(queue, partial_host, partial_buf)

            num_sum, den_sum = _reduce_partials(partial_host)
            num[ii, jj] = num_sum / size
            den[ii, jj] = den_sum / size
            if den_sum == 0.0:
                fss_val[ii, jj] = np.nan
            else:
                fss_val[ii, jj] = 1.0 - num_sum / den_sum

    return num, den, fss_val
