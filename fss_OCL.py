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

// Kernel 1b: Threshold to binary, but zeroed at missing points.
// `mask` is 1 at valid grid points, 0 where obs OR fcst is missing.  A NaN input
// already compares false, but the `&& mask` also removes points that are valid in
// this field yet missing in the other.
__kernel void threshold_binary_masked(
    __global const float *input, float threshold,
    __global const int *mask,
    __global int *output, int width, int height
) {
    int gid = get_global_id(0);
    if (gid >= width * height) return;
    output[gid] = (input[gid] > threshold && mask[gid]) ? 1 : 0;
}

// Kernel 4b: Missing-data fused box-filter + reduction.
// Like fss_box_reduce, but each window is normalised by its valid-point count
// C = win_area - (missing points in window), read from the invalid-mask SAT, and
// every contribution is divided by C and the window weighted by C:
//     num = sum((Sf-So)^2 / C),  den = sum((Sf^2+So^2) / C),  wsum = sum(C)
// (the sum(C) normaliser cancels in the FSS ratio).  Windows with C <= 0 drop out.
// Output layout (stride 3): [num_partial, den_partial, wsum_partial] per group.
__kernel void fss_box_reduce_masked(
    __global const long *sat_fcst, __global const long *sat_obs,
    __global const long *sat_invalid,
    __global float *partial_out,
    int width, int height, int win_half, float win_area,
    __local float *scratch            // size = 3 * local_size floats
) {
    int lid       = get_local_id(0);
    int gs        = get_local_size(0);
    int size      = width * height;

    float l_num = 0.0f, l_den = 0.0f, l_wsum = 0.0f;

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
        float cnt = (float)(
            sat_invalid[y1*width+x1] + sat_invalid[y0*width+x0]
          - sat_invalid[y0*width+x1] - sat_invalid[y1*width+x0]);

        float C = win_area - cnt;
        if (C > 0.5f) {
            float invC = 1.0f / C;
            float d = fhat - ohat;
            l_num  += d * d * invC;
            l_den  += (fhat * fhat + ohat * ohat) * invC;
            l_wsum += C;
        }
    }

    // Work-group tree reduction over three streams
    scratch[lid]          = l_num;
    scratch[lid + gs]     = l_den;
    scratch[lid + 2 * gs] = l_wsum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = gs >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            scratch[lid]          += scratch[lid + s];
            scratch[lid + gs]     += scratch[lid + gs + s];
            scratch[lid + 2 * gs] += scratch[lid + 2 * gs + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        int grp = get_group_id(0);
        partial_out[grp * 3]     = scratch[0];
        partial_out[grp * 3 + 1] = scratch[gs];
        partial_out[grp * 3 + 2] = scratch[2 * gs];
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


def _reduce_partials3(partial_host):
    """Sum the three masked-path work-group streams (num, den, wsum) in float64."""
    p64 = partial_host.astype(np.float64)
    return p64[0::3].sum(), p64[1::3].sum(), p64[2::3].sum()


def _validity_mask(fcst, obs):
    """Boolean mask of points valid in *both* fields (a point is missing if either
    forecast or observation is NaN there).  Mirrors fss_SAT / fss_FFT."""
    return ~(np.isnan(fcst) | np.isnan(obs))


def _fss_opencl_masked(fcst, obs, threshold, window, ctx, queue, program):
    """Missing-data FSS (single threshold+window) on the OpenCL backend.

    A grid point is missing where obs or fcst is NaN.  Binary exceedance fields are
    masked to zero at missing points, and each window is weighted by its valid-point
    count C = win_area - (missing points in window), giving the same weight =
    valid-count semantics as the SAT method.  Device math stays float32; with no
    missing data this reduces to the clean path.
    """
    h, w = fcst.shape
    size = h * w
    win_half = window // 2
    win_area = float((2 * win_half + 1) ** 2)

    mask = _validity_mask(fcst, obs)
    invalid = (~mask).astype(np.int32)
    mask_i = mask.astype(np.int32)

    mf = cl.mem_flags
    fcst_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fcst.astype(np.float32))
    obs_buf  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=obs.astype(np.float32))
    mask_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mask_i)
    inv_buf  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=invalid)
    bin_fcst = cl.Buffer(ctx, mf.READ_WRITE, size * 4)
    bin_obs  = cl.Buffer(ctx, mf.READ_WRITE, size * 4)
    sat_fcst = cl.Buffer(ctx, mf.READ_WRITE, size * 8)
    sat_obs  = cl.Buffer(ctx, mf.READ_WRITE, size * 8)
    sat_inv  = cl.Buffer(ctx, mf.READ_WRITE, size * 8)

    # Invalid-mask SAT (threshold-independent)
    program.sat_row_scan(queue, (h,), None, inv_buf, sat_inv, np.int32(w), np.int32(h))
    program.sat_col_scan(queue, (w,), None, sat_inv, np.int32(w), np.int32(h))

    # Masked threshold + SAT for forecast / observation
    program.threshold_binary_masked(queue, (size,), None,
        fcst_buf, np.float32(threshold), mask_buf, bin_fcst, np.int32(w), np.int32(h))
    program.threshold_binary_masked(queue, (size,), None,
        obs_buf,  np.float32(threshold), mask_buf, bin_obs,  np.int32(w), np.int32(h))
    program.sat_row_scan(queue, (h,), None, bin_fcst, sat_fcst, np.int32(w), np.int32(h))
    program.sat_col_scan(queue, (w,), None, sat_fcst, np.int32(w), np.int32(h))
    program.sat_row_scan(queue, (h,), None, bin_obs,  sat_obs,  np.int32(w), np.int32(h))
    program.sat_col_scan(queue, (w,), None, sat_obs,  np.int32(w), np.int32(h))

    ls = _LOCAL_SIZE
    n_groups = (size + ls - 1) // ls
    gs = n_groups * ls
    partial_buf  = cl.Buffer(ctx, mf.WRITE_ONLY, n_groups * 3 * 4)
    partial_host = np.empty(n_groups * 3, dtype=np.float32)

    program.fss_box_reduce_masked(
        queue, (gs,), (ls,),
        sat_fcst, sat_obs, sat_inv, partial_buf,
        np.int32(w), np.int32(h), np.int32(win_half), np.float32(win_area),
        cl.LocalMemory(ls * 3 * 4))
    cl.enqueue_copy(queue, partial_host, partial_buf)

    num_sum, den_sum, wsum = _reduce_partials3(partial_host)
    if wsum == 0.0:
        return 0.0, 0.0, np.nan
    num_mean = num_sum / wsum
    den_mean = den_sum / wsum
    if den_sum == 0.0:
        return num_mean, den_mean, np.nan
    return num_mean, den_mean, 1.0 - num_sum / den_sum


def fss_opencl(fcst, obs, threshold, window, ctx, queue, program):
    if not _validity_mask(fcst, obs).all():
        return _fss_opencl_masked(fcst, obs, threshold, window, ctx, queue, program)

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


def _fss_opencl_arr_masked(fcst, obs, thresholds, windows, ctx, queue, program):
    """Missing-data threshold x window sweep on the OpenCL backend.

    The invalid-mask SAT is threshold-independent and built once, reused across all
    thresholds (like the SAT method); the masked threshold + SAT are rebuilt per
    threshold.  See ``_fss_opencl_masked`` for the weighting semantics.
    """
    h, w = fcst.shape
    size = h * w

    mask = _validity_mask(fcst, obs)
    invalid = (~mask).astype(np.int32)
    mask_i = mask.astype(np.int32)

    mf = cl.mem_flags
    fcst_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fcst.astype(np.float32))
    obs_buf  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=obs.astype(np.float32))
    mask_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mask_i)
    inv_buf  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=invalid)
    bin_fcst = cl.Buffer(ctx, mf.READ_WRITE, size * 4)
    bin_obs  = cl.Buffer(ctx, mf.READ_WRITE, size * 4)
    sat_fcst = cl.Buffer(ctx, mf.READ_WRITE, size * 8)
    sat_obs  = cl.Buffer(ctx, mf.READ_WRITE, size * 8)
    sat_inv  = cl.Buffer(ctx, mf.READ_WRITE, size * 8)

    # Invalid-mask SAT: built once, reused across all thresholds
    program.sat_row_scan(queue, (h,), None, inv_buf, sat_inv, np.int32(w), np.int32(h))
    program.sat_col_scan(queue, (w,), None, sat_inv, np.int32(w), np.int32(h))

    ls = _LOCAL_SIZE
    n_groups = (size + ls - 1) // ls
    gs = n_groups * ls
    partial_buf  = cl.Buffer(ctx, mf.WRITE_ONLY, n_groups * 3 * 4)
    partial_host = np.empty(n_groups * 3, dtype=np.float32)

    num = np.zeros((thresholds.size, windows.size))
    den = np.zeros_like(num)
    fss_val = np.zeros_like(num)

    for ii, threshold in enumerate(thresholds):
        program.threshold_binary_masked(queue, (size,), None,
            fcst_buf, np.float32(threshold), mask_buf, bin_fcst, np.int32(w), np.int32(h))
        program.threshold_binary_masked(queue, (size,), None,
            obs_buf,  np.float32(threshold), mask_buf, bin_obs,  np.int32(w), np.int32(h))
        program.sat_row_scan(queue, (h,), None, bin_fcst, sat_fcst, np.int32(w), np.int32(h))
        program.sat_col_scan(queue, (w,), None, sat_fcst, np.int32(w), np.int32(h))
        program.sat_row_scan(queue, (h,), None, bin_obs,  sat_obs,  np.int32(w), np.int32(h))
        program.sat_col_scan(queue, (w,), None, sat_obs,  np.int32(w), np.int32(h))

        for jj, window in enumerate(windows):
            win_half = int(window) // 2
            win_area = float((2 * win_half + 1) ** 2)

            program.fss_box_reduce_masked(
                queue, (gs,), (ls,),
                sat_fcst, sat_obs, sat_inv, partial_buf,
                np.int32(w), np.int32(h), np.int32(win_half), np.float32(win_area),
                cl.LocalMemory(ls * 3 * 4))
            cl.enqueue_copy(queue, partial_host, partial_buf)

            num_sum, den_sum, wsum = _reduce_partials3(partial_host)
            if wsum == 0.0:
                num[ii, jj] = 0.0
                den[ii, jj] = 0.0
                fss_val[ii, jj] = np.nan
                continue
            num[ii, jj] = num_sum / wsum
            den[ii, jj] = den_sum / wsum
            if den_sum == 0.0:
                fss_val[ii, jj] = np.nan
            else:
                fss_val[ii, jj] = 1.0 - num_sum / den_sum

    return num, den, fss_val


def fss_opencl_arr(fcst, obs, thresholds, windows, ctx, queue, program):
    if not isinstance(thresholds, np.ndarray):
        thresholds = np.array(thresholds)
    if not isinstance(windows, np.ndarray):
        windows = np.array(windows)

    if not _validity_mask(fcst, obs).all():
        return _fss_opencl_arr_masked(fcst, obs, thresholds, windows, ctx, queue, program)

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
