"""Given a dataset, find the optimal threshold for quantizing it.
The reference distribution is `q`, and the candidate distribution is `p`.
`q` is a truncated version of the original distribution.
Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
"""
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import math
import sys 
from scipy import stats

def _smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor and taking the
    corresponding amount off the non-zero values.
    Ref: http://web.engr.illinois.edu/~hanj/cs412/bk3/KL-divergence.pdf
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist

def normalize_distrib(p) : 
    p /= sum(p)
    return p

def get_optimal_threshold(distrib_name, distrib) : 
    arr = distrib
    num_bins=16001
    num_quantized_bins=255

    # if isinstance(arr, NDArray):
    #     arr = arr.asnumpy()
    # elif isinstance(arr, list):
    #     assert len(arr) != 0
    #     for i, nd in enumerate(arr):
    #         if isinstance(nd, NDArray):
    #             arr[i] = nd.asnumpy()
    #         elif not isinstance(nd, np.ndarray):
    #             raise TypeError('get_optimal_threshold only supports input type of NDArray,'
    #                             ' list of np.ndarrays or NDArrays, and np.ndarray,'
    #                             ' while received type=%s' % (str(type(nd))))
    #     arr = np.concatenate(arr)
    # elif not isinstance(arr, np.ndarray):
    #     raise TypeError('get_optimal_threshold only supports input type of NDArray,'
    #                     ' list of NDArrays and np.ndarray,'
    #                     ' while received type=%s' % (str(type(arr))))
    min_val = np.min(arr)
    max_val = np.max(arr)
    th = max(abs(min_val), abs(max_val))
    print("Min Val : %f, Max Val : %f th : %f" %(min_val, max_val, th))
    hist, hist_edges = np.histogram(arr, bins=num_bins, range=(-th, th))

    plt.plot(hist_edges[:-1], hist)
    plt.ylabel('No of times')
    plt.show()

    zero_bin_idx = num_bins // 2                                             #4000
    num_half_quantized_bins = num_quantized_bins // 2                        #127
    assert np.allclose(hist_edges[zero_bin_idx] + hist_edges[zero_bin_idx + 1],
                       0, rtol=1e-5, atol=1e-7)

    thresholds = np.zeros(num_bins // 2 + 1 - num_quantized_bins // 2)
    divergence = np.zeros_like(thresholds)
    quantized_bins = np.zeros(num_quantized_bins, dtype=np.int32)

    for i in range(num_quantized_bins // 2, num_bins // 2 + 1):
        p_bin_idx_start = zero_bin_idx - i
        p_bin_idx_stop = zero_bin_idx + i + 1
        thresholds[i - num_half_quantized_bins] = hist_edges[p_bin_idx_stop]
        sliced_nd_hist = hist[p_bin_idx_start:p_bin_idx_stop]
#         print("i : %d start : %d, stop : %d threshold : %f hist size : %d" %(i, p_bin_idx_start,p_bin_idx_stop, hist_edges[p_bin_idx_stop], sliced_nd_hist.size))

        # generate reference distribution p
        p = sliced_nd_hist.copy()
        assert p.size % 2 == 1
        assert p.size >= num_quantized_bins

        left_outlier_count = np.sum(hist[0:p_bin_idx_start])
    #     print("Left Outlier Count : %d, Left : %d" %(left_outlier_count,p[0]))
        p[0] += left_outlier_count
        # put right outlier count in p[-1]
        right_outlier_count = np.sum(hist[p_bin_idx_stop:])
    #     print("Right Outlier Count : %d, Right : %d" %(right_outlier_count,p[-1]))
        p[-1] += right_outlier_count

        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (p != 0).astype(np.int32)
    #     print(is_nonzeros)
        # calculate how many bins should be merged to generate quantized distribution q
        num_merged_bins = sliced_nd_hist.size // num_quantized_bins

        # merge hist into num_quantized_bins bins
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[num_quantized_bins * num_merged_bins:].sum()
        # expand quantized_bins into p.size bins
        q = np.zeros(sliced_nd_hist.size, dtype=np.float32)
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            if j == num_quantized_bins - 1:
                stop = len(is_nonzeros)
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        q[p == 0] = 0

        p = _smooth_distribution(p)
        # There is a chance that q is an invalid probability distribution.
        try:
            q = _smooth_distribution(q)
        except ValueError:
            divergence[i - num_half_quantized_bins] = float("inf")
        
        p = normalize_distrib(p)
        q = normalize_distrib(q)
        divergence[i - num_half_quantized_bins] = stats.entropy(p, q, 2)
    #plt.figure(num=None, figsize=(70, 70), dpi=1000, facecolor='w', edgecolor='k')
#     file = open((distrib_name + '_divergence.txt'), "w")
#     file.writelines(["%s " % str(item)  for item in divergence])
#     file.close()
    
    plt.plot(np.log10(divergence))
    plt.ylabel('kl diverg')
    plt.savefig((distrib_name + '.png'))
    plt.show()

    min_divergence_idx = np.argmin(divergence)
    min_divergence = divergence[min_divergence_idx]
    opt_th = thresholds[min_divergence_idx]
    print("min val : %f max_val : %f, last div : %f min_div : %f, opt_th : %f " %(min_val, max_val, divergence[-1], min_divergence, opt_th))
    return [opt_th, min_val, max_val]

# def clip_values(x, max) : 
#     indices = np.argwhere(x > max)
#     print(indices)
#     # x[indices] = max
#     return x

def quantize_vector(x, threshold) : 
    scale_factor = threshold/255.0
    # print("scale factor : ", scale_factor)
    # print("original x : ", x[0:5])
    x_quant = x/scale_factor
    # print("x_quant before clip: ", x_quant)
    x_quant = np.clip(x_quant, a_min = 0, a_max = 255)
    # print("x_quant after clip: ", x_quant)
    x_quant = np.round(x_quant)
    x_quant = x_quant.astype('uint8')
    # print("x_quant after int8 conversion: ", x_quant[0:5])
    return x_quant

def quantize_vector_pow2(x, threshold) : 
    scale_factor = threshold/255.0
    # print("scale factor : ", scale_factor)
    # print("original x : ", x[0:5])
    x_quant = x/scale_factor
    # print("x_quant before clip: ", x_quant)
    x_quant = np.clip(x_quant, a_min = 0, a_max = 255)
    # print("x_quant after clip: ", x_quant)
    x_quant = np.round(x_quant)
    x_quant = x_quant.astype('int16')
    # print("x_quant after int8 conversion: ", x_quant[0:5])
    return x_quant

def dequantize_vector(q, threshold) : 
    scale_factor = threshold/255.0
    # print("threshold : ", threshold, " scale_factor : ", scale_factor)
    # print("before dequant : ", q)
    x = q*scale_factor
    # print("after dequant : ", x)

    # raw_input(" asdasd")
    return x

def dequantize_vector_pow2(q, threshold) : 
    # scale_factor = threshold/256.0
    scale_factor_reciprocal = (1/threshold) * 256
    scale_factor_exp        = math.log(scale_factor_reciprocal, 2) - 1
    scale_factor_exp_int    = np.uint8(scale_factor_exp)
    # print("sclae factor exp : ", scale_factor_exp, " sclae factor exp int : ", scale_factor_exp_int)
    q_np = np.array(q)
    x = np.right_shift(q_np, scale_factor_exp_int)
    x = np.round(x/2.0)
    x = x.astype('int16')
    # print("q : ", q_np[0:10], " q : ", x[0:10])
    # raw_input()
    # print("threshold : ", threshold, " scale_factor : ", scale_factor)
    # print("before dequant : ", q)
    # x = q*scale_factor
    # print("after dequant : ", x)

    # raw_input(" asdasd")
    return x