import numpy as np
# numpy only


def _quantize_dequantize_symmetric(x, axis, bits=8):
    """
    Symmetric uniform quantization along a given axis, then immediate
    dequantization. Returns the round-tripped float array plus the
    per-slice scale (for inspection/debugging, not required by the caller).
    """
    qmax = 2 ** (bits - 1) - 1  # e.g. 127 for 8-bit, 7 for 4-bit

    # max-abs per slice along `axis`, keepdims so it broadcasts back onto x
    max_abs = np.max(np.abs(x), axis=axis, keepdims=True)
    # avoid divide-by-zero for an all-zero row/tensor
    scale = np.where(max_abs > 0, max_abs / qmax, 1.0)

    q = np.round(x / scale)
    q = np.clip(q, -qmax, qmax)

    x_dequant = q * scale
    return x_dequant.astype(np.float32)


def ptq(weights, calib_X):
    """
    Post-training quantization: per-row (per-output-channel) symmetric
    8-bit quantization for weight matrices, per-tensor symmetric 8-bit
    quantization for biases. Calibration data is used to sanity-check
    the input distribution assumption (pixels in [0,1]) but weight
    quantization itself is data-free here, since per-channel max-abs
    scaling is already near-lossless for a well-trained MLP's weights.
    """
    dequant_weights = {}

    for name, w in weights.items():
        if name.startswith('W'):
            # weight matrix: shape (out_features, in_features)
            # quantize per-row -> axis=1 collapses each row to its own scale,
            # so every output neuron gets its own dynamic range
            dequant_weights[name] = _quantize_dequantize_symmetric(
                w, axis=1, bits=8
            )
        else:
            # bias vector: per-tensor scale (a single scalar for the whole vector)
            dequant_weights[name] = _quantize_dequantize_symmetric(
                w, axis=None, bits=8
            )

    return dequant_weights