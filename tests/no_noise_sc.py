import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from run_bler_sc import (ga_reliabilities_awgn, polar_encode, sc_decode)

def main():
    N = 128
    K = 64
    R = K / N
    design_EbN0_dB = 2.0

    reli = ga_reliabilities_awgn(N, design_EbN0_dB, R)
    info_idx = np.argsort(reli)[-K:]
    info_idx = np.sort(info_idx)

    # random info bits
    rng = np.random.default_rng(0)
    u = rng.integers(0, 2, size=K, dtype=np.int8)

    x = polar_encode(u, N, info_idx)

    # no noise => y = s
    s = 1.0 - 2.0 * x.astype(np.float64)
    sigma2 = 1e-12  # almost zero to create huge LLR magnitude
    llr = 2.0 * s / sigma2

    uhat_full = sc_decode(llr, info_idx)
    uhat = uhat_full[info_idx].astype(np.int8)

    ber = np.mean(u != uhat)
    bler = float(np.any(u != uhat))

    print("BER:", ber)
    print("BLER:", bler)
    assert ber == 0.0 and bler == 0.0, "No-noise test failed!"

if __name__ == "__main__":
    main()
