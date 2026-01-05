import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from run_bler_sc import ga_reliabilities_awgn, sc_decode, polar_transform

def main():
    N = 128
    K = 64
    R = K / N
    design_EbN0_dB = 2.0

    reli = ga_reliabilities_awgn(N, design_EbN0_dB, R)
    info_idx = np.argsort(reli)[-K:]
    info_idx = np.sort(info_idx)

    # IMPORTANT: many implementations expect FROZEN indices, not INFO indices
    all_idx = np.arange(N)
    frozen_idx = np.setdiff1d(all_idx, info_idx)

    rng = np.random.default_rng(0)
    u = rng.integers(0, 2, size=K, dtype=np.int8)

    # Encode using the SAME polar_transform convention as the main code
    u_full = np.zeros(N, dtype=np.int8)
    u_full[info_idx] = u
    x = polar_transform(u_full)

    # No-noise channel
    s = 1.0 - 2.0 * x.astype(np.float64)

    # Large but finite LLR
    sigma2 = 1e-2
    llr = 2.0 * s / sigma2

    # Decode (pass frozen_idx)
    uhat_full = sc_decode(llr, frozen_idx)

    # If sc_decode returns full-length uhat:
    if len(uhat_full) == N:
        uhat = np.array(uhat_full, dtype=np.int8)[info_idx]
    else:
        # If sc_decode already returns info bits (length K)
        uhat = np.array(uhat_full, dtype=np.int8)
        if len(uhat) != K:
            raise RuntimeError(f"Unexpected sc_decode output length: {len(uhat_full)}")

    ber = float(np.mean(u != uhat))
    bler = float(np.any(u != uhat))

    print("BER:", ber)
    print("BLER:", bler)

    assert ber == 0.0 and bler == 0.0, "No-noise test failed (frozen/info convention mismatch)"

if __name__ == "__main__":
    main()
