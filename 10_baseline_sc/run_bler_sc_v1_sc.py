import numpy as np
from datetime import datetime
import csv
import os

# =========================
# 0) Channel utilities
# =========================
def sigma2_from_ebn0(ebn0_db: float, R: float) -> float:
    ebn0 = 10 ** (ebn0_db / 10.0)
    # For BPSK with Es=1: sigma^2 = 1 / (2*R*Eb/N0)
    return 1.0 / (2.0 * R * ebn0)

def bpsk_mod(x_bits: np.ndarray) -> np.ndarray:
    # 0 -> +1, 1 -> -1
    return 1.0 - 2.0 * x_bits.astype(np.float64)

def awgn(s: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return s + rng.normal(0.0, sigma, size=s.shape)

def llr_awgn(y: np.ndarray, sigma2: float) -> np.ndarray:
    return 2.0 * y / sigma2

# =========================
# 1) Polar construction: Gaussian Approximation (GA)
# =========================
def _phi(x: float) -> float:
    return np.exp(-0.4527 * (x ** 0.86) + 0.0218)

def _phi_inv(y: float) -> float:
    # clamp to avoid log(0) and instability near 1
    y = np.clip(y, 1e-300, 1.0 - 1e-12)
    return ((-np.log(y) - 0.0218) / 0.4527) ** (1 / 0.86)

def ga_reliabilities_awgn(N: int, R: float, ebn0_db_design: float) -> np.ndarray:
    """
    Returns GA mean-LLR reliabilities in SC-tree order (permuted domain).
    """
    n = int(np.log2(N))
    assert 2**n == N, "N must be power of 2"
    ebn0 = 10 ** (ebn0_db_design / 10.0)
    sigma2 = 1.0 / (2.0 * R * ebn0)
    m0 = 2.0 / sigma2  # mean LLR for BPSK/AWGN
    m = np.array([m0], dtype=np.float64)

    for _ in range(n):
        m_next = np.empty(m.size * 2, dtype=np.float64)
        for i, mi in enumerate(m):
            m_next[2*i]   = _phi_inv(1 - (1 - _phi(mi))**2)  # "upper" channel
            m_next[2*i+1] = 2 * mi                           # "lower" channel
        m = m_next
    return m

def info_frozen_indices_permuted(N: int, K: int, R: float, ebn0_db_design: float):
    """
    Choose info/frozen indices directly in permuted domain (SC-tree order).
    """
    reliab = ga_reliabilities_awgn(N, R, ebn0_db_design)
    order = np.argsort(-reliab)        # descending reliability
    info = np.sort(order[:K])
    frozen = np.setdiff1d(np.arange(N), info)
    return frozen, info

# =========================
# 2) Polar transform (Arikan)
# =========================
def polar_transform(u: np.ndarray) -> np.ndarray:
    x = u.copy().astype(np.uint8)
    N = x.size
    step = 1
    while step < N:
        for i in range(0, N, 2*step):
            a = x[i:i+step]
            b = x[i+step:i+2*step]
            x[i:i+step] = a ^ b
            x[i+step:i+2*step] = b
        step *= 2
    return x

# =========================
# 3) SC decoder (correct, partial sums)
# =========================
def f_min_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Use sign(0)=+1 to avoid zero-sign issues
    sa = np.where(a >= 0, 1.0, -1.0)
    sb = np.where(b >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(a), np.abs(b))

def g_func(a: np.ndarray, b: np.ndarray, beta: np.ndarray) -> np.ndarray:
    # beta in {0,1}
    return b + (1.0 - 2.0*beta) * a

def sc_decode(llr: np.ndarray, frozen_set: set) -> np.ndarray:
    """
    Correct SC:
    - u_hat stored globally in permuted domain
    - recursion returns partial sums beta
    """
    N = llr.size
    u_hat = np.zeros(N, dtype=np.uint8)

    def recurse(alpha: np.ndarray, offset: int) -> np.ndarray:
        n = alpha.size
        if n == 1:
            i = offset
            if i in frozen_set:
                bit = 0
            else:
                bit = 0 if alpha[0] >= 0 else 1
            u_hat[i] = bit
            return np.array([bit], dtype=np.uint8)

        half = n // 2
        a = alpha[:half]
        b = alpha[half:]

        # left
        alpha_left = f_min_sum(a, b)
        beta_left = recurse(alpha_left, offset)

        # right (g uses beta_left)
        alpha_right = g_func(a, b, beta_left.astype(np.float64))
        beta_right = recurse(alpha_right, offset + half)

        # combine beta for parent
        beta = np.empty(n, dtype=np.uint8)
        beta[:half] = beta_left ^ beta_right
        beta[half:] = beta_right
        return beta

    recurse(llr, 0)
    return u_hat

# =========================
# 4) Runner
# =========================
def main():
    # ----- config -----
    N = 128
    R = 0.5
    K = int(N * R)

    design_EbN0 = 2.0
#    ebn0_grid = [0.0, 0.5, 1.0, 1.5, 2.0]
    ebn0_grid = [2.0]
    E_min = 20
    N_max = 2000
    """
    E_min = 200
    N_max = 50000
"""    
    seed = 0

    # quick debug switch
#    NO_NOISE_TEST = False   # set True to test y = s (should give BLER~0)
    NO_NOISE_TEST = True   

    # ----- setup -----
    rng = np.random.default_rng(seed)
    frozen_idx, info_idx = info_frozen_indices_permuted(N, K, R, design_EbN0)
    frozen_set = set(int(i) for i in frozen_idx)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("10_baseline_sc", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"run_{run_id}.txt")
    results_csv = os.path.join("10_baseline_sc", "results.csv")

    header = (
        f"[RUN] {run_id}\n"
        f"N={N}, K={K}, R={R}, design_EbN0={design_EbN0} dB, seed={seed}\n"
        f"Stopping rule: E_min={E_min} OR N_max={N_max}\n"
        f"Eb/N0 grid: {ebn0_grid}\n"
        f"NO_NOISE_TEST={NO_NOISE_TEST}\n\n"
    )

    print(header.strip())
    print("-"*60)

    with open(log_path, "w", encoding="utf-8") as f_log:
        f_log.write(header)

        for ebn0_db in ebn0_grid:
            sigma2 = sigma2_from_ebn0(ebn0_db, R)
            sigma = np.sqrt(sigma2)

            frames = 0
            block_err = 0
            bit_err = 0
            total_bits = 0

            while (block_err < E_min) and (frames < N_max):
                # info bits (K)
                u = rng.integers(0, 2, size=K, dtype=np.uint8)

                # build u in permuted domain directly
                u_perm = np.zeros(N, dtype=np.uint8)
                u_perm[info_idx] = u

                # encode
                x = polar_transform(u_perm)

                # channel
                s = bpsk_mod(x)
                y = s if NO_NOISE_TEST else awgn(s, sigma, rng)
                llr = llr_awgn(y, sigma2)

                # decode (permuted domain)
                uhat_perm = sc_decode(llr, frozen_set)
                uhat = uhat_perm[info_idx]   # extract info bits in same domain

                # metrics
                block_err += int(np.any(u != uhat))
                bit_err += int(np.sum(u != uhat))
                total_bits += K
                frames += 1

            BLER = block_err / frames
            BER = bit_err / total_bits

            line = f"Eb/N0={ebn0_db:>4.1f} dB | frames={frames:>6d} | BLER={BLER:.4e} | BER={BER:.4e}"
            print(line)
            f_log.write(line + "\n")

            # append to results.csv
            with open(results_csv, "a", newline="", encoding="utf-8") as f_csv:
                csv.writer(f_csv).writerow([
                    datetime.now().strftime("%Y-%m-%d"),
                    run_id,
                    N, K, R,
                    design_EbN0,
                    ebn0_db,
                    frames,
                    block_err,
                    f"{BLER:.6e}",
                    bit_err,
                    f"{BER:.6e}",
                    "SC(min-sum), GA-frozen, permuted-domain"
                ])

    print("-"*60)
    print(f"[Saved] Log: {log_path}")
    print(f"[Saved] Results: {results_csv}")

if __name__ == "__main__":
    main()

