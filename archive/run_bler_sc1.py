import numpy as np
from datetime import datetime
import csv
import os

def sigma2_from_ebn0(ebn0_db: float, R: float) -> float:
    ebn0 = 10 ** (ebn0_db / 10.0)
    return 1.0 / (2.0 * R * ebn0)

def bpsk_mod(x_bits: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * x_bits.astype(float)

def awgn(s: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return s + rng.normal(0.0, sigma, size=s.shape)

def llr_awgn(y: np.ndarray, sigma2: float) -> np.ndarray:
    return 2.0 * y / sigma2

# ---- GA construction ----
def _phi(x: float) -> float:
    return np.exp(-0.4527 * (x ** 0.86) + 0.0218)

def _phi_inv(y: float) -> float:
    y = np.clip(y, 1e-300, 1 - 1e-12)
    return ((-np.log(y) - 0.0218) / 0.4527) ** (1 / 0.86)

def ga_reliabilities_awgn(N: int, R: float, ebn0_db_design: float) -> np.ndarray:
    n = int(np.log2(N))
    assert 2**n == N, "N must be power of 2"
    ebn0 = 10 ** (ebn0_db_design / 10.0)
    sigma2 = 1.0 / (2.0 * R * ebn0)
    m0 = 2.0 / sigma2
    m = np.array([m0], dtype=float)
    for _ in range(n):
        m_next = np.empty(m.size * 2, dtype=float)
        for i, mi in enumerate(m):
            m_next[2*i]   = _phi_inv(1 - (1 - _phi(mi))**2)
            m_next[2*i+1] = 2 * mi
        m = m_next
    return m

"""
def frozen_and_info_indices(N: int, K: int, R: float, ebn0_db_design: float = 2.0):
    reliab = ga_reliabilities_awgn(N, R, ebn0_db_design)
    order = np.argsort(-reliab)
    info = np.sort(order[:K])
    frozen = np.setdiff1d(np.arange(N), info)
    return frozen, info
"""
def frozen_and_info_indices_permuted(N: int, K: int, R: float, ebn0_db_design: float, br: np.ndarray):
    reliab = ga_reliabilities_awgn(N, R, ebn0_db_design)  # reliab in SC-tree order
    order = np.argsort(-reliab)
    info_perm = np.sort(order[:K])   # choose info in permuted domain
    frozen_perm = np.setdiff1d(np.arange(N), info_perm)
    # map back to original domain (optional)
    info = br[info_perm]
    frozen = br[frozen_perm]
    return frozen, info, frozen_perm, info_perm



# ---- Polar encoder ----
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

def polar_encode(u_info: np.ndarray, N: int, info_idx: np.ndarray) -> np.ndarray:
    x = np.zeros(N, dtype=np.uint8)
    x[info_idx] = u_info.astype(np.uint8)
    return polar_transform(x)

#    u_full = np.zeros(N, dtype=np.uint8)
#    u_full[info_idx] = u
#    u_perm = u_full[br]              # apply B_N: u_perm[i] = u_full[bitrev(i)]
#    x = polar_transform(u_perm)


# ---- SC decoder (min-sum) ----
def f_min_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))

def g_func(a: np.ndarray, b: np.ndarray, u: np.ndarray) -> np.ndarray:
    return b + (1.0 - 2.0*u) * a

"""
def sc_decode(llr: np.ndarray, frozen_set: set) -> np.ndarray:
    def recurse(alpha: np.ndarray, offset: int) -> np.ndarray:
        n = alpha.size
        if n == 1:
            i = offset
            if i in frozen_set:
                return np.array([0], dtype=np.uint8)
            return np.array([0 if alpha[0] >= 0 else 1], dtype=np.uint8)

        half = n // 2
        a = alpha[:half]
        b = alpha[half:]

        alpha_left = f_min_sum(a, b)
        u_left = recurse(alpha_left, offset)

        alpha_right = g_func(a, b, u_left.astype(float))
        u_right = recurse(alpha_right, offset + half)

        u = np.empty(n, dtype=np.uint8)
        u[:half] = u_left
        u[half:] = u_right
        return u

    return recurse(llr, 0)
"""

def sc_decode(llr: np.ndarray, frozen_set: set) -> np.ndarray:
    """
    Correct SC decoder:
    - Stores u_hat decisions globally
    - Recursion returns partial sums beta for upper layers
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
            return np.array([bit], dtype=np.uint8)  # beta at leaf

        half = n // 2
        a = alpha[:half]
        b = alpha[half:]

        # Left
        alpha_left = f_min_sum(a, b)
        beta_left = recurse(alpha_left, offset)

        # Right (IMPORTANT: g uses beta_left partial sums)
        alpha_right = g_func(a, b, beta_left.astype(float))
        beta_right = recurse(alpha_right, offset + half)

        # Combine partial sums beta for parent
        beta = np.empty(n, dtype=np.uint8)
        beta[:half] = beta_left ^ beta_right
        beta[half:] = beta_right
        return beta

    recurse(llr, 0)
    return u_hat


def bit_reverse_permutation(N: int) -> np.ndarray:
    n = int(np.log2(N))
    assert 2**n == N
    br = np.zeros(N, dtype=int)
    for i in range(N):
        b = format(i, f"0{n}b")
        br[i] = int(b[::-1], 2)
    return br


def main():
    N = 128
    
    br = bit_reverse_permutation(N)
    inv_br = np.zeros(N, dtype=int)
    inv_br[br] = np.arange(N)

    R = 0.5
    K = int(N * R)

    design_EbN0 = 2.0
#    ebn0_grid = [0.0, 0.5, 1.0, 1.5, 2.0]
    ebn0_grid = [2.0]
#    E_min = 200
    E_min = 20
#    N_max = 50000
    N_max = 2000
    seed = 0

    rng = np.random.default_rng(seed)
#    frozen_idx, info_idx = frozen_and_info_indices(N, K, R, ebn0_db_design=design_EbN0)
    frozen_idx, info_idx, frozen_perm_idx, info_perm_idx = frozen_and_info_indices_permuted(
    N, K, R, design_EbN0, br
)

#    frozen_set = set(int(i) for i in frozen_idx)
    frozen_perm = inv_br[frozen_idx]          # map frozen positions to permuted domain
#    frozen_set = set(int(i) for i in frozen_perm)
    frozen_set = set(int(i) for i in frozen_perm_idx)


    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("10_baseline_sc", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"run_{run_id}.txt")
    results_csv = os.path.join("10_baseline_sc", "results.csv")

    header = f"[RUN] {run_id}\nN={N}, K={K}, R={R}, design_EbN0={design_EbN0} dB, seed={seed}\n"
    header += f"Stopping rule: E_min={E_min} OR N_max={N_max}\nEb/N0 grid: {ebn0_grid}\n\n"

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
#                u = rng.integers(0, 2, size=K, dtype=np.uint8)
#                x = polar_encode(u, N, info_idx)
#                s = bpsk_mod(x)
                u = rng.integers(0, 2, size=K, dtype=np.uint8)

                # build full u vector
                u_full = np.zeros(N, dtype=np.uint8)
                u_full[info_idx] = u

                # apply bit-reversal then transform
                u_perm = u_full[br]
                x = polar_transform(u_perm)

                s = bpsk_mod(x)

#                y = awgn(s, sigma, rng)
                y = s
                llr = llr_awgn(y, sigma2)

#                uhat_full = sc_decode(llr, frozen_set)
#                uhat = uhat_full[info_idx]
                uhat_perm = sc_decode(llr, frozen_set)    # estimates in permuted domain
                uhat_full = uhat_perm[inv_br]             # invert permutation back to original u order
                uhat = uhat_full[info_idx]


                block_err += int(np.any(u != uhat))
                bit_err += int(np.sum(u != uhat))
                total_bits += K
                frames += 1

            BLER = block_err / frames
            BER = bit_err / total_bits

            line = f"Eb/N0={ebn0_db:>4.1f} dB | frames={frames:>6d} | BLER={BLER:.4e} | BER={BER:.4e}"
            print(line)
            f_log.write(line + "\n")

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
                    "SC(min-sum), GA-frozen"
                ])

    print("-"*60)
    print(f"[Saved] Log: {log_path}")
    print(f"[Saved] Results: {results_csv}")

if __name__ == "__main__":
    main()
