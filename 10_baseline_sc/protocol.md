# Baseline v1: Polar SC (min-sum) on AWGN

## Setup
- Code: Polar (N=128, K=64, R=1/2)
- Construction: Gaussian Approximation (GA), design Eb/N0 = 2.0 dB
- Modulation: BPSK (0->+1, 1->-1)
- Channel: AWGN
- Decoder: Successive Cancellation (SC), min-sum f-function, exact g-function with partial sums
- Metrics: BLER (primary), BER (secondary)

## Simulation protocol
- Eb/N0 grid: 0.0, 0.5, 1.0, 1.5, 2.0 dB
- Stopping rule: E_min = 200 block errors OR N_max = 50,000 frames
- RNG seed: 0
- Outputs: logs + per-run CSV + figures via scripts/
