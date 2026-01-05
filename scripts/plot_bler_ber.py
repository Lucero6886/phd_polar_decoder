import matplotlib
matplotlib.use("Agg")

#!/usr/bin/env python3
import argparse
import csv
import os
import re
import matplotlib.pyplot as plt

def read_run_csv(path: str):
    ebn0, bler, ber = [], [], []
    run_id = "unknown"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row.get("run_id", run_id)
            ebn0.append(float(row["EbN0_dB"]))
            bler.append(float(row["BLER"]))
            ber.append(float(row["BER"]))
    order = sorted(range(len(ebn0)), key=lambda i: ebn0[i])
    ebn0 = [ebn0[i] for i in order]
    bler = [bler[i] for i in order]
    ber  = [ber[i]  for i in order]
    return run_id, ebn0, bler, ber

def sanitize(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\-]+", "_", s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    run_id, x, bler, ber = read_run_csv(args.input)
    tag = sanitize(run_id)

    plt.figure()
    plt.semilogy(x, bler, marker="o")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BLER")
    plt.title(f"Polar SC BLER vs Eb/N0 (run_id={run_id})")
    plt.grid(True, which="both")
    out1 = os.path.join(args.outdir, f"BLER_SC_{tag}.png")
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.semilogy(x, ber, marker="o")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.title(f"Polar SC BER vs Eb/N0 (run_id={run_id})")
    plt.grid(True, which="both")
    out2 = os.path.join(args.outdir, f"BER_SC_{tag}.png")
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {out1}")
    print(f"[Saved] {out2}")

if __name__ == "__main__":
    main()
