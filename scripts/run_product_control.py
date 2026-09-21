#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dqpt_wigner.product_rotation import simulate
from dqpt_wigner.publication import plot_hierarchy_control


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sites", type=int, default=64)
    ap.add_argument("--tmax", type=float, default=1.45)
    ap.add_argument("--nt", type=int, default=501)
    ap.add_argument("--out", type=Path, default=Path("build/reproduction/product_control"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    r = simulate(args.n_sites, args.tmax, args.nt)
    df = pd.DataFrame({
        "time": r.times, "p0": r.one_site_probabilities[:,0],
        "p1": r.one_site_probabilities[:,1], "p2": r.one_site_probabilities[:,2],
        "r0": r.branch_rates[:,0], "r1": r.branch_rates[:,1],
        "r2": r.branch_rates[:,2], "total_rate": r.total_rate,
        "polarization": r.polarization,
        "critical_time": np.full_like(r.times, r.critical_time),
    })
    csv = args.out / "precession_control.csv"
    df.to_csv(csv, index=False)
    np.savetxt(args.out / "critical_wigner.csv", r.critical_wigner, delimiter=",")
    summary = {
        "n_sites": args.n_sites, "critical_time": r.critical_time,
        "critical_negativity": r.critical_negativity,
        "interpretation": "support-driven DQPT-II with a Wigner-positive critical state",
    }
    (args.out / "precession_control_summary.json").write_text(json.dumps(summary, indent=2))
    plot_hierarchy_control(df, r.critical_wigner, args.out / "fig1_hierarchy_control")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
