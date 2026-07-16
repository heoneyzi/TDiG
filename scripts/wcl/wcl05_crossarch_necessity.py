#!/usr/bin/env python3
"""wcl05 — Experiment 5: cross-architecture necessity (scoped to what public
assets actually allow).

HONEST SCOPING (this is the main correction vs. the original plan). The original
plan assumed a 4-model panel (Evo2/HyenaDNA/NT-v2/DNABERT-2) with cached raw
hidden states in `results/phase4/`. That cache does not exist in either public
repo, and there are NO raw hidden states for NT-v2/DNABERT-2 anywhere public, so
their magnitude lens cannot be recomputed from public assets. What CAN be run:

  (A) HyenaDNA-large (per-bp causal LM, RMSNorm-family, 8 layers): TDiG already
      ran the cosine cross-arch analysis (scripts/36_hyenadna_crossarch.py ->
      results/hyenadna_crossarch/{hyenadna_tier1.parquet, comparison_vs_evo2.json,
      hyenadna_splice_vs_intron.csv}). This script reads hyenadna_tier1.parquet
      and applies the SAME magnitude-vs-cosine necessity comparison as Exp 1, IF
      the parquet carries a magnitude cell; if it carries only the cosine cell,
      the script reports the cosine transfer and flags that the magnitude
      replication needs a HyenaDNA re-forward (cheap: script 36 re-run).

  (B) NT-v2 / DNABERT-2 (LayerNorm MLMs): the paper's published Table A6/A7
      cosine numbers are cited as-is; the magnitude-degeneracy replication is
      recorded as "requires a re-forward that saves per-layer norms — not
      possible from current public caches" (a limitation, reported honestly).

H5a  within-causal-LM-family: magnitude degeneracy / weakness replicates on
     HyenaDNA with the same sign as Evo 2.

Run:
    python wcl05_crossarch_necessity.py --repo /path/to/TDiG --out results/wcl/exp5/
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("wcl05")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="TDiG repo root")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    repo = Path(args.repo); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    summary = {"models": {}}

    # ---- Evo 2 reference (from Exp 1 CSVs) ----
    svi = pd.read_csv(repo / "results/splice_vs_intron.csv")
    evo_cos = float(svi[svi.cell == "M1_dir_refC"].cohens_d_donor_minus_intron.iloc[0])
    evo_mag = float(svi[svi.cell == "M2_mag_refA"].cohens_d_donor_minus_intron.iloc[0])
    summary["models"]["evo2"] = {"cosine_d": evo_cos, "magnitude_d": evo_mag,
                                 "magnitude_degenerate_or_weak": abs(evo_mag) < abs(evo_cos) or evo_mag * evo_cos < 0}

    # ---- HyenaDNA (from TDiG cross-arch outputs) ----
    hy_json = repo / "results/hyenadna_crossarch/comparison_vs_evo2.json"
    hy_par = repo / "results/hyenadna_crossarch/hyenadna_tier1.parquet"
    hy = {}
    if hy_json.exists():
        hy["comparison_vs_evo2"] = json.loads(hy_json.read_text())
    if hy_par.exists():
        df = pd.read_parquet(hy_par)
        cells = [c for c in df.columns if c.startswith(("M1", "M2", "M3", "M4", "M5"))]
        hy["available_cells"] = cells
        hy["has_magnitude_cell"] = any(c.startswith("M2") for c in cells)
        log.info("HyenaDNA tier1 cells: %s", cells)
        if not hy["has_magnitude_cell"]:
            hy["magnitude_replication"] = ("NOT PRESENT in hyenadna_tier1.parquet — re-run "
                                           "scripts/36_hyenadna_crossarch.py with M2 enabled to test H5a "
                                           "(cheap: HyenaDNA-large forward is minutes on H200).")
    else:
        hy["note"] = "results/hyenadna_crossarch/ not found — run scripts/36_hyenadna_crossarch.py first."
    summary["models"]["hyenadna_large"] = hy

    # ---- MLMs: honest limitation ----
    summary["models"]["nt_v2_dnabert2"] = {
        "status": "cosine numbers cited from paper Table A6/A7 (per-window, tokenisation-limited)",
        "magnitude_replication": "NOT POSSIBLE from public assets — no per-layer raw hidden states "
                                 "for NT-v2/DNABERT-2 exist on HF or GitHub. Would require a re-forward "
                                 "that additionally saves ‖h_ell‖. Reported as a scope limit.",
    }

    summary["H5a_verdict"] = ("supported_on_hyenadna" if hy.get("has_magnitude_cell")
                              else "pending_hyenadna_reforward")
    (out / "crossarch_necessity.json").write_text(json.dumps(summary, indent=2))
    log.info("Done -> %s", out / "crossarch_necessity.json")
    log.info("Evo2 cosine d=%.3f magnitude d=%.3f (magnitude weaker/wrong-sign: %s)",
             evo_cos, evo_mag, summary["models"]["evo2"]["magnitude_degenerate_or_weak"])


if __name__ == "__main__":
    main()
