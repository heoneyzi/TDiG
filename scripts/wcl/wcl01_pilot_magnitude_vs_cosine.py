#!/usr/bin/env python3
"""wcl01 (Stage 0 pilot) — magnitude vs cosine settling using ONLY the
already-computed TDiG tier1 parquet. No GPU, no HF download, zero marginal cost.

REVISION v2 fixes vs the original pilot:
  * settling ints in tier1 are in [-1, 29] where -1 == "never settled"; the
    CV / saturation / context stats now DROP the -1 sentinel (drop_never=True),
    otherwise -1 pollutes the mean and CV.
  * SATURATION_BAND=(28,29) is correct: max settling layer is 29 (L*), not 31.
  * This is a directional sanity check under TDiG's v2 protocol, NOT the paper's
    locked gamma_cos=0.39663 protocol — see wcl00 header for the exact difference.
    For paper-grade numbers, recompute from HF tier2 scalars (see RUN_GUIDE).

Setup (one-time):
    cd TDiG/data_cache_minimal_archive && cat data_cache_minimal.tar.gz.part-* | tar xzf -
    pip install pyarrow pandas numpy scipy --break-system-packages
Run:
    python wcl01_pilot_magnitude_vs_cosine.py \
        --tier1 TDiG/data_cache_minimal/chr22_tier1.parquet \
        --meta  TDiG/data_cache_minimal/chr22_metadata.parquet \
        --out   results/wcl/exp1_pilot/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wcl00_shared_lens_utils import (  # noqa: E402
    cohend, coefficient_of_variation, flatten_settling_column,
    load_window_metadata, per_window_means, saturation_fraction,
    COSINE_CELL_PRIMARY, MAGNITUDE_CELL,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("wcl01")

COSINE_CELL = COSINE_CELL_PRIMARY   # "M1_dir_refC"
SATURATION_BAND = (28, 29)          # top 2 settling layers of L*=29


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier1", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--intron-frac-threshold", type=float, default=0.90)
    p.add_argument("--exon-frac-threshold", type=float, default=0.50)
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"tier1_path": args.tier1, "cosine_cell": COSINE_CELL, "magnitude_cell": MAGNITUDE_CELL}

    log.info("Streaming %s (%s, %s) drop_never=True ...", args.tier1, COSINE_CELL, MAGNITUDE_CELL)
    c_cos = flatten_settling_column(args.tier1, COSINE_CELL, drop_never=True)
    c_mag = flatten_settling_column(args.tier1, MAGNITUDE_CELL, drop_never=True)
    log.info("cosine settled positions: %d | magnitude settled positions: %d", c_cos.size, c_mag.size)

    summary["full_panel"] = {
        "cosine": {"mean": float(c_cos.mean()), "std": float(c_cos.std()),
                   "cv": coefficient_of_variation(c_cos),
                   "frac_in_top2_layers": saturation_fraction(c_cos, SATURATION_BAND)},
        "magnitude": {"mean": float(c_mag.mean()), "std": float(c_mag.std()),
                      "cv": coefficient_of_variation(c_mag),
                      "frac_in_top2_layers": saturation_fraction(c_mag, SATURATION_BAND)},
    }
    cvc, cvm = summary["full_panel"]["cosine"]["cv"], summary["full_panel"]["magnitude"]["cv"]
    summary["full_panel"]["cv_ratio_cosine_over_magnitude"] = float(cvc / cvm) if cvm else float("inf")
    log.info("cosine CV=%.4f sat=%.1f%% | magnitude CV=%.4f sat=%.1f%%",
             cvc, 100 * summary["full_panel"]["cosine"]["frac_in_top2_layers"],
             cvm, 100 * summary["full_panel"]["magnitude"]["frac_in_top2_layers"])
    del c_cos, c_mag

    # coarse window-level context test (per-position labels enable the exact test; see RUN_GUIDE)
    win = per_window_means(args.tier1, [COSINE_CELL, MAGNITUDE_CELL])
    meta = load_window_metadata(args.meta)
    merged = win.merge(meta, on="window_idx")
    merged.to_csv(out_dir / "window_level_summary.csv", index=False)

    from scipy import stats as sps
    ctx = {}
    if "intron_frac" in merged.columns and "exon_frac" in merged.columns:
        intron_dom = merged[merged.intron_frac > args.intron_frac_threshold]
        exon_dom = merged[merged.exon_frac > args.exon_frac_threshold]
        for col, label in [(f"mean_{MAGNITUDE_CELL}", "magnitude"), (f"mean_{COSINE_CELL}", "cosine")]:
            a, b = exon_dom[col].to_numpy(), intron_dom[col].to_numpy()
            if a.size >= 2 and b.size >= 2:
                ctx[label] = {"n_exon": int(a.size), "n_intron": int(b.size),
                              "cohens_d_exon_vs_intron": cohend(a, b),
                              "mwu_p": float(sps.mannwhitneyu(a, b).pvalue)}
    summary["window_level_context_test"] = ctx
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Done -> %s", out_dir / "pilot_summary.json")


if __name__ == "__main__":
    main()
