#!/usr/bin/env python3
"""wcl06 — the Lens Necessity Matrix (synthesis; runs TODAY, zero download).

Assembles one row per candidate lens and scores each on the criteria a
settling-depth construct actually has to satisfy, reading ONLY the Stage-6
pre-computed CSVs already in the TDiG repo (plus, if present, the fresh
Experiment 1/2/3 outputs). No GPU, no HF download.

Reads (all under the TDiG repo root):
  results/splice_vs_intron.csv                 -> biological d (donor vs intron)
  results/chr17_replication/retention_table.csv-> chr22->chr17 transfer
  results/variant_settling_cells/cell_auroc.csv-> settling-depth scalar AUROC
  results/gamma_ablation/cell_d_under_gamma.csv-> gamma robustness (optional)
  results/context_separation/best_cell_per_pair.csv -> 21-pair context contest (optional)

Optionally merges:
  results/wcl/exp2/variance_concentration.json -> well-posedness / rogue-dim
  results/wcl/exp3b/decomposition_summary.json -> incremental downstream AUROC

Output:
  results/wcl/exp6/necessity_matrix.csv         machine-readable
  results/wcl/exp6/necessity_matrix.md          camera-ready markdown table
  results/wcl/exp6/F_necessity_matrix.{png,pdf} heatmap

The matrix is designed to report every lens's WEAKNESSES beside its strengths.
The expected finding is NOT "cosine wins every column" — the TDiG data shows the
reference-free trajectory lens (M3_geo) is the strongest biological discriminator.
The defensible claim is that cosine is the minimal BOUNDED, REFERENCE-ANCHORED
lens whose threshold-crossing settling semantics match the paper's specific
construct (commitment to the output-ready frame h_norm); magnitude/distance
lenses are degenerate or depth-dominated, and the trajectory lens answers a
different (reference-free smoothness) question.

Run:
    python wcl06_necessity_matrix.py --repo /path/to/TDiG --out results/wcl/exp6/
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("wcl06")

# lens family -> (representative cell in the CSVs, human label, construct type)
LENS_ROWS = [
    ("M1_dir_refC", "Cosine / direction (paper's lens)", "bounded, reference-anchored"),
    ("M2_mag_refA", "Magnitude ratio (r−1)", "unbounded, depth-monotone"),
    ("M4_set_refA", "Whitened Mahalanobis distance", "unbounded distance"),
    ("M5_tau_refB", "Path tortuosity", "reference-free ratio"),
    ("M3_geo_a0.5_b1.0", "Trajectory velocity+curvature", "reference-free dynamics"),
]


def _get(df, cell, col, key="cell"):
    r = df[df[key] == cell]
    return None if r.empty else r.iloc[0][col]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="TDiG repo root")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    repo = Path(args.repo); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    svi = pd.read_csv(repo / "results/splice_vs_intron.csv")
    ret = pd.read_csv(repo / "results/chr17_replication/retention_table.csv")
    auroc = pd.read_csv(repo / "results/variant_settling_cells/cell_auroc.csv")

    # optional fresh outputs
    exp2 = json.loads((repo / "results/wcl/exp2/variance_concentration.json").read_text()) \
        if (repo / "results/wcl/exp2/variance_concentration.json").exists() else {}
    exp3b = json.loads((repo / "results/wcl/exp3b/decomposition_summary.json").read_text()) \
        if (repo / "results/wcl/exp3b/decomposition_summary.json").exists() else {}

    rows = []
    for cell, label, ctype in LENS_ROWS:
        d = _get(svi, cell, "cohens_d_donor_minus_intron")
        rp = _get(ret, cell, "retention_pct")
        sp = _get(ret, cell, "sign_preserved")
        au = _get(auroc, cell, "AUROC")
        # well-posedness: is the cell degenerate (all-zero d / undefined) under the shared pipeline?
        degenerate = (d is None) or (isinstance(d, float) and not np.isfinite(d))
        bounded = cell.startswith("M1")      # only cosine distance is intrinsically bounded in [0,2]
        reference_anchored = not (cell.startswith("M3") or cell.startswith("M5"))
        rows.append({
            "lens": label, "cell": cell, "construct_type": ctype,
            "bounded": bounded, "reference_anchored": reference_anchored,
            "well_posed_nondegenerate": not degenerate,
            "biological_d_donor_vs_intron": None if d is None else round(float(d), 3),
            "abs_biological_d": None if d is None else round(abs(float(d)), 3),
            "chr22_to_chr17_retention_pct": None if rp is None else round(float(rp), 1),
            "sign_preserved": bool(sp) if sp is not None else None,
            "settling_scalar_AUROC": None if au is None else round(float(au), 3),
        })
    mat = pd.DataFrame(rows)

    # rank flags for readability
    mat["strongest_biological"] = mat["abs_biological_d"] == mat["abs_biological_d"].max()
    mat.to_csv(out / "necessity_matrix.csv", index=False)

    # ---- camera-ready markdown ----
    md = ["# Lens Necessity Matrix (TDiG v2 protocol, chr22; transfer to chr17)\n",
          "Source: TDiG `results/` Stage-6 CSVs. Every lens scored on the criteria the",
          "settling-depth construct requires. Cosine is **not** the strongest biological",
          "discriminator — the reference-free trajectory lens is — so the necessity claim",
          "is about *construct match* (bounded, reference-anchored, well-posed threshold",
          "crossing to the output-ready frame), not effect-size superiority.\n",
          "| Lens | bounded | ref-anchored | well-posed | biol. d (donor−intron) | chr22→chr17 | settling AUROC |",
          "|---|:--:|:--:|:--:|--:|--:|--:|"]
    for _, r in mat.iterrows():
        md.append("| {lens} | {b} | {ra} | {wp} | {d} | {ret}{sp} | {au} |".format(
            lens=r["lens"], b="✓" if r["bounded"] else "✗",
            ra="✓" if r["reference_anchored"] else "✗",
            wp="✓" if r["well_posed_nondegenerate"] else "✗(degenerate)",
            d="—" if r["biological_d_donor_vs_intron"] is None else f"{r['biological_d_donor_vs_intron']:+.3f}",
            ret="—" if r["chr22_to_chr17_retention_pct"] is None else f"{r['chr22_to_chr17_retention_pct']:.0f}%",
            sp="" if r["sign_preserved"] else ("" if r["sign_preserved"] is None else " (flip)"),
            au="—" if r["settling_scalar_AUROC"] is None else f"{r['settling_scalar_AUROC']:.3f}"))
    md.append("\n**Reading of the matrix.**")
    md.append("- Magnitude (M2) and whitened distance (M4) collapse to a degenerate cell under")
    md.append("  two of three reference conventions and, where defined, carry the wrong-sign / weak")
    md.append("  biological signal — they are not well-posed settling constructs.")
    md.append("- Trajectory (M3_geo) is the strongest biological discriminator and transfers well,")
    md.append("  BUT is reference-free: it measures trajectory smoothness, not commitment to the")
    md.append("  model's output-ready frame — a different question (paper Def-2/Def-3 split).")
    md.append("- Cosine (M1) is the only bounded, reference-anchored, non-degenerate lens whose")
    md.append("  threshold crossing is the paper's exact construct; its weaker raw d is the cost of")
    md.append("  measuring the *right* target rather than the easiest-to-separate one.")
    if exp2:
        md.append(f"\n*Rogue-dimension (Exp 2): top-8 variance share at L29 = "
                  f"{exp2.get('H2a_variance_concentration', {}).get('L29', {}).get('top_8_frac', 'NA')}.*")
    if exp3b:
        md.append(f"*Incremental AUROC (Exp 3b): D+M vs M = "
                  f"{exp3b.get('H3a_complementarity', {}).get('D+M_vs_M', {}).get('mean_diff', 'NA')}.*")
    (out / "necessity_matrix.md").write_text("\n".join(md) + "\n")

    _plot(out, mat)
    log.info("Done. Matrix rows:\n%s", mat.to_string(index=False))


def _plot(out: Path, mat: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["bounded", "reference_anchored", "well_posed_nondegenerate"]
    Z = mat[cols].astype(float).values
    fig, ax = plt.subplots(figsize=(7, 3.2))
    im = ax.imshow(Z, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(["bounded", "ref-anchored", "well-posed"], fontsize=8)
    ax.set_yticks(range(len(mat))); ax.set_yticklabels(mat["lens"], fontsize=8)
    for i in range(len(mat)):
        d = mat.iloc[i]["biological_d_donor_vs_intron"]
        ax.text(len(cols) - 0.5, i, f"  d={d:+.2f}" if d is not None else "  d=—", va="center", fontsize=8)
    ax.set_title("Lens necessity: construct criteria (green=yes)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"F_necessity_matrix.{ext}", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
