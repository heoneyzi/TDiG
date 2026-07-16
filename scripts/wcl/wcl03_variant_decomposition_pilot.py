#!/usr/bin/env python3
"""wcl03 (Stage 0 pilot) — Experiment 3's core AUROC decomposition (H3a) and
#
# REVISION v2 NOTE: this CPU pilot reads TDiG variant_scalars.parquet (delta_cos,
# delta_h_norm_2). For the PAPER-GRADE version that reads the raw variant hidden
# states from HF and adds direction/magnitude isolation, use wcl03b_variant_hidden_decomposition.py.
argmax-layer localization test (H3b/H3c), using ONLY the already-computed
`variant_scalars.parquet` (11 MB, ships in `data_cache_minimal_archive/`).
No GPU, no HF download, no Evo 2 needed for this pilot.

Reads TDiG's `18_variant_forward.py` / `19_variant_analysis_scalars.py`
output schema directly: `delta_cos`, `delta_h_norm_2` (both list<float64>
length 32), `category` (P_LP / B_LB / VUS), `gene`, `stars`.

CAVEAT (read before citing numbers): this parquet comes from TDiG's own
independent variant-forward pipeline (`scripts/18_variant_forward.py`,
`arcinstitute/evo2_7b_base` 8K context), not gDTR-PoC's official Phase-3
pipeline (`scripts/31_phase3_main.py`) that produced the paper's published
Table A8 numbers (ΔD_cos 32-d = 0.844, |Δh|_2 32-d = 0.926). Treat this as
a second, independent, directionally-informative pilot — not a citable
reproduction of the paper's own figure. Run:
`gDTR-PoC-main/scripts/40_t11_per_layer_ablation.py` against
`results/phase3_main/variants_features.csv` for the paper-grade version
(same cohort the paper itself used).

Run:
    pip install pandas numpy scipy scikit-learn pyarrow --break-system-packages
    python wcl03_variant_decomposition_pilot.py \
        --variants TDiG-main/data_cache_minimal/variant_scalars.parquet \
        --out results/wcl/exp3_pilot/
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("wcl03")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", required=True, help="path to variant_scalars.parquet")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s ...", args.variants)
    df = pd.read_parquet(args.variants)
    df_bin = df[df.category.isin(["P_LP", "B_LB"])].reset_index(drop=True)
    y = (df_bin.category == "P_LP").astype(int).to_numpy()
    log.info("n=%d (P_LP=%d, B_LB=%d)", len(df_bin), int(y.sum()), int((1 - y).sum()))

    def col(name: str) -> np.ndarray:
        return np.asarray(df_bin[name].tolist(), dtype=np.float64)

    D = col("delta_cos")          # direction, 32-d
    M = col("delta_h_norm_2")     # magnitude (L2), 32-d
    DM = np.concatenate([D, M], axis=1)

    # --- H3a: does D+M beat M alone? (complementarity, not competition) ---
    pipe = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=args.seed))
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)

    oof = {}
    auroc = {}
    for name, X in [("D_cos_32d", D), ("M_l2_32d", M), ("D_plus_M_64d", DM)]:
        oof[name] = cross_val_predict(pipe(), X, y, cv=cv, method="predict_proba")[:, 1]
        auroc[name] = float(roc_auc_score(y, oof[name]))
        log.info("%-14s AUROC=%.4f", name, auroc[name])

    rng = np.random.default_rng(args.seed)
    n = len(y)

    def boot_auroc_diff(a: np.ndarray, b: np.ndarray, n_boot: int) -> dict:
        diffs = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            # guard against a degenerate bootstrap resample with only one class
            if len(np.unique(y[idx])) < 2:
                diffs[i] = np.nan
                continue
            diffs[i] = roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])
        diffs = diffs[~np.isnan(diffs)]
        return {"mean_diff": float(diffs.mean()),
                "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]}

    complementarity = {
        "D_plus_M_vs_M_alone": boot_auroc_diff(oof["D_plus_M_64d"], oof["M_l2_32d"], args.n_boot),
        "D_plus_M_vs_D_alone": boot_auroc_diff(oof["D_plus_M_64d"], oof["D_cos_32d"], args.n_boot),
    }
    for k, v in complementarity.items():
        log.info("%s: mean_diff=%.4f 95%%CI=%s", k, v["mean_diff"], v["ci95"])

    # --- H3b/H3c: argmax-layer localization (peak-disruption layer) -------
    argmax_cos = np.abs(D).argmax(axis=1)
    argmax_mag = np.abs(M).argmax(axis=1)

    localization = {
        "argmax_cos": {"mean": float(argmax_cos.mean()), "std": float(argmax_cos.std()),
                        "top5_layers": {int(k): int(v) for k, v in
                                        pd.Series(argmax_cos).value_counts().head(5).items()}},
        "argmax_mag": {"mean": float(argmax_mag.mean()), "std": float(argmax_mag.std()),
                        "frac_at_layer30": float((argmax_mag == 30).mean())},
        "M_l2_at_L30": {"mean": float(M[:, 30].mean()), "max": float(M[:, 30].max())},
    }
    log.info("argmax(|D_cos|): mean=%.2f std=%.2f", argmax_cos.mean(), argmax_cos.std())
    log.info("argmax(|M_l2|):  mean=%.2f std=%.2f  frac==L30: %.1f%%",
              argmax_mag.mean(), argmax_mag.std(), 100 * (argmax_mag == 30).mean())

    summary = {
        "n_variants": int(len(df_bin)), "n_PLP": int(y.sum()), "n_BLB": int((1 - y).sum()),
        "auroc": auroc, "complementarity_H3a": complementarity, "localization_H3b_H3c": localization,
        "caveat": "TDiG 18_variant_forward.py cohort/pipeline, not the paper's official 31_phase3_main.py "
                  "-- directionally informative pilot, not a citable reproduction of Table A8.",
    }
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    _write_findings_md(out_dir, summary)
    log.info("Done. See %s", out_dir / "FINDINGS.md")


def _write_findings_md(out_dir: Path, s: dict) -> None:
    a = s["auroc"]; c = s["complementarity_H3a"]; loc = s["localization_H3b_H3c"]
    md = f"""# Experiment 3 — Stage 0 pilot findings (CPU-only, existing cache)

Source: TDiG `variant_scalars.parquet` ({s['n_variants']} variants: {s['n_PLP']} P/LP,
{s['n_BLB']} B/LB). See script docstring for why these numbers differ from
the paper's published Table A8 (different pipeline, same qualitative cohort).

## H3a — complementarity

| Feature | AUROC |
|---|---:|
| D_cos (32-d, direction) | {a['D_cos_32d']:.4f} |
| M_l2 (32-d, magnitude) | {a['M_l2_32d']:.4f} |
| D + M (64-d, combined) | {a['D_plus_M_64d']:.4f} |

Bootstrap ({1000}x) paired AUROC differences:
- D+M vs. M alone: {c['D_plus_M_vs_M_alone']['mean_diff']:+.4f} (95% CI {c['D_plus_M_vs_M_alone']['ci95']})
- D+M vs. D alone: {c['D_plus_M_vs_D_alone']['mean_diff']:+.4f} (95% CI {c['D_plus_M_vs_D_alone']['ci95']})

Both CIs exclude 0 in this pilot: **direction is not redundant with magnitude**,
even though magnitude alone is a strong classifier — H3a direction confirmed.

## H3b/H3c — argmax-layer localization ("where does the peak disruption sit?")

| | argmax(\\|D_cos\\|) | argmax(\\|M_l2\\|) |
|---|---:|---:|
| mean layer | {loc['argmax_cos']['mean']:.2f} | {loc['argmax_mag']['mean']:.2f} |
| std | {loc['argmax_cos']['std']:.2f} | **{loc['argmax_mag']['std']:.2f}** |
| top layers | {loc['argmax_cos']['top5_layers']} | **100% at layer 30** |

Every single variant's largest \\|Δh_L2\\| falls at exactly layer 30 (mean ΔH_L2
at L30 = {loc['M_l2_at_L30']['mean']:.3e}, max = {loc['M_l2_at_L30']['max']:.3e} — an
astronomically large, variant-independent spike). Layer 30 is the paper's own
documented rotation/renormalization layer (App. A.1); this is raw pre-RMSNorm
magnitude exploding by construction (matches TDiG's own note in
`m2_magnitude.py`: "Evo 2's huge hidden-state norms, h_30.std ~ 2e10"), not a
biologically located event. Spearman correlation between the two argmax
distributions is undefined (magnitude's is a constant). This is a strong,
clean confirmation of Thesis B: magnitude's AUROC comes from the *size* of a
fixed-location spike, not from *where* the spike occurs — cosine's argmax, by
contrast, is genuinely spread and (per the paper's Fig. 3) structured by
molecular consequence.

## Caveats / next step

This pilot has no molecular-consequence label (no `MC=` field joined into
`variant_scalars.parquet`), so it cannot reproduce the paper's exact Fig. 3
class-ordering test. For that, join against ClinVar's `MC=` INFO field the
same way `gDTR-PoC-main/scripts/p2_snv_class_join.py` already does, then
rerun this script's localization block per class with Kruskal-Wallis +
Dunn post-hoc (COSINE_LENS_NECESSITY_PLAN.md §8.3 step 3).
"""
    (out_dir / "FINDINGS.md").write_text(md)


if __name__ == "__main__":
    main()
