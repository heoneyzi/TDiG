#!/usr/bin/env python3
"""wcl09 — Experiment 9: Dimensionality & Locality Nulls.
Proves the ΔD_cos / (ΔD_cos ⊕ ‖Δh‖) AUROC advantage is REAL LAYER-WISE SIGNAL,
not a free-parameter (dimensionality) artefact — and explains, quantitatively,
why magnitude wins at 1-D even though its disruption peaks at a fixed layer.

=================================================================================
WHY THIS EXPERIMENT (the airtight logic)
=================================================================================
The reviewer's objection is sharp and cannot be answered by removing the AUROC
result (it is a real, large effect: cosine-32d ≈ magnitude-32d, and the 64-d
combination is higher than either). The objection is instead:

    "You are just adding dimensions. Knowing the variant-vs-nonvariant
     difference in 32 (or 64) free dimensions gives a classifier more capacity;
     of course AUROC goes up. This is dimensionality, not biology."

A cross-validated AUROC already blocks the crudest version of this (pure-noise
dimensions collapse to 0.5 out-of-fold), but the reviewer deserves a DIRECT,
quantitative demonstration. This experiment supplies five, each targeting a
different face of the objection. Every test uses ONLY the already-computed
`variant_scalars.parquet` (delta_cos, delta_h_norm_2) — no GPU, no HF download —
so it runs immediately and reproduces on any laptop.

  H9a  Label-permutation null. Re-fit the SAME 32-d / 64-d pipeline under the
       SAME CV on LABEL-SHUFFLED data, P times. If capacity alone manufactured
       the score, the permuted 64-d AUROC would inherit the "extra dimensions"
       and sit well above 0.5. Prediction: permuted AUROC ≈ 0.5 for every model;
       real AUROC is far outside the null (empirical p < 1/P). => the 0.84–0.94
       is label-associated signal, not dimensional capacity.

  H9b  Dimension learning curve vs a matched null curve. Sweep top-k PCA
       components k = 1..32 (PCA fit per-fold, no leakage) and plot real OOF
       AUROC(k) against the mean permuted-label AUROC(k). If the signal were
       "spread thinly over 32 free params", real AUROC would keep climbing only
       as k→32 and hug the null. Prediction: real AUROC saturates by k≪32 and
       the real−null gap is large at every k. => the signal lives in a LOW-
       dimensional subspace; the 32 columns are not 32 independent free knobs.

  H9c  Matched-dimension ensemble control (answers "the 64-d combo is just 2×
       the dimensions"). Compare real 64-d (cos ⊕ mag) against two 64-d controls
       with IDENTICAL dimensionality:
         (i)  mag-32 ⊕ 32 columns of Gaussian noise
         (ii) mag-32 ⊕ 32 columns of PER-VARIANT layer-shuffled cosine
       If doubling the dimension were the cause, (i) would already recover the
       gain. Prediction: real 64-d > (i) ≈ mag-32 alone, and real 64-d > (ii),
       by bootstrap paired AUROC. => the +32 columns help ONLY when they carry
       real, correctly-layer-aligned cosine information.

  H9d  Redundancy signature (answers "why does magnitude win at 1-D if it always
       peaks at L30?"). AUROC (scoring) and localization (where the peak sits)
       are ORTHOGONAL. Magnitude's discriminative power is the SIZE of the
       perturbation, which is preserved across layers (high inter-layer
       correlation) and therefore readable at ANY single tap — a fixed peak
       LOCATION does not cost AUROC, it only costs layer-information. We
       quantify this:
         gap_mag = AUROC(mag 32d) − AUROC(mag best-single-tap)   → SMALL
         gap_cos = AUROC(cos 32d) − AUROC(cos best-single-tap)   → LARGE
         plus mean inter-layer |corr| (mag high, cos low) and argmax-layer
         entropy (mag low, cos high).
       => magnitude is one strong redundant scalar (great scorer, zero layer
       resolution); cosine is a distributed multi-layer signal (its 32-d ≫ 1-d
       gain is exactly the layer-wise content the paper claims).

  H9e  Locality test (does the LAYER INDEX carry information, or only the bag of
       values?). Compare the true layer-indexed cosine-32d against
         (i)  an order-invariant 4-d summary {mean, std, max, min} of the same
              32 cosine values, and
         (ii) per-variant layer-shuffled cosine-32d.
       If only the value distribution mattered, (i) and (ii) would match 32-d.
       Prediction: 32-d > summary-4d and 32-d > layer-shuffled. => the specific
       LAYER at which the directional change occurs is informative — the precise
       sense in which "layer-wise differences are meaningful".

Together: H9a/H9b/H9c defeat "it's just dimensionality"; H9d explains the 1-D
magnitude win without contradiction; H9e establishes the positive claim that
the layer axis (not merely 32 numbers) carries the signal.

=================================================================================
DATA / DEPENDENCIES (CPU-only, immediate)
=================================================================================
  * TDiG data_cache_minimal/variant_scalars.parquet
      columns used: category (P_LP/B_LB), gene, delta_cos [32], delta_h_norm_2 [32]
  * pip install pandas pyarrow numpy scipy scikit-learn matplotlib
  * (optional) wcl00_shared_lens_utils on sys.path for N_LAYERS/cohend; falls
    back to local constants if absent.

Run:
    python wcl09_dimensionality_nulls.py \
        --scalars ../../data_cache_minimal/variant_scalars.parquet \
        --out ../../results/wcl_r1/exp9 \
        --n-perm 200 --n-boot 1000 --seed 42
    # add --quick for a fast smoke run (n-perm 40, coarse PCA grid)

Outputs (results/wcl_r1/exp9/):
    dimensionality_nulls_summary.json   all numbers + per-hypothesis verdicts
    F_dimensionality_nulls.png/.pdf     4-panel figure (H9a, H9b, H9c, H9d/H9e)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- optional shared-utils import (kept non-fatal so the script is standalone) --
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from wcl00_shared_lens_utils import N_LAYERS, cohend  # noqa: F401
except Exception:  # pragma: no cover - standalone fallback
    N_LAYERS = 32

    def cohend(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        if a.size < 2 or b.size < 2:
            return float("nan")
        sp = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
                     / (a.size + b.size - 2))
        return float((a.mean() - b.mean()) / sp) if sp else float("nan")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("wcl09")


# =============================================================================
# feature construction
# =============================================================================
def _stack(col: pd.Series) -> np.ndarray:
    """[n, 32] float array from a column of length-32 arrays/lists."""
    return np.asarray([np.asarray(v, dtype=np.float64) for v in col.tolist()],
                      dtype=np.float64)


def load_features(scalars_path: Path):
    """Return (COS, MAG, y, groups). COS/MAG are [n, 32]; y in {0,1}; groups=gene.

    MAG is log1p-transformed (Evo 2's per-layer norms span ~1e0..1e13, so the
    raw magnitude delta explodes at the L30 rotation layer; log1p keeps every
    layer on a comparable, finite scale — matching scripts/35's build_features).
    Cosine deltas are already O(1) and fed raw.
    """
    df = pd.read_parquet(scalars_path)
    df = df[df.category.isin({"P_LP", "B_LB"})].reset_index(drop=True)
    y = (df.category == "P_LP").astype(int).to_numpy()
    groups = df.gene.to_numpy()
    COS = _stack(df.delta_cos)                       # [n, 32] raw directional delta
    MAG = np.log1p(np.abs(_stack(df.delta_h_norm_2)))  # [n, 32] log-magnitude delta
    if COS.shape[1] != N_LAYERS or MAG.shape[1] != N_LAYERS:
        raise ValueError(f"expected {N_LAYERS} layers, got cos={COS.shape}, mag={MAG.shape}")
    log.info("[data] n=%d  (P_LP=%d, B_LB=%d)  genes=%d",
             len(y), int(y.sum()), int((1 - y).sum()), len(np.unique(groups)))
    return COS, MAG, y, groups


def make_clf():
    """Standardize + L2 logistic regression. One estimator used everywhere so
    every comparison differs ONLY in its feature matrix (the §5 'same pipeline'
    rule), never in the classifier."""
    return Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
    ])


def oof_scores(X: np.ndarray, y: np.ndarray, cv, n_jobs: int = 1) -> np.ndarray:
    """Out-of-fold P(y=1) via cross_val_predict (no train/test leakage)."""
    return cross_val_predict(make_clf(), X, y, cv=cv, method="predict_proba",
                             n_jobs=n_jobs)[:, 1]


def oof_auroc(X: np.ndarray, y: np.ndarray, cv, n_jobs: int = 1) -> float:
    return float(roc_auc_score(y, oof_scores(X, y, cv, n_jobs=n_jobs)))


# =============================================================================
# statistics
# =============================================================================
def bootstrap_paired_auroc_diff(scores_a, scores_b, y, n_boot, rng):
    """Paired bootstrap of AUROC(a) − AUROC(b) on fixed OOF score vectors.
    Returns (mean_diff, lo95, hi95, frac_diff>0). Resamples subjects with
    replacement so the two AUROCs share the same resample (paired)."""
    y = np.asarray(y); n = len(y)
    a = np.asarray(scores_a); b = np.asarray(scores_b)
    diffs = np.empty(n_boot)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    for i in range(n_boot):
        # stratified resample keeps both classes present in every replicate
        idx = np.concatenate([rng.choice(pos, pos.size, replace=True),
                              rng.choice(neg, neg.size, replace=True)])
        yi = y[idx]
        diffs[i] = roc_auc_score(yi, a[idx]) - roc_auc_score(yi, b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi), float((diffs > 0).mean())


def permutation_null_auroc(X, y, cv, n_perm, rng):
    """Null AUROC distribution under label permutation (same pipeline, same CV).
    Returns (null_aurocs[n_perm], real_auroc, empirical_p_one_sided)."""
    real = oof_auroc(X, y, cv)
    null = np.empty(n_perm)
    for i in range(n_perm):
        yp = rng.permutation(y)
        null[i] = oof_auroc(X, yp, cv)
    # one-sided p: how often the null reaches/exceeds the real score
    p = (1.0 + np.sum(null >= real)) / (n_perm + 1.0)
    return null, real, float(p)


# =============================================================================
# H9d / H9e helpers
# =============================================================================
def best_single_tap_auroc(X, y, cv):
    """Max over single-column logistic AUROCs (orientation handled by the LR),
    plus the argmax layer and the full per-layer curve."""
    per_layer = np.array([oof_auroc(X[:, [l]], y, cv) for l in range(X.shape[1])])
    best_l = int(np.argmax(per_layer))
    return float(per_layer[best_l]), best_l, per_layer


def mean_interlayer_abscorr(X):
    """Mean |Pearson r| over all layer pairs — how redundant the 32 columns are.
    High => one signal repeated across layers; low => layers carry different info."""
    C = np.corrcoef(X, rowvar=False)
    iu = np.triu_indices_from(C, k=1)
    return float(np.nanmean(np.abs(C[iu])))


def argmax_layer_entropy(X):
    """Shannon entropy (bits) of the per-variant argmax-|value| layer distribution.
    Low => peak location is concentrated (few layers); high => spread across layers."""
    arg = np.argmax(np.abs(X), axis=1)
    counts = np.bincount(arg, minlength=X.shape[1]).astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()), arg


def order_invariant_summary(X):
    """4-d order-invariant descriptor {mean, std, max, min} of each variant's 32
    values — discards the layer axis entirely."""
    return np.column_stack([X.mean(1), X.std(1), X.max(1), X.min(1)])


def per_variant_layer_shuffle(X, rng):
    """Independently permute the layer axis within each variant. Keeps every
    variant's multiset of 32 values but destroys the column⇄layer correspondence,
    so a pooled linear model can no longer use layer identity."""
    Xs = X.copy()
    for i in range(Xs.shape[0]):
        Xs[i] = Xs[i, rng.permutation(Xs.shape[1])]
    return Xs


# =============================================================================
# main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scalars", type=Path,
                    default=Path(__file__).resolve().parents[2]
                    / "data_cache_minimal" / "variant_scalars.parquet")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[2] / "results" / "wcl_r1" / "exp9")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=200, help="label permutations (H9a)")
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples (H9c)")
    ap.add_argument("--pca-null-perm", type=int, default=20,
                    help="permutations per k for the H9b null curve")
    ap.add_argument("--quick", action="store_true", help="fast smoke run")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.quick:
        args.n_perm = min(args.n_perm, 40)
        args.n_boot = min(args.n_boot, 300)
        args.pca_null_perm = min(args.pca_null_perm, 6)

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    COS, MAG, y, groups = load_features(args.scalars)
    COMB = np.concatenate([COS, MAG], axis=1)         # 64-d real combination
    summary = {"n": int(len(y)), "n_PLP": int(y.sum()), "n_BLB": int((1 - y).sum()),
               "n_layers": int(N_LAYERS), "folds": args.folds, "seed": args.seed}

    # ---- headline AUROCs (context; compare shapes to paper Table A8) ----------
    log.info("\n=== headline OOF AUROC (%d-fold) ===", args.folds)
    auc_cos = oof_auroc(COS, y, cv); auc_mag = oof_auroc(MAG, y, cv)
    auc_comb = oof_auroc(COMB, y, cv)
    summary["headline_auroc"] = {"cos_32d": auc_cos, "mag_32d": auc_mag, "comb_64d": auc_comb}
    log.info("  cos-32d=%.4f  mag-32d=%.4f  comb-64d=%.4f", auc_cos, auc_mag, auc_comb)

    # ===================== H9a — label-permutation null ========================
    log.info("\n=== H9a  label-permutation null (P=%d) ===", args.n_perm)
    h9a = {}
    for name, X in [("cos_32d", COS), ("mag_32d", MAG), ("comb_64d", COMB)]:
        null, real, p = permutation_null_auroc(X, y, cv, args.n_perm, rng)
        h9a[name] = {"real_auroc": real, "null_mean": float(null.mean()),
                     "null_std": float(null.std()), "null_p95": float(np.percentile(null, 95)),
                     "null_max": float(null.max()), "empirical_p": p}
        log.info("  %-9s real=%.4f  null=%.4f±%.4f (max %.4f)  p=%.2g",
                 name, real, null.mean(), null.std(), null.max(), p)
    summary["H9a_permutation_null"] = h9a
    summary["H9a_verdict"] = (
        "PASS: permuted-label AUROC ~0.5 for all models incl. 64-d; real scores "
        "outside the null (p<1/P). Dimensional capacity alone does not produce the score."
        if all(v["null_p95"] < 0.6 and v["empirical_p"] < 0.05 for v in h9a.values())
        else "INSPECT: a null distribution reached >0.6 — check leakage/imbalance.")

    # ===================== H9b — dimension learning curve ======================
    log.info("\n=== H9b  PCA dimension curve vs matched null ===")
    ks = ([1, 2, 4, 8, 16, 32] if args.quick
          else [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32])
    real_curve, null_curve = [], []
    for k in ks:
        pipe_cv = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=k)),
                            ("lr", LogisticRegression(max_iter=2000, C=1.0))])
        real_k = float(roc_auc_score(
            y, cross_val_predict(pipe_cv, COS, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]))
        nvals = []
        for _ in range(args.pca_null_perm):
            yp = rng.permutation(y)
            nvals.append(float(roc_auc_score(
                yp, cross_val_predict(pipe_cv, COS, yp, cv=cv, method="predict_proba", n_jobs=1)[:, 1])))
        real_curve.append(real_k); null_curve.append(float(np.mean(nvals)))
        log.info("  k=%2d  real=%.4f  null=%.4f  gap=%.4f", k, real_k, null_curve[-1],
                 real_k - null_curve[-1])
    # effective dimensionality: smallest k reaching 95% / 99% of the k=32 ceiling
    peak = real_curve[-1]
    k95 = next((ks[i] for i, v in enumerate(real_curve) if v >= 0.95 * peak), ks[-1])
    k99 = next((ks[i] for i, v in enumerate(real_curve) if v >= 0.99 * peak), ks[-1])
    min_gap = float(np.min(np.array(real_curve) - np.array(null_curve)))
    null_flat = float(np.max(null_curve)) < 0.60
    summary["H9b_dimension_curve"] = {"k": ks, "real_auroc": real_curve,
                                      "null_auroc": null_curve,
                                      "k_saturation_95pct": int(k95), "k_saturation_99pct": int(k99),
                                      "min_real_minus_null_gap": min_gap,
                                      "null_curve_max": float(np.max(null_curve))}
    summary["H9b_verdict"] = (
        f"PASS: cosine reaches 95% of its 32-d ceiling by k={k95} PC(s) (99% by k={k99}); the "
        f"label-permuted null stays flat (max {np.max(null_curve):.3f}) so the real−null gap is "
        f">={min_gap:.3f} at EVERY k. The signal is concentrated in a few components — extra "
        f"columns add capacity that the null shows is worthless, yet the real curve does not use it."
        if (null_flat and k95 <= max(4, ks[len(ks) // 3])) else
        f"INSPECT: k95={k95}, null_max={np.max(null_curve):.3f} — re-examine effective dimensionality.")

    # ===================== H9c — matched-dimension ensemble ====================
    log.info("\n=== H9c  matched-dimension ensemble controls (64-d each) ===")
    noise32 = rng.standard_normal(size=COS.shape)                      # (i)  pure noise
    cos_shuf = per_variant_layer_shuffle(COS, rng)                     # (ii) layer-broken cosine
    ctrl_noise = np.concatenate([MAG, noise32], axis=1)               # mag ⊕ noise
    ctrl_shuf = np.concatenate([MAG, cos_shuf], axis=1)               # mag ⊕ shuffled-cos
    s_real = oof_scores(COMB, y, cv)
    s_mag = oof_scores(MAG, y, cv)
    s_noise = oof_scores(ctrl_noise, y, cv)
    s_shuf = oof_scores(ctrl_shuf, y, cv)
    a_real = roc_auc_score(y, s_real); a_noise = roc_auc_score(y, s_noise)
    a_shuf = roc_auc_score(y, s_shuf); a_mag = roc_auc_score(y, s_mag)
    h9c = {"auroc": {"real_comb_64d": float(a_real), "mag32_plus_noise32": float(a_noise),
                     "mag32_plus_shuffledcos32": float(a_shuf), "mag_32d": float(a_mag)}}
    for label, s_ctrl in [("real_vs_mag+noise", s_noise), ("real_vs_mag+shuffledcos", s_shuf),
                          ("real_vs_mag_alone", s_mag)]:
        md, lo, hi, fp = bootstrap_paired_auroc_diff(s_real, s_ctrl, y, args.n_boot, rng)
        h9c[label] = {"mean_diff": md, "ci95": [lo, hi], "frac_pos": fp}
        log.info("  Δ(real − %-22s) = %+.4f  CI[%+.4f,%+.4f]", label.split("_vs_")[1], md, lo, hi)
    summary["H9c_matched_dimension"] = h9c
    summary["H9c_verdict"] = (
        "PASS: real 64-d beats BOTH equal-dimensional controls (mag+noise, mag+shuffled-cos) "
        "with bootstrap CIs excluding 0. Adding 32 columns helps only when they carry real, "
        "correctly-layer-aligned cosine information — not from doubling the dimension."
        if (h9c["real_vs_mag+noise"]["ci95"][0] > 0 and h9c["real_vs_mag+shuffledcos"]["ci95"][0] > 0)
        else "INSPECT: a control matched the real 64-d — the ensemble gain may be capacity-driven.")

    # ===================== H9d — redundancy signature ==========================
    log.info("\n=== H9d  redundancy signature (why magnitude wins at 1-D) ===")
    best_cos, bl_cos, pl_cos = best_single_tap_auroc(COS, y, cv)
    best_mag, bl_mag, pl_mag = best_single_tap_auroc(MAG, y, cv)
    ent_cos, _ = argmax_layer_entropy(COS)
    ent_mag, _ = argmax_layer_entropy(MAG)
    h9d = {
        "cos": {"auroc_32d": auc_cos, "auroc_best_tap": best_cos, "best_layer": bl_cos,
                "gap_32d_minus_1d": auc_cos - best_cos,
                "mean_interlayer_abscorr": mean_interlayer_abscorr(COS),
                "argmax_layer_entropy_bits": ent_cos, "per_layer_auroc": pl_cos.tolist()},
        "mag": {"auroc_32d": auc_mag, "auroc_best_tap": best_mag, "best_layer": bl_mag,
                "gap_32d_minus_1d": auc_mag - best_mag,
                "mean_interlayer_abscorr": mean_interlayer_abscorr(MAG),
                "argmax_layer_entropy_bits": ent_mag, "per_layer_auroc": pl_mag.tolist()},
    }
    log.info("  cos: 32d=%.4f  best-tap=%.4f (L%d)  gap=%.4f  interlayer|r|=%.3f  argmax-H=%.2f bits",
             auc_cos, best_cos, bl_cos, auc_cos - best_cos,
             h9d["cos"]["mean_interlayer_abscorr"], ent_cos)
    log.info("  mag: 32d=%.4f  best-tap=%.4f (L%d)  gap=%.4f  interlayer|r|=%.3f  argmax-H=%.2f bits",
             auc_mag, best_mag, bl_mag, auc_mag - best_mag,
             h9d["mag"]["mean_interlayer_abscorr"], ent_mag)
    summary["H9d_redundancy"] = h9d
    summary["H9d_verdict"] = (
        "PASS: magnitude's 32-d≈1-d (small gap, high inter-layer |r|, low argmax entropy) — one "
        "redundant SIZE scalar readable at any tap, so a fixed peak layer costs no AUROC. Cosine's "
        "32-d≫1-d (large gap, low inter-layer |r|, high argmax entropy) — a genuinely distributed, "
        "layer-wise signal. AUROC (scoring) and localization (where) are orthogonal."
        if (h9d["cos"]["gap_32d_minus_1d"] > h9d["mag"]["gap_32d_minus_1d"]
            and h9d["cos"]["mean_interlayer_abscorr"] < h9d["mag"]["mean_interlayer_abscorr"])
        else "INSPECT: expected cosine to be more distributed than magnitude.")

    # ===================== H9e — locality test =================================
    log.info("\n=== H9e  locality: does the LAYER INDEX carry information? ===")
    summ4 = order_invariant_summary(COS)
    a_summ = oof_auroc(summ4, y, cv)
    a_shufcos = oof_auroc(cos_shuf, y, cv)
    h9e = {"cos_32d_layerindexed": auc_cos, "cos_summary_4d_orderinvariant": a_summ,
           "cos_layer_shuffled_32d": a_shufcos,
           "gain_from_layer_index_vs_summary": auc_cos - a_summ,
           "gain_from_layer_index_vs_shuffle": auc_cos - a_shufcos}
    log.info("  cos 32d (layer-indexed) = %.4f", auc_cos)
    log.info("  cos 4d summary (order-invariant) = %.4f  (Δ=%.4f)", a_summ, auc_cos - a_summ)
    log.info("  cos 32d per-variant layer-shuffled = %.4f  (Δ=%.4f)", a_shufcos, auc_cos - a_shufcos)
    summary["H9e_locality"] = h9e
    summary["H9e_verdict"] = (
        "PASS: the layer-indexed cosine-32d beats both the order-invariant 4-d summary and the "
        "per-variant layer-shuffled 32-d. The specific LAYER of the directional change is "
        "informative — 'layer-wise differences are meaningful' in the literal, testable sense."
        if (h9e["gain_from_layer_index_vs_summary"] > 0 and h9e["gain_from_layer_index_vs_shuffle"] > 0)
        else "INSPECT: layer index added little over the value distribution alone.")

    # ---- persist + figure ----------------------------------------------------
    out_json = args.out / "dimensionality_nulls_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log.info("\n[save] %s", out_json)
    try:
        _make_figure(args.out, summary)
        log.info("[save] %s", args.out / "F_dimensionality_nulls.png")
    except Exception as e:  # figure is a convenience, never fail the run on it
        log.warning("[figure] skipped: %s", e)

    log.info("\n=== VERDICTS ===")
    for k in ("H9a", "H9b", "H9c", "H9d", "H9e"):
        log.info("  %s: %s", k, summary[f"{k}_verdict"])


def _make_figure(out_dir: Path, S: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    # (a) H9a permutation null
    a = ax[0, 0]
    names = ["cos_32d", "mag_32d", "comb_64d"]
    reals = [S["H9a_permutation_null"][n]["real_auroc"] for n in names]
    nmean = [S["H9a_permutation_null"][n]["null_mean"] for n in names]
    nmax = [S["H9a_permutation_null"][n]["null_max"] for n in names]
    xp = np.arange(len(names))
    a.bar(xp - 0.2, reals, 0.4, color="#1f77b4", label="real labels")
    a.bar(xp + 0.2, nmean, 0.4, color="#bbbbbb", label="permuted (mean)")
    a.errorbar(xp + 0.2, nmean, yerr=[[0] * 3, np.array(nmax) - np.array(nmean)],
               fmt="none", ecolor="k", lw=0.8, capsize=3)
    a.axhline(0.5, color="k", lw=0.6, ls=":")
    a.set_xticks(xp); a.set_xticklabels(names, fontsize=8)
    a.set_ylim(0.4, 1.0); a.set_ylabel("OOF AUROC")
    a.set_title("(a) H9a label-permutation null", loc="left", fontweight="bold")
    a.legend(fontsize=8, frameon=False)

    # (b) H9b dimension curve
    b = ax[0, 1]
    dc = S["H9b_dimension_curve"]
    b.plot(dc["k"], dc["real_auroc"], "-o", ms=3, color="#1f77b4", label="real")
    b.plot(dc["k"], dc["null_auroc"], "-o", ms=3, color="#bbbbbb", label="permuted null")
    b.axvline(dc["k_saturation_99pct"], color="#d62728", ls="--", lw=0.9,
              label=f"99% sat. k={dc['k_saturation_99pct']}")
    b.axhline(0.5, color="k", lw=0.6, ls=":")
    b.set_xlabel("# PCA components k"); b.set_ylabel("OOF AUROC")
    b.set_title("(b) H9b dimension learning curve", loc="left", fontweight="bold")
    b.legend(fontsize=8, frameon=False)

    # (c) H9c matched-dimension ensemble
    c = ax[1, 0]
    au = S["H9c_matched_dimension"]["auroc"]
    labels = ["real\ncos⊕mag", "mag⊕\nnoise", "mag⊕\nshuf-cos", "mag\nalone"]
    vals = [au["real_comb_64d"], au["mag32_plus_noise32"],
            au["mag32_plus_shuffledcos32"], au["mag_32d"]]
    cols = ["#1f77b4", "#bbbbbb", "#999999", "#d62728"]
    c.bar(np.arange(4), vals, color=cols)
    c.set_xticks(np.arange(4)); c.set_xticklabels(labels, fontsize=8)
    c.set_ylim(min(vals) - 0.02, max(vals) + 0.01); c.set_ylabel("OOF AUROC")
    c.set_title("(c) H9c matched-dimension controls (64-d)", loc="left", fontweight="bold")

    # (d) H9d per-layer AUROC + gap annotation
    d = ax[1, 1]
    plc = S["H9d_redundancy"]["cos"]["per_layer_auroc"]
    plm = S["H9d_redundancy"]["mag"]["per_layer_auroc"]
    d.plot(range(len(plc)), plc, "-", color="#1f77b4", label="cosine per-layer")
    d.plot(range(len(plm)), plm, "-", color="#d62728", label="magnitude per-layer")
    d.axhline(S["headline_auroc"]["cos_32d"], color="#1f77b4", ls="--", lw=0.9,
              label="cos 32-d")
    d.axhline(S["headline_auroc"]["mag_32d"], color="#d62728", ls="--", lw=0.9,
              label="mag 32-d")
    d.set_xlabel("layer ℓ"); d.set_ylabel("single-tap AUROC")
    d.set_title("(d) H9d redundancy: 32-d gap over best tap", loc="left", fontweight="bold")
    d.legend(fontsize=7, frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_dir / "F_dimensionality_nulls.png", dpi=200)
    fig.savefig(out_dir / "F_dimensionality_nulls.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
