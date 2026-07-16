#!/usr/bin/env python3
"""wcl03b — Experiment 3 (paper-grade): localization vs. scoring, from the raw
variant hidden states on HF (the piece the CPU pilot wcl03 could not do).

Resolves the single hardest reviewer question: if ‖Δh‖_2 beats ΔD_cos on the
paper's own ClinVar cohort (Table A8: 0.926 vs 0.844), why does the paper use
cosine at all? Answer to demonstrate (not assert): the two features do
different jobs — magnitude scores HOW MUCH a variant perturbs (dominated by the
L30 rotation spike), direction localizes WHERE computation commits.

DATA:  HF darejinn/TDiG-evo2-hidden-states ::
         variant_h_ell_ref.h5, variant_h_ell_alt.h5  (10910, 32, 4096) fp32
       TDiG data_cache_minimal :: variant_scalars.parquet (category/gene/consequence,
         + cross-check columns delta_cos, delta_h_norm_2).

Row i of the h5 files == row i of variant_scalars.parquet (both written in order
by scripts/18_variant_forward.py). We verify the alignment before use.

H3a  complementarity : joint (ΔD_cos ⊕ ‖Δh‖) beats ‖Δh‖ alone by DeLong, even
     though ‖Δh‖ alone beats ΔD_cos alone.
H3b  localization structure : argmax layer of ‖Δh‖_2 is a near-constant spike
     (expected ~L30), while argmax of |ΔD_cos| is spread and biology-ordered.
H3c  disruption-size interpretation : ‖Δh‖'s argmax correlates with total
     disruption size, not with |ΔD_cos|'s argmax (Spearman < 0.4).
NEW  direction/magnitude isolation : decompose Δh_ell = (magnitude change) +
     (direction change) and show which half carries the AUROC.

Run (after downloading the two 5.8 GB variant files):
    python wcl03b_variant_hidden_decomposition.py \
        --ref variant_h_ell_ref.h5 --alt variant_h_ell_alt.h5 \
        --scalars data_cache_minimal/variant_scalars.parquet \
        --out results/wcl/exp3b/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wcl00_shared_lens_utils import N_LAYERS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("wcl03b")


def per_layer_features(h_ref: np.ndarray, h_alt: np.ndarray, ref_key_layer: int = 29):
    """From [N, L, H] ref/alt hidden states compute per-layer features [N, L]:
      dcos  : 1 - cos(h_alt_ell, h_ref_ell)                  (direction change)
      dmag  : | ‖h_alt_ell‖ - ‖h_ref_ell‖ | / ‖h_ref_ell‖    (pure magnitude change)
      dl2   : ‖h_alt_ell - h_ref_ell‖ / ‖h_ref_ell‖          (total, = Paper 2's ‖Δh‖ up to norm)
    Uses fp64 accumulation for numerical stability across Evo 2's huge late norms.
    """
    hr = h_ref.astype(np.float64); ha = h_alt.astype(np.float64)
    nr = np.linalg.norm(hr, axis=-1); na = np.linalg.norm(ha, axis=-1)          # [N, L]
    dot = np.einsum("nld,nld->nl", hr, ha)
    cos = dot / np.clip(nr * na, 1e-12, None)
    dcos = 1.0 - cos
    dmag = np.abs(na - nr) / np.clip(nr, 1e-12, None)
    dl2 = np.linalg.norm(ha - hr, axis=-1) / np.clip(nr, 1e-12, None)
    return {"dcos": dcos, "dmag": dmag, "dl2": dl2, "nr": nr, "na": na}


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def cv_auroc(X, y, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return oof, auroc(y, oof)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True); p.add_argument("--alt", required=True)
    p.add_argument("--scalars", required=True, help="variant_scalars.parquet")
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=1000); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    import h5py
    import pandas as pd

    meta = pd.read_parquet(args.scalars)
    log.info("variant_scalars rows: %d  columns: %s", len(meta), list(meta.columns)[:12])

    with h5py.File(args.ref, "r") as fr, h5py.File(args.alt, "r") as fa:
        N = fr["h_ell"].shape[0]
        assert fr["h_ell"].ndim == 3 and fr["h_ell"].shape[1] == N_LAYERS, fr["h_ell"].shape
        if len(meta) != N:
            log.warning("row count mismatch: scalars=%d h5=%d — aligning on min", len(meta), N)
        n = min(N, len(meta))
        h_ref = fr["h_ell"][:n].astype(np.float32)
        h_alt = fa["h_ell"][:n].astype(np.float32)
    meta = meta.iloc[:n].reset_index(drop=True)

    feats = per_layer_features(h_ref, h_alt)
    # ---- alignment sanity check against precomputed delta_h_norm_2 ----
    if "delta_h_norm_2" in meta.columns:
        stored = np.asarray(meta["delta_h_norm_2"].tolist(), dtype=np.float64)   # [n, 32]
        recomputed_abs = np.linalg.norm(h_alt.astype(np.float64) - h_ref.astype(np.float64), axis=-1)
        # compare on a few rows/layers (stored may be raw ‖Δh‖, ours is normalised — check correlation)
        from scipy.stats import spearmanr
        rho = float(spearmanr(recomputed_abs[:200].ravel(), stored[:200].ravel()).statistic)
        log.info("alignment Spearman(recomputed ‖Δh‖, stored delta_h_norm_2) = %.3f", rho)
    else:
        rho = None

    # binary cohort
    m = meta["category"].isin(["P_LP", "B_LB"])
    y = (meta.loc[m, "category"] == "P_LP").astype(int).to_numpy()
    D = feats["dcos"][m.to_numpy()]     # [nb, 32]
    Mabs = feats["dl2"][m.to_numpy()]   # total ‖Δh‖-family (Paper 2's magnitude feature)
    Mpure = feats["dmag"][m.to_numpy()] # pure magnitude change
    log.info("binary cohort n=%d (P_LP=%d B_LB=%d)", len(y), int(y.sum()), int((1 - y).sum()))

    # ---- H3a complementarity ----
    oof, aur = {}, {}
    for name, X in [("D_cos_32d", D), ("M_l2_32d", Mabs), ("M_magpure_32d", Mpure),
                    ("D_plus_M_64d", np.concatenate([D, Mabs], axis=1))]:
        oof[name], aur[name] = cv_auroc(X, y, args.seed)
        log.info("%-16s AUROC=%.4f", name, aur[name])

    def boot_diff(a, b):
        n = len(y); d = np.empty(args.n_boot)
        for i in range(args.n_boot):
            idx = rng.integers(0, n, n)
            if len(np.unique(y[idx])) < 2:
                d[i] = np.nan; continue
            d[i] = auroc(y[idx], a[idx]) - auroc(y[idx], b[idx])
        d = d[~np.isnan(d)]
        return {"mean_diff": float(d.mean()), "ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))]}

    H3a = {"D+M_vs_M": boot_diff(oof["D_plus_M_64d"], oof["M_l2_32d"]),
           "D+M_vs_D": boot_diff(oof["D_plus_M_64d"], oof["D_cos_32d"])}

    # ---- H3b/H3c localization ----
    amax_cos = np.abs(D).argmax(1); amax_mag = np.abs(Mabs).argmax(1)
    total_disrupt = np.abs(Mabs).max(1)
    from scipy.stats import spearmanr
    H3c = {
        "spearman_argmaxMag_vs_totalDisruption": float(spearmanr(amax_mag, total_disrupt).statistic),
        "spearman_argmaxMag_vs_argmaxCos": float(spearmanr(amax_mag, amax_cos).statistic),
    }
    localization = {
        "argmax_cos_mean": float(amax_cos.mean()), "argmax_cos_std": float(amax_cos.std()),
        "argmax_mag_mean": float(amax_mag.mean()), "argmax_mag_std": float(amax_mag.std()),
        "argmax_mag_modal_layer": int(np.bincount(amax_mag).argmax()),
        "argmax_mag_frac_modal": float((amax_mag == np.bincount(amax_mag).argmax()).mean()),
    }
    log.info("argmax(|ΔD_cos|): mean=%.2f std=%.2f  |  argmax(‖Δh‖): mean=%.2f std=%.2f modal=L%d (%.0f%%)",
             localization["argmax_cos_mean"], localization["argmax_cos_std"],
             localization["argmax_mag_mean"], localization["argmax_mag_std"],
             localization["argmax_mag_modal_layer"], 100 * localization["argmax_mag_frac_modal"])

    # ---- direction/magnitude AUROC isolation, per layer ----
    per_layer = {"layer": list(range(N_LAYERS)),
                 "auroc_dcos": [auroc(y, D[:, l]) for l in range(N_LAYERS)],
                 "auroc_dl2": [auroc(y, Mabs[:, l]) for l in range(N_LAYERS)],
                 "auroc_dmag_pure": [auroc(y, Mpure[:, l]) for l in range(N_LAYERS)]}

    summary = {
        "n_variants": int(len(y)), "alignment_spearman": rho,
        "auroc": aur, "H3a_complementarity": H3a, "H3b_localization": localization,
        "H3c_interpretation": H3c, "per_layer_auroc": per_layer,
        "note": "Uses HF variant_h_ell_{ref,alt}.h5 (paper-grade); cross-checked against "
                "variant_scalars.parquet. Absolute AUROC follows TDiG's 18_variant_forward cohort; "
                "compare deltas/shapes to the paper's Table A8, not absolute values.",
    }
    (out / "decomposition_summary.json").write_text(json.dumps(summary, indent=2))
    _plot(out, per_layer, amax_cos, amax_mag)
    log.info("Done -> %s", out / "decomposition_summary.json")


def _plot(out, pl, amax_cos, amax_mag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(pl["layer"], pl["auroc_dcos"], "-o", ms=3, label="ΔD_cos (direction)")
    a1.plot(pl["layer"], pl["auroc_dl2"], "-s", ms=3, label="‖Δh‖ (total)")
    a1.axvline(29, ls=":", c="red", lw=1); a1.axvline(30, ls=":", c="orange", lw=1)
    a1.set_xlabel("layer ℓ"); a1.set_ylabel("single-layer AUROC"); a1.legend(fontsize=8)
    a1.set_title("Per-layer discriminative mass")
    a2.hist(amax_cos, bins=range(0, 33), alpha=.6, label="argmax |ΔD_cos|")
    a2.hist(amax_mag, bins=range(0, 33), alpha=.6, label="argmax ‖Δh‖")
    a2.set_xlabel("peak-disruption layer"); a2.set_ylabel("# variants"); a2.legend(fontsize=8)
    a2.set_title("Localization: spread (cosine) vs spike (magnitude)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"F_localization_vs_scoring.{ext}", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
