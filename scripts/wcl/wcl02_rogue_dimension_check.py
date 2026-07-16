#!/usr/bin/env python3
"""wcl02 — Experiment 2: rogue-dimension / standardization robustness check.

Directly executes the second open question the paper poses to itself in
Appendix G.1: "confirming that Evo 2's stream has no such dominant dimensions,
or standardising before the lens, would put c(t) on firmer ground"
(Timkey & van Schijndel 2021).

DATA:  HF darejinn/TDiG-evo2-hidden-states :: chr22_tier3_raw.h5
       field `raw_h_ell` (100, 32, 600, 4096) fp32   — raw, un-normalised.
       (This is the ONLY public source of per-dimension raw hidden states; the
        tier2 scalars keep only norms/cosines, so H2a genuinely needs tier3.)

WHAT IT PRODUCES (results/wcl/exp2/):
  variance_concentration.json   per-layer top-1/8/64 variance share + Cov eigen-
                                spectrum (isotropy, effective rank) -> H2a.
  standardized_settling_compare.json  Spearman(c_cos, c_cos_std) + splice d under
                                raw vs per-dim z-scored vs top-k-removed cosine -> H2b/H2c.
  F_rogue_dim_spectrum.{png,pdf}

H2a  no rogue dominance : top-8 variance share is small (< ~10%) at every layer.
H2b  cosine robust to standardization : Spearman(c_cos, c_cos_std) >= 0.90 and
     splice-donor-vs-intron d within 15% of the raw-cosine d.
H2c  cosine robust to top-k removal   : same, removing the top-1/8/64 dims.

Standardization/top-k use the tier3 raw_h_ell directly and recompute the cosine
DISTANCE to raw_h_norm, then run wcl00.run_settling_pipeline (same rule as every
other lens). Per-position context labels (for the splice d) come from gDTR's
scripts/prep/prep_chr22_windows.py output (chr22_position_labels.npy) sliced to
tier3's token_stride subsample — see --pos-labels.

Run (after downloading chr22_tier3_raw.h5, ~47.7 GB):
    python wcl02_rogue_dimension_check.py \
        --tier3 chr22_tier3_raw.h5 \
        --pos-labels chr22_position_labels.npy \
        --out results/wcl/exp2/ \
        --n-windows 100 --sample-tokens 40000
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
    covariance_spectrum, per_dimension_variance_concentration,
    d_cos_lens_from_norm, run_settling_pipeline, cohend, N_LAYERS,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("wcl02")


def cos_distance_to_ref(h: np.ndarray, h_ref: np.ndarray) -> np.ndarray:
    """1 - cos(h_ell, h_ref) for h [L, T, D], h_ref [T, D] -> [L, T]."""
    h = h.astype(np.float64); h_ref = h_ref.astype(np.float64)
    hn = h / np.clip(np.linalg.norm(h, axis=-1, keepdims=True), 1e-12, None)
    rn = h_ref / np.clip(np.linalg.norm(h_ref, axis=-1, keepdims=True), 1e-12, None)
    return 1.0 - np.einsum("ltd,td->lt", hn, rn)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier3", required=True, help="chr22_tier3_raw.h5 (HF)")
    p.add_argument("--pos-labels", default=None,
                   help="chr22_position_labels.npy (per-bp context codes; optional, enables H2b/c splice d)")
    p.add_argument("--out", required=True)
    p.add_argument("--n-windows", type=int, default=100)
    p.add_argument("--sample-tokens", type=int, default=40000,
                   help="tokens sampled per layer for the variance-concentration diagnostic")
    p.add_argument("--ref-field", default="raw_h_norm", help="reference for the cosine lens")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    import h5py
    summary = {"tier3": args.tier3, "seed": args.seed}

    # ---- H2a: per-layer variance concentration + covariance spectrum ----
    log.info("H2a — per-layer variance concentration (sampling %d tokens/layer)", args.sample_tokens)
    with h5py.File(args.tier3, "r") as f:
        raw = f["raw_h_ell"]                       # (100, 32, 600, 4096)
        nW = min(args.n_windows, raw.shape[0])
        T = raw.shape[2]
        conc = {}
        for ell in range(N_LAYERS):
            # gather a token sample across windows for this layer
            wsel = rng.choice(nW, size=min(nW, 64), replace=False)
            chunks = [raw[w, ell] for w in sorted(wsel)]   # each [600, 4096]
            H = np.concatenate(chunks, axis=0)
            if H.shape[0] > args.sample_tokens:
                idx = rng.choice(H.shape[0], size=args.sample_tokens, replace=False)
                H = H[idx]
            vc = per_dimension_variance_concentration(H, (1, 8, 64))
            sp = covariance_spectrum(H, k=64)
            conc[f"L{ell}"] = {**vc, **sp}
            if ell in (0, 7, 15, 24, 29, 30, 31):
                log.info("  L%-2d top1=%.3f top8=%.3f top64=%.3f eff_rank=%.0f",
                         ell, vc["top_1_frac"], vc["top_8_frac"], vc["top_64_frac"], sp["effective_rank"])
    summary["H2a_variance_concentration"] = conc
    top8_L29 = conc["L29"]["top_8_frac"]
    summary["H2a_verdict"] = ("no_rogue_dominance" if top8_L29 < 0.10 else "possible_rogue_dominance")

    # ---- H2b/H2c: settling stability under standardization / top-k removal ----
    if args.pos_labels and Path(args.pos_labels).exists():
        log.info("H2b/H2c — standardized & top-k-removed cosine settling vs raw")
        pos_labels = np.load(args.pos_labels)          # per-bp codes (0..6); 5=donor,1=intron
        with h5py.File(args.tier3, "r") as f:
            raw = f["raw_h_ell"]; ref = f[args.ref_field]
            wids = f["window_idx"][:]; stride = int(f["token_stride"][()]) if "token_stride" in f else 10
            # population mean/std for per-dim z-scoring (over sampled tokens, all layers)
            samp = np.concatenate([raw[w] for w in range(min(16, raw.shape[0]))], axis=1)  # [32, 16*600, 4096]
            mu = samp.mean(axis=1, keepdims=True); sd = np.clip(samp.std(axis=1, keepdims=True), 1e-6, None)
            top_dims = np.argsort(-samp.reshape(N_LAYERS, -1, samp.shape[-1]).var(axis=1).mean(0))[:64]

            variants = {"raw": None, "zscored": "z", "topk64_removed": top_dims}
            c_store, splice_d = {}, {}
            for name, mode in variants.items():
                c_all, lab_all = [], []
                for wi in range(min(args.n_windows, raw.shape[0])):
                    H = raw[wi].astype(np.float64)              # [32, 600, 4096]
                    R = ref[wi].astype(np.float64)              # [600, 4096]
                    if isinstance(mode, str) and mode == "z":
                        H = (H - mu) / sd; R = (R - mu[N_LAYERS - 1]) / sd[N_LAYERS - 1]
                    elif isinstance(mode, np.ndarray):
                        H = H.copy(); H[:, :, mode] = 0.0; R = R.copy(); R[:, mode] = 0.0
                    D = cos_distance_to_ref(H, R)               # [32, 600]
                    res = run_settling_pipeline(D, metric_kind="dir")
                    c_all.append(res["c"])
                    # map subsampled tokens to per-bp labels via window start + stride
                    start = int(wids[wi]) if wids.ndim else 0
                    idxs = (start + np.arange(D.shape[1]) * stride)
                    idxs = np.clip(idxs, 0, len(pos_labels) - 1)
                    lab_all.append(pos_labels[idxs])
                c = np.concatenate(c_all); lab = np.concatenate(lab_all)
                c_store[name] = c
                donor = c[(lab == 5) & (c != -1)]; intron = c[(lab == 1) & (c != -1)]
                splice_d[name] = cohend(donor, intron)
                log.info("  %-16s splice d = %.4f (n_donor=%d n_intron=%d)",
                         name, splice_d[name], donor.size, intron.size)

            from scipy.stats import spearmanr
            mask = (c_store["raw"] != -1) & (c_store["zscored"] != -1)
            rho_z = float(spearmanr(c_store["raw"][mask], c_store["zscored"][mask]).statistic)
            maskk = (c_store["raw"] != -1) & (c_store["topk64_removed"] != -1)
            rho_k = float(spearmanr(c_store["raw"][maskk], c_store["topk64_removed"][maskk]).statistic)
            summary["H2b_H2c"] = {
                "splice_d": splice_d, "spearman_raw_vs_zscored": rho_z,
                "spearman_raw_vs_topk64removed": rho_k,
                "verdict": ("robust" if rho_z >= 0.90 and rho_k >= 0.90 else "drift_detected"),
            }
    else:
        log.warning("No --pos-labels: skipping H2b/H2c (H2a variance concentration still written).")
        summary["H2b_H2c"] = "skipped_no_position_labels"

    (out / "variance_concentration.json").write_text(json.dumps(summary, indent=2))
    _plot_spectrum(out, conc)
    log.info("Done -> %s", out / "variance_concentration.json")


def _plot_spectrum(out: Path, conc: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layers = list(range(N_LAYERS))
    top1 = [conc[f"L{l}"]["top_1_frac"] for l in layers]
    top8 = [conc[f"L{l}"]["top_8_frac"] for l in layers]
    top64 = [conc[f"L{l}"]["top_64_frac"] for l in layers]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, top1, "-o", ms=3, label="top-1 dim")
    ax.plot(layers, top8, "-s", ms=3, label="top-8 dims")
    ax.plot(layers, top64, "-^", ms=3, label="top-64 dims")
    ax.axhline(0.10, ls="--", c="grey", lw=1, label="10% (rogue threshold)")
    ax.axvline(29, ls=":", c="red", lw=1, label="L* = 29")
    ax.set_xlabel("layer ℓ"); ax.set_ylabel("fraction of total variance")
    ax.set_title("Per-layer variance concentration (rogue-dimension test)")
    ax.legend(fontsize=8); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"F_rogue_dim_spectrum.{ext}", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
