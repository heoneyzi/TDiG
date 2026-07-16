"""wcl00 — shared utilities for the "Why Cosine Lens" (WCL) experiment suite.

REVISION v2 (grounded in the *actually-existing* public assets).
=================================================================
This module replaces the original wcl00 draft, which was written against a
`gDTR-PoC-main/` layout with `results/phase1.6/…` caches and a
`TDiG-main/src/tdig/metrics/*` package. Neither exists. The verified reality
(checked against the two public repos + the HF dataset card, 2026-07) is:

  * Paper-1 code lives in  github.com/darejinn/gDTR         (repo alias: gDTR)
      - src/gdtr.py          -> running_min / settling_depth_discrete / _interp
      - src/ur_gdtr_evo2.py  -> cosine_lens
      - src/calibration.py, src/stats.py, src/model_loader_evo2.py, logit_lens_evo2.py
      - paper/gdtr_paper_ICML_3.tex  -> the camera-ready source (Appendix G.1 etc.)
  * Phase-2 code lives in  github.com/YAICON-8th-Think-Deep-in-Genome/TDiG (alias: TDiG)
      - NO src/ package. Metric formulas are INLINE in scripts/15_chr22_forward.py
        (`compute_v2_settling`) and documented in METRICS_GUIDE.md.
      - data_cache_minimal_archive/  -> tier1 parquets (Stage 4, per-token c(t)
        for all 17 cells), variant_scalars.parquet (Stage 5), population_stats/
        (Stage 3: gamma_calibration_v2.json + sigma_ref_inv_{A,B,C}.npy).
      - results/*.csv  -> Stage-6 pre-computed aggregates (Cohen d, AUROC, CIs,
        chr22->chr17 retention, 21-pair context contest, L29 SVD, patching).
  * Raw hidden states live ONLY on HF: darejinn/TDiG-evo2-hidden-states
      - {chr22,chr17}_tier3_raw.h5 (47.7 GB): raw_h_ell (100,32,600,4096) fp32,
        raw_h_ell_rmsnormed fp16, raw_h_norm fp16, window_idx, token_stride, done_mask
        *** NOTE the token axis is 600, not 6000: tier3 stores a stride-10 subsample
            of each 6000-token window (token_stride). The tier2 SCALARS below are at
            full 6000 resolution. ***
      - {chr22,chr17}_tier2_scalars.h5 (614 MB): cos_refA/B/C fp16, norm_h_ell fp32,
        step_norm_raw/rms fp32, step_cos fp16, D_Mset_A/B/C fp32, norm_h_29/rms_h_29/h_norm fp32.
        Shapes are (100, 32, 6000) for the per-(window,layer,token) fields.
      - variant_h_ell_{ref,alt}.h5 (5.8 GB): h_ell (10910,32,4096) fp32,
        h_norm (10910,4096) fp16, done_mask (10910,) uint8.

Calibration facts (from TDiG METRICS_GUIDE.md, authoritative):
  * gamma = q70 of the metric distribution at a PER-METRIC anchor layer:
        M1 (dir) / M2 (mag) / M4 (set) : anchor L = 28
        M3 (geo)                        : anchor L = 26
        M5 (tau)                        : anchor L = 27
  * persistence W = 3 (three consecutive layers must satisfy threshold), except
    M4_set which is monotone-direct (no W).
  * settling search runs over layers 0..29 (max_layer=29), i.e. c(t) in [-1, 29]
    where -1 == "never settled".  This is TDiG's v2 protocol and does NOT equal
    the paper's gamma_cos=0.39663 (q70 at the penultimate layer for the cosine
    Ref-C construct). Numbers from tier1 are therefore an *internal*, apples-to-
    apples cross-lens comparison, not a reproduction of the paper's Fig. 2 scale.

This module has three parts:
  1. Cache/results readers (CPU, no GPU): tier1 parquet streaming, results CSVs.
  2. Lens formulas + a self-contained settling pipeline that reproduces TDiG's
     `compute_v2_settling` so *every* candidate lens can be scored through the
     identical rule (the §5 "same pipeline for every lens" requirement).
  3. HF readers for tier2 scalars / tier3 raw / variant hidden states, matched
     to the schema above.

Install:  pip install pyarrow pandas numpy scipy h5py scikit-learn --break-system-packages
(torch only needed for the GPU stages: variant hidden-state loads work with numpy/h5py.)
"""
from __future__ import annotations

import logging
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

N_LAYERS = 32           # Evo 2 7B
HIDDEN_SIZE = 4096
MAX_SETTLE_LAYER = 29   # settling searched over layers 0..29 (L*=29); 30=rotation, 31=passthrough
DEFAULT_W = 3           # persistence window
# per-metric gamma anchor layers (METRICS_GUIDE.md)
ANCHOR_LAYER = {"dir": 28, "mag": 28, "set": 28, "geo": 26, "tau": 27}

# The 17 cells present in {chr22,chr17}_tier1.parquet (verified against
# results/splice_vs_intron.csv column `cell`).
TIER1_CELLS = [
    "M1_dir_refA", "M1_dir_refB", "M1_dir_refC",
    "M2_mag_refA", "M2_mag_refB_diag", "M2_mag_refC_diag",
    "M3_geo_a0.0_b1.0", "M3_geo_a0.5_b1.0", "M3_geo_a1.0_b1.0",
    "M3_geo_a1.0_b0.5", "M3_geo_a1.0_b0.0",
    "M4_set_refA", "M4_set_refB", "M4_set_refC",
    "M5_tau_refA", "M5_tau_refB", "M5_tau_refC",
]
COSINE_CELL_PRIMARY = "M1_dir_refC"    # closest to the paper's D_cos construct (h_ell vs h_norm)
MAGNITUDE_CELL = "M2_mag_refA"          # the only non-degenerate magnitude reference


# --------------------------------------------------------------------------
# 1. Cache-reader helpers (CPU only; reuse TDiG's data_cache_minimal + results/)
# --------------------------------------------------------------------------

def iter_tier1_columns(parquet_path: str, columns: Sequence[str],
                       batch_size: int = 1024) -> Iterator["pyarrow.RecordBatch"]:  # noqa: F821
    """Stream a TDiG tier1 parquet in batches, materialising only `columns`.

    Each cell column is a list<int> of per-token settling depths; the full
    table expands to ~78M scalars per column, so never use a plain
    pandas.read_parquet on all 17 cells at once.
    """
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    cols = list(columns)
    if "window_idx" not in cols:
        cols = ["window_idx"] + cols
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        yield batch


def flatten_settling_column(parquet_path: str, column: str,
                            batch_size: int = 1024, dtype=np.int16,
                            drop_never: bool = False) -> np.ndarray:
    """Flatten one settling-depth list-column into a 1-D array (memory-safe).

    Settling ints are in [-1, 29] where -1 == never settled. Set
    drop_never=True to exclude -1 before computing CV / saturation stats.
    """
    parts: List[np.ndarray] = []
    for batch in iter_tier1_columns(parquet_path, [column], batch_size):
        col = batch.column(column)
        flat = col.flatten().to_numpy(zero_copy_only=False)
        parts.append(flat.astype(dtype))
    if not parts:
        return np.array([], dtype=dtype)
    out = np.concatenate(parts)
    if drop_never:
        out = out[out != -1]
    return out


def per_window_means(parquet_path: str, columns: Sequence[str],
                     batch_size: int = 1024) -> "pandas.DataFrame":  # noqa: F821
    """Per-window mean/std of settling depth (window-level context proxy)."""
    import pandas as pd
    rows = []
    for batch in iter_tier1_columns(parquet_path, list(columns), batch_size):
        wi = batch.column("window_idx").to_numpy()
        per_col = {c: batch.column(c) for c in columns}
        for i in range(batch.num_rows):
            rec = {"window_idx": int(wi[i])}
            ok = True
            for c, arr in per_col.items():
                if not arr[i].is_valid:
                    ok = False; break
                v = arr[i].values.to_numpy(zero_copy_only=False)
                v = v[v != -1]                     # drop never-settled
                if v.size == 0:
                    ok = False; break
                rec[f"mean_{c}"] = float(v.mean())
                rec[f"std_{c}"] = float(v.std())
            if ok:
                rows.append(rec)
    return pd.DataFrame(rows)


def load_window_metadata(parquet_path: str) -> "pandas.DataFrame":  # noqa: F821
    """Load {chr22,chr17}_metadata.parquet; add intron_frac / exon_frac if the
    per-window context counts are present."""
    import pandas as pd
    meta = pd.read_parquet(parquet_path)
    ctx_cols = ["n_coding_exon", "n_intron", "n_splice", "n_intergenic", "n_5utr", "n_3utr"]
    present = [c for c in ctx_cols if c in meta.columns]
    if present:
        total = meta[present].sum(axis=1) + 1e-9
        if "n_intron" in meta.columns:
            meta["intron_frac"] = meta["n_intron"] / total
        if "n_coding_exon" in meta.columns:
            meta["exon_frac"] = meta["n_coding_exon"] / total
    return meta


def load_results_csv(repo_root: str, rel_path: str) -> "pandas.DataFrame":  # noqa: F821
    """Convenience loader for a Stage-6 pre-computed results CSV, e.g.
    load_results_csv(TDIG_ROOT, 'results/splice_vs_intron.csv')."""
    import os
    import pandas as pd
    return pd.read_csv(os.path.join(repo_root, rel_path))


# --------------------------------------------------------------------------
# 2. Effect-size / diagnostic helpers
# --------------------------------------------------------------------------

def cohend(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d, positive => a larger than b (matches gDTR src/stats.py
    and TDiG scripts/13_analyze_chr22_v2.py::cohen_d)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    s_p = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    return float((a.mean() - b.mean()) / s_p) if s_p else float("nan")


def coefficient_of_variation(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    m = x.mean()
    return float(x.std() / m) if m != 0 else float("nan")


def saturation_fraction(x: np.ndarray, band: Tuple[int, int]) -> float:
    x = np.asarray(x)
    lo, hi = band
    return float(np.mean((x >= lo) & (x <= hi)))


# --------------------------------------------------------------------------
# 3. Lens formulas (require raw h_ell(t)) + settling pipeline
# --------------------------------------------------------------------------

def settling_persistence(D: np.ndarray, gamma: float, W: int = DEFAULT_W,
                         max_layer: int = MAX_SETTLE_LAYER) -> np.ndarray:
    """Reproduce TDiG's v2 settling rule (scripts/15_chr22_forward.py::
    compute_v2_settling) so ANY non-negative [L, T] distance can be scored the
    same way as the cosine lens.

    Rule: c(t) = first layer ell (0..max_layer) such that D[j, t] <= gamma for
    all j in [ell, ell+W-1]; -1 if no such run exists. This is the running-min
    envelope's discrete equivalent used for every WCL candidate lens.

    Args:
        D: [n_layers, n_tokens] non-negative distance (cosine dist, |r-1|, L2, ...).
        gamma: threshold.
        W: persistence window (use W=1 for the monotone-direct M4_set variant).
        max_layer: highest settling layer to consider (29 = L*).
    Returns:
        int32[n_tokens] settling depth in [-1, max_layer].
    """
    D = np.asarray(D, dtype=np.float64)
    L, T = D.shape
    below = D <= gamma                                    # [L, T] bool
    c = np.full(T, -1, dtype=np.int32)
    hi = min(max_layer, L - W)
    for ell in range(0, hi + 1):
        run = np.ones(T, dtype=bool)
        for j in range(ell, ell + W):
            run &= below[j]
        newly = run & (c == -1)
        c[newly] = ell
    return c


def calibrate_gamma(D: np.ndarray, anchor_layer: int, q: float = 0.70) -> float:
    """gamma = q-quantile of D at the anchor layer over all tokens (TDiG recipe).
    D is [n_layers, n_tokens]."""
    return float(np.quantile(np.asarray(D)[anchor_layer], q))


def run_settling_pipeline(D: np.ndarray, metric_kind: str = "dir",
                          gamma: Optional[float] = None, q: float = 0.70,
                          W: Optional[int] = None) -> Dict[str, object]:
    """Score a [n_layers, n_tokens] distance matrix through TDiG's settling rule
    with the correct per-metric anchor + persistence.

    metric_kind in {"dir","mag","set","geo","tau"} selects the anchor layer and,
    for "set", the monotone-direct (W=1) rule.
    """
    D = np.asarray(D, dtype=np.float64)
    anchor = ANCHOR_LAYER.get(metric_kind, 28)
    if gamma is None:
        gamma = calibrate_gamma(D, anchor, q)
        log.info("gamma[%s] = %.5f (q%d @ L%d)", metric_kind, gamma, int(q * 100), anchor)
    w = W if W is not None else (1 if metric_kind == "set" else DEFAULT_W)
    c = settling_persistence(D, gamma, W=w)
    settled = c[c != -1]
    return {"gamma": gamma, "anchor_layer": anchor, "W": w, "c": c,
            "frac_never": float((c == -1).mean()),
            "mean_settled": float(settled.mean()) if settled.size else float("nan"),
            "std_settled": float(settled.std()) if settled.size else float("nan")}


def d_cos_lens_from_norm(cos_to_ref: np.ndarray) -> np.ndarray:
    """Cosine DISTANCE lens from a stored cosine-similarity (or the tier2
    cos_ref* field). D_cos = 1 - cos. Accepts [L, T] or [W, L, T]."""
    return 1.0 - np.asarray(cos_to_ref, dtype=np.float64)


def d_mag_lens_from_norms(norm_h_ell: np.ndarray, ref_layer: int = 29,
                          mode: str = "abs") -> np.ndarray:
    """Magnitude lens |r-1| (M2, Ref A) from per-layer norms alone — pure numpy,
    no raw h needed. Accepts [L, T] or [W, L, T].

    D_mag(ell,t) = |‖h_ell‖/‖h_ref‖ - 1|  (mode='abs')  or  |log r| (mode='log').
    """
    a = np.asarray(norm_h_ell, dtype=np.float64)
    if a.ndim == 2:      ref = a[ref_layer:ref_layer + 1, :]
    elif a.ndim == 3:    ref = a[:, ref_layer:ref_layer + 1, :]
    else:                raise ValueError(f"expected 2-D/3-D, got {a.shape}")
    r = a / np.clip(ref, 1e-12, None)
    return np.abs(r - 1.0) if mode == "abs" else np.abs(np.log(np.clip(r, 1e-12, None)))


def d_l2_lens_from_h(h_ell: np.ndarray, ref_layer: int = 29) -> np.ndarray:
    """Relative L2 lens (M6) from raw hidden states [L, T, H]:
    D_L2(ell,t) = ‖h_ell - h_ref‖ / ‖h_ref‖. Returns [L, T]."""
    h = np.asarray(h_ell, dtype=np.float64)          # [L, T, H]
    ref = h[ref_layer]                               # [T, H]
    ref_norm = np.linalg.norm(ref, axis=-1)          # [T]
    diff = np.linalg.norm(h - ref[None], axis=-1)    # [L, T]
    return diff / np.clip(ref_norm[None], 1e-12, None)


# --------------------------------------------------------------------------
# 4. HuggingFace readers — darejinn/TDiG-evo2-hidden-states (verified schema)
# --------------------------------------------------------------------------

HF_REPO_ID = "darejinn/TDiG-evo2-hidden-states"
HF_FILES = {
    "chr22_tier2": "chr22_tier2_scalars.h5",   # 614 MB  <- Exp 1 (magnitude, CPU-only)
    "chr17_tier2": "chr17_tier2_scalars.h5",   # 614 MB
    "variant_ref": "variant_h_ell_ref.h5",     # 5.8 GB  <- Exp 3b (variant decomposition)
    "variant_alt": "variant_h_ell_alt.h5",     # 5.8 GB
    "chr22_tier3": "chr22_tier3_raw.h5",       # 47.7 GB <- Exp 2 (rogue-dim, needs raw h)
    "chr17_tier3": "chr17_tier3_raw.h5",       # 47.7 GB
}


def download_hf_file(key: str, local_dir: str, revision: str = "main") -> str:
    """huggingface_hub download (resumable, checksum-verified). Run on a box with
    unrestricted egress — a tool sandbox may 403 on the large-binary resolve URL.
    CLI equivalent: huggingface-cli download darejinn/TDiG-evo2-hidden-states
    <filename> --repo-type dataset --local-dir <local_dir>."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset",
                           filename=HF_FILES[key], revision=revision, local_dir=local_dir)
    log.info("Downloaded %s -> %s", HF_FILES[key], path)
    return path


def load_tier2_scalars(h5_path: str,
                       fields: Sequence[str] = ("cos_refC", "norm_h_ell"),
                       window_slice: Optional[slice] = None) -> Dict[str, np.ndarray]:
    """Load selected (100, 32, 6000) fields from *_tier2_scalars.h5.

    Highest-value low-cost download for Experiment 1: `norm_h_ell` lets you
    compute the magnitude lens with plain numpy, and `cos_refA/B/C` are the
    cosine SIMILARITY fields (D_cos = 1 - cos; verify sign on first use). Load
    1-2 fields at a time (each ~230-460 MB); use window_slice to page windows.
    """
    import h5py
    out: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        log.info("tier2 keys: %s", list(f.keys()))
        for name in fields:
            if name not in f:
                raise KeyError(f"{name!r} not in {list(f.keys())}")
            out[name] = f[name][window_slice] if window_slice is not None else f[name][:]
    return out


def load_tier3_raw(h5_path: str, field: str = "raw_h_ell_rmsnormed",
                   window_idx: Optional[int] = None,
                   layer: Optional[int] = None) -> np.ndarray:
    """Load from *_tier3_raw.h5 (100, 32, 600, 4096). ALWAYS slice — the full
    array is ~24 GB in fp32. `field` in {raw_h_ell, raw_h_ell_rmsnormed, raw_h_norm}.

    Examples:
        load_tier3_raw(p, "raw_h_ell", window_idx=0, layer=29)  # [600, 4096] one window/layer
        load_tier3_raw(p, "raw_h_ell", window_idx=0)            # [32, 600, 4096] one window, all layers
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        if field not in f:
            raise KeyError(f"{field!r} not in {list(f.keys())}")
        dset = f[field]
        if window_idx is None:
            raise ValueError("window_idx is required — never load the full 24 GB array")
        if field == "raw_h_norm":                       # (100, 600, 4096)
            return dset[window_idx]
        if layer is None:
            return dset[window_idx]                      # (32, 600, 4096)
        return dset[window_idx, layer]                   # (600, 4096)


def load_variant_hidden_states(h5_ref_path: str, h5_alt_path: str,
                               layer: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load ref/alt hidden states from variant_h_ell_{ref,alt}.h5 (10910,32,4096).

    Pass `layer` to load one layer slice ([10910,4096], ~180 MB fp32) instead of
    the full (10910,32,4096) (~5.7 GB). Row order matches variant_scalars.parquet
    row order (both come from scripts/18_variant_forward.py's write loop) — join
    on row index, and spot-check ‖h_alt-h_ref‖ against variant_scalars'
    delta_h_norm_2 for a few rows before trusting the alignment.
    """
    import h5py
    with h5py.File(h5_ref_path, "r") as fr, h5py.File(h5_alt_path, "r") as fa:
        if layer is not None:
            return fr["h_ell"][:, layer, :], fa["h_ell"][:, layer, :]
        return fr["h_ell"][:], fa["h_ell"][:]


def per_dimension_variance_concentration(h_layer: np.ndarray,
                                         top_ks: Sequence[int] = (1, 8, 64)) -> Dict[str, float]:
    """Rogue-dimension diagnostic (Experiment 2): fraction of total variance the
    top-k highest-variance dimensions explain, for a [n_tokens, HIDDEN] sample of
    one layer's raw hidden states. (Timkey & van Schijndel 2021 test.)"""
    var_per_dim = np.asarray(h_layer).var(axis=0, ddof=1)          # [HIDDEN]
    order = np.argsort(-var_per_dim)
    total = float(var_per_dim.sum())
    out = {"total_variance": total}
    sorted_var = var_per_dim[order]
    for k in sorted(top_ks):
        out[f"top_{k}_frac"] = float(sorted_var[:k].sum()) / total if total > 0 else float("nan")
    return out


def covariance_spectrum(h_layer: np.ndarray, k: int = 64) -> Dict[str, object]:
    """Covariance eigen-spectrum of a [n_tokens, HIDDEN] layer sample (Experiment
    2, the covariance analogue of TDiG's transition-matrix SVD in
    results/L29_svd/). Returns the top-k eigenvalue share + an isotropy score."""
    x = np.asarray(h_layer, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    # economy SVD of centred data: eigenvalues of Cov = s^2 / (n-1)
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    ev = (s ** 2) / (x.shape[0] - 1)
    total = float(ev.sum())
    frac = ev / total if total > 0 else ev
    return {"top1_frac": float(frac[0]), "top8_frac": float(frac[:8].sum()),
            "top64_frac": float(frac[:min(k, ev.size)].sum()),
            "isotropy_score": float(ev.min() / ev.max()) if ev.max() > 0 else float("nan"),
            "effective_rank": float(np.exp(-(frac[frac > 0] * np.log(frac[frac > 0])).sum()))}
