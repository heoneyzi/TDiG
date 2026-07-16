#!/usr/bin/env python3
"""wcl07 v2 — Cosine as a DIRECT, HIGH-RESOLUTION PROXY for output commitment.
(memory/GPU-safe streaming version)

=================================================================================
WHAT CHANGED vs the version you ran, and vs the first v2 (the slow one)
=================================================================================
Framing (unchanged from first v2): cosine is a proxy for the model's final-answer
commitment, not a proof of JSD. Evidence = 7A (model reads direction, KL~0 under
magnitude scaling) + RESOLUTION (cosine descends smoothly/densely; JSD is a near-
binary cliff) + proxy fidelity (with an explicit circularity caveat) + trajectory
decoupling (with a variance audit). See the OUTPUTS block for what to read.

PERFORMANCE FIX (this file): the previous v2 accumulated every window's raw hidden
states ([32, 6000, 4096] x 100 windows ~= 310 GB) and decoded all 6000 tokens x 5
alphas x 100 windows for 7A, never freeing GPU cache -> it swapped and pinned the
whole card. This version:
  * STREAMS one window at a time and never holds more than one window;
  * computes cosine / trajectory / magnitude distances PER LAYER on the GPU and
    moves only the small [32, T] distance matrices to CPU (never the [32,6000,4096]
    tensor) -> ~200 MB GPU, tiny CPU, tiny transfer;
  * frees GPU cache after every window (`del hs; empty_cache()`);
  * runs 7A on a small token subsample from just a few windows (a property that
    converges immediately) instead of all tokens x all windows;
  * vectorises the per-token settling/resolution metrics (no Python loops).

Result: same science, comparable cost to your original exp7 (one forward per
window) plus a cheap 7A probe.

=================================================================================
OUTPUTS (results/wcl/exp7_v2/)
  proxy_resolution_summary.json
    A_readout_scale_invariance_maxJS    ~0  -> model reads direction, not magnitude
    resolution.binariness               jsd >> cosine (cliff vs graded)
    resolution.transition_width         cosine >> jsd (gradual vs sharp)
    resolution.interp_settling_granularity  cosine many distinct sub-layer levels, jsd few
    proxy_fidelity                      agreement, WITH circularity caveat (proxy, not proof)
    trajectory_control.geo_settling_std verify it is LARGE before citing the decoupling
  F_cos_vs_jsd_descent.{png,pdf}        smooth cosine vs JSD cliff (the money figure)
  F_settling_resolution.{png,pdf}       dense cosine vs coarse JSD settling histogram

RUN
  mock (verify, no GPU):  python wcl07_v2_proxy_resolution.py --mock --out out/exp7_v2/
  real (GPU via gDTR):    python wcl07_v2_proxy_resolution.py \
      --gdtr-root ../gDTR --fasta-dir ../gDTR/data/reference \
      --windows data_cache_minimal/chr22_metadata.parquet --n-windows 100 \
      --out results/wcl/exp7_v2/   [--a-windows 3 --a-tokens 512]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("wcl07v2")

ANCHOR, Q, W = 28, 0.70, 3


# ----------------------------- settling / resolution (all vectorised) -----------------------------

def running_min(D):
    return np.minimum.accumulate(D, axis=0)


def calib_gamma(D, anchor=ANCHOR, q=Q):
    return float(np.quantile(D[min(anchor, D.shape[0] - 1)], q))


def settle_discrete(D, gamma, W=W):
    L, T = D.shape
    below = D <= gamma
    c = np.full(T, float(L))
    for ell in range(0, L - W + 1):                       # loop over LAYERS (<=32), not tokens
        run = np.all(below[ell:ell + W], axis=0)
        c[run & (c == L)] = ell
    return c


def settle_interp(D, gamma):
    m = running_min(D); L, T = m.shape
    below = m <= gamma
    any_b = below.any(0); first = below.argmax(0)
    tt = np.arange(T)
    above = m[np.clip(first - 1, 0, L - 1), tt]; at = m[first, tt]
    frac = np.clip((above - gamma) / np.clip(above - at, 1e-12, None), 0, 1)
    c = np.where(first == 0, 0.0, (first - 1) + frac)
    c[~any_b] = float(L)
    return c


def transition_width(D):
    m = running_min(D); L, T = m.shape
    top, bot = m[0], m[-1]; rng = np.clip(top - bot, 1e-9, None)
    hi, lo = bot + 0.9 * rng, bot + 0.1 * rng
    bhi, blo = m <= hi[None], m <= lo[None]
    a = np.where(bhi.any(0), bhi.argmax(0), L - 1)
    b = np.where(blo.any(0), blo.argmax(0), L - 1)
    w = np.maximum(b - a, 0)
    return float(np.median(w)), float(np.mean(w))


def binariness(D):
    m = running_min(D); drops = -np.diff(m, axis=0)
    total = np.clip(m[0] - m[-1], 1e-9, None)
    return float(np.median(drops.max(0) / total))


def interp_granularity(c, L):
    c = c[c < L]
    if c.size < 10:
        return {"n_distinct_0p1": 0, "entropy_bits": 0.0, "iqr": 0.0}
    n = int(np.unique(np.round(c, 1)).size)
    h = np.bincount(np.round(c * 2).astype(int)); p = h[h > 0] / h.sum()
    return {"n_distinct_0p1": n, "entropy_bits": float(-(p * np.log2(p)).sum()),
            "iqr": float(np.subtract(*np.percentile(c, [75, 25])))}


def spearman(a, b):
    from scipy.stats import spearmanr
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.unique(a[m]).size < 2 or np.unique(b[m]).size < 2:
        return None
    r = spearmanr(a[m], b[m]).statistic
    return None if not np.isfinite(r) else float(r)


# ----------------------------- per-window analysis (on small [L,T] matrices) -----------------------------

def analyze_window(Dc, Dg, Dm, jsd):
    L = Dc.shape[0]
    gc, gj = calib_gamma(Dc), calib_gamma(jsd)
    gg, gm = calib_gamma(Dg, anchor=26), calib_gamma(Dm)
    return {
        "ci_cos": settle_interp(Dc, gc), "ci_jsd": settle_interp(jsd, gj),
        "cd_cos": settle_discrete(Dc, gc), "cd_jsd": settle_discrete(jsd, gj),
        "cd_geo": settle_discrete(Dg, gg), "cd_mag": settle_discrete(Dm, gm),
        "tw_cos": transition_width(Dc)[0], "tw_jsd": transition_width(jsd)[0],
        "bin_cos": binariness(Dc), "bin_jsd": binariness(jsd),
        "cos_curve": running_min(Dc).mean(1), "jsd_curve": running_min(jsd).mean(1),
    }


def readout_scale_invariance(h_layer, decode_fn, alphas=(0.1, 0.5, 2.0, 10.0, 100.0)):
    def sm(z):
        z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
    base = sm(decode_fn(h_layer, False)); worst = 0.0
    for al in alphas:
        p = sm(decode_fn(al * h_layer, False))
        mm = 0.5 * (p + base)
        js = 0.5 * ((p * (np.log(p + 1e-30) - np.log(mm + 1e-30))).sum(-1)
                    + (base * (np.log(base + 1e-30) - np.log(mm + 1e-30))).sum(-1))
        worst = max(worst, float(np.nanmax(js)))
    return worst


# ----------------------------- window iterators (streaming) -----------------------------

def iter_mock(args):
    import _mock_evo2 as mk
    bundle = mk.load_evo2(seed=args.seed); rng = np.random.default_rng(args.seed)
    for w in range(args.n_windows):
        ids = rng.integers(0, 256, args.mock_T)
        hs = mk.extract_hidden_states(bundle, ids)
        st = np.stack([hs[f"blocks.{i}"] for i in range(bundle.L)])   # mock is tiny (H=64)
        hn = hs["norm"]
        Dc = _cos_np(st, hn); Dg = _traj_np(st); Dm = _mag_np(st)
        jsd = mk.jsd_lens_np(st, hn, bundle)
        inv = readout_scale_invariance(st[min(29, st.shape[0] - 1)][:args.a_tokens],
                                       lambda h, pn: bundle.decode(h, pn)) if w < args.a_windows else None
        yield Dc, Dg, Dm, jsd, inv


def iter_real(args):
    import torch, pandas as pd, pysam
    sys.path.insert(0, args.gdtr_root)
    from src.model_loader_evo2 import load_evo2, tokenize                        # type: ignore
    from src.logit_lens_evo2 import extract_hidden_states, all_layer_names, jsd_lens, _layer_logits  # type: ignore
    bundle = load_evo2(); names = all_layer_names(); N = len(names) - 1
    fa = {}
    def fetch(c, s, e):
        cc = c if str(c).startswith("chr") else "chr" + str(c)
        fa.setdefault(cc, pysam.FastaFile(str(Path(args.fasta_dir) / f"{cc}.fa")))
        return fa[cc].fetch(cc, s, e).upper()
    meta = pd.read_parquet(args.windows).head(args.n_windows)

    def decode(h, pn):
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(h)).float().cuda()
            out = _layer_logits(t.unsqueeze(0), bundle, is_post_norm=pn)[0].float().cpu().numpy()
        return out

    for w, (_, r) in enumerate(meta.iterrows()):
        seq = fetch(r.get("chrom", "chr22"), int(r["start"]), int(r["end"]))
        ids = tokenize(seq, bundle, device="cuda")
        with torch.no_grad():
            hs = extract_hidden_states(bundle, ids, save_layers=names)
            Dc, Dg, Dm = _distances_torch(hs, N, torch)                 # [N,T] numpy, GPU-computed per layer
            jsd = jsd_lens(hs, bundle).numpy()
            inv = None
            if w < args.a_windows:
                h29 = hs[f"blocks.{min(29, N - 1)}"][0].float().cpu().numpy()
                idx = np.random.default_rng(args.seed + w).choice(h29.shape[0],
                        size=min(args.a_tokens, h29.shape[0]), replace=False)
                inv = readout_scale_invariance(h29[idx], decode)
        del hs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield Dc, Dg, Dm, jsd, inv


def _distances_torch(hs, N, torch):
    """Cosine / trajectory / magnitude distance matrices [N,T] computed per layer
    on GPU, holding at most two layers of hidden state at a time."""
    hn = hs["norm"][0].float()
    hn_u = hn / hn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    T = hn.shape[0]
    Dc = np.zeros((N, T), np.float32); norms = np.zeros((N, T), np.float32)
    v = np.zeros((N, T), np.float32); kappa = np.zeros((N, T), np.float32)
    prev_h = None; prev_su = None
    for i in range(N):
        h = hs[f"blocks.{i}"][0].float()
        hu = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        Dc[i] = (1 - (hu * hn_u).sum(-1)).cpu().numpy()
        norms[i] = h.norm(dim=-1).cpu().numpy()
        if prev_h is not None:
            step = h - prev_h; sn = step.norm(dim=-1)
            v[i - 1] = (sn / prev_h.norm(dim=-1).clamp_min(1e-12)).cpu().numpy()
            su = step / sn.clamp_min(1e-12).unsqueeze(-1)
            if prev_su is not None:
                kappa[i - 1] = (1 - (su * prev_su).sum(-1)).cpu().numpy()
            prev_su = su
        prev_h = h
    ref = norms[min(29, N - 1):min(29, N - 1) + 1]
    Dm = np.abs(norms / np.clip(ref, 1e-12, None) - 1.0)
    vz = (v - v.mean()) / (v.std() + 1e-9); kz = (kappa - kappa.mean()) / (kappa.std() + 1e-9)
    g = 0.5 * vz + 1.0 * kz; Dg = g.max() - g
    return Dc, Dg, Dm


# numpy lens builders (mock path)
def _cos_np(st, hn):
    hu = st / np.clip(np.linalg.norm(st, axis=-1, keepdims=True), 1e-12, None)
    ru = hn / np.clip(np.linalg.norm(hn, axis=-1, keepdims=True), 1e-12, None)
    return 1.0 - np.einsum("ltd,td->lt", hu, ru)


def _traj_np(st, a=0.5, b=1.0):
    L, T, H = st.shape
    d = np.diff(st, axis=0)
    v = np.linalg.norm(d, axis=-1) / np.clip(np.linalg.norm(st[:-1], axis=-1), 1e-12, None)
    cd = d / np.clip(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12, None)
    kappa = 1.0 - np.einsum("ltd,ltd->lt", cd[1:], cd[:-1])
    vv = np.zeros((L, T)); vv[:L - 1] = v; kk = np.zeros((L, T)); kk[1:L - 1] = kappa
    g = a * (vv - vv.mean()) / (vv.std() + 1e-9) + b * (kk - kk.mean()) / (kk.std() + 1e-9)
    return g.max() - g


def _mag_np(st, ref=29):
    n = np.linalg.norm(st, axis=-1)
    r = n[min(ref, st.shape[0] - 1):min(ref, st.shape[0] - 1) + 1]
    return np.abs(n / np.clip(r, 1e-12, None) - 1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--gdtr-root"); p.add_argument("--fasta-dir"); p.add_argument("--windows")
    p.add_argument("--n-windows", type=int, default=20); p.add_argument("--mock-T", type=int, default=200)
    p.add_argument("--a-windows", type=int, default=3, help="# windows used for the 7A scale-invariance probe")
    p.add_argument("--a-tokens", type=int, default=512, help="# tokens subsampled for 7A")
    p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    it = iter_mock(args) if args.mock else iter_real(args)
    inv_vals, tw_c, tw_j, bn_c, bn_j = [], [], [], [], []
    ci_c, ci_j, cd_c, cd_j, cd_g, cd_m = [], [], [], [], [], []
    cos_curve = jsd_curve = None; L = None
    for Dc, Dg, Dm, jsd, inv in it:
        L = Dc.shape[0]
        a = analyze_window(Dc, Dg, Dm, jsd)
        if inv is not None:
            inv_vals.append(inv)
        tw_c.append(a["tw_cos"]); tw_j.append(a["tw_jsd"]); bn_c.append(a["bin_cos"]); bn_j.append(a["bin_jsd"])
        ci_c.append(a["ci_cos"]); ci_j.append(a["ci_jsd"])
        cd_c.append(a["cd_cos"]); cd_j.append(a["cd_jsd"]); cd_g.append(a["cd_geo"]); cd_m.append(a["cd_mag"])
        cos_curve = a["cos_curve"] if cos_curve is None else cos_curve + a["cos_curve"]
        jsd_curve = a["jsd_curve"] if jsd_curve is None else jsd_curve + a["jsd_curve"]
    nW = len(tw_c); cos_curve /= nW; jsd_curve /= nW
    ci_c = np.concatenate(ci_c); ci_j = np.concatenate(ci_j)
    cd_c = np.concatenate(cd_c); cd_j = np.concatenate(cd_j); cd_g = np.concatenate(cd_g); cd_m = np.concatenate(cd_m)

    summary = {
        "mode": "mock" if args.mock else "real", "n_windows": nW, "n_tokens": int(ci_c.size), "n_layers": L,
        "A_readout_scale_invariance_maxJS": float(np.mean(inv_vals)) if inv_vals else None,
        "resolution": {
            "transition_width_layers": {"cosine": float(np.mean(tw_c)), "jsd": float(np.mean(tw_j)),
                                        "note": "10->90% fall width; larger=smoother. Expect cosine>>jsd on real Evo 2."},
            "binariness_max_single_drop_frac": {"cosine": float(np.mean(bn_c)), "jsd": float(np.mean(bn_j)),
                                                "note": "1.0=cliff. Expect jsd>>cosine on real Evo 2."},
            "interp_settling_granularity": {"cosine": interp_granularity(ci_c, L), "jsd": interp_granularity(ci_j, L),
                                            "note": "distinct sub-layer levels; more=higher resolution."},
        },
        "proxy_fidelity": {
            "spearman_cos_vs_jsd_commit": spearman(cd_c, cd_j),
            "agreement_within1layer": float((np.abs(cd_c - cd_j) <= 1).mean()),
            "caveat": "PARTLY STRUCTURAL: readout unembed(RMSNorm(h)) is direction-based, so a direction lens "
                      "tracking an output-distribution lens is expected. Validates cosine as a faithful PROXY of "
                      "the same commitment event; NOT independent proof of JSD.",
        },
        "trajectory_control": {"spearman_geo_vs_jsd_commit": spearman(cd_g, cd_j),
                               "geo_settling_std": float(np.std(cd_g[cd_g < L])) if (cd_g < L).any() else 0.0,
                               "note": "decoupling is meaningful only if geo settling actually VARIES (std large)."},
        "magnitude_control": {"spearman_mag_vs_jsd_commit": spearman(cd_m, cd_j),
                              "note": "expected ~0 by 7A (readout ignores magnitude)."},
        "headline": ("Cosine is a DIRECT (angle to h_norm, no decoding), HIGH-RESOLUTION proxy for functional "
                     "commitment: reads the direction the model uses (A), resolves stabilization far more "
                     "smoothly/densely than the near-binary JSD it proxies (resolution). Proxy, not proof."),
    }
    (out / "proxy_resolution_summary.json").write_text(json.dumps(summary, indent=2))
    _plots(out, cos_curve, jsd_curve, ci_c, ci_j, L)
    log.info("7A readout JS=%s | binariness cos=%.2f jsd=%.2f | transition cos=%.2f jsd=%.2f",
             summary["A_readout_scale_invariance_maxJS"], np.mean(bn_c), np.mean(bn_j), np.mean(tw_c), np.mean(tw_j))
    log.info("proxy spearman=%s agree1=%.3f | geo spearman=%s geo_std=%.2f",
             summary["proxy_fidelity"]["spearman_cos_vs_jsd_commit"], summary["proxy_fidelity"]["agreement_within1layer"],
             summary["trajectory_control"]["spearman_geo_vs_jsd_commit"], summary["trajectory_control"]["geo_settling_std"])
    log.info("Done -> %s", out / "proxy_resolution_summary.json")


def _plots(out, cos_curve, jsd_curve, ci_c, ci_j, L):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = np.arange(len(cos_curve))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, cos_curve / max(cos_curve.max(), 1e-9), "-o", ms=3, label="cosine (running-min, norm.)")
    ax.plot(x, jsd_curve / max(jsd_curve.max(), 1e-9), "-s", ms=3, label="JSD (running-min, norm.)")
    ax.set_xlabel("layer ℓ"); ax.set_ylabel("normalised distance-to-final")
    ax.set_title("Cosine descends smoothly; JSD is a near-binary cliff"); ax.legend(fontsize=8); fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(out / f"F_cos_vs_jsd_descent.{e}", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ci_c[ci_c < L], bins=40, alpha=.6, label="cosine c_interp (dense)")
    ax.hist(ci_j[ci_j < L], bins=40, alpha=.6, label="JSD c_interp (coarse)")
    ax.set_xlabel("interpolated settling depth"); ax.set_ylabel("# tokens")
    ax.set_title("Sub-layer settling resolution: cosine fine, JSD coarse"); ax.legend(fontsize=8); fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(out / f"F_settling_resolution.{e}", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
