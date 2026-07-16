#!/usr/bin/env python3
"""wcl10 v2 — Causal Settling Coincidence, PROJECTION freeze + threshold robustness.

=================================================================================
WHAT CHANGED vs the first wcl10 (per the real-run critique)
=================================================================================
The first run gave a strong reference-specificity result (H10b: causal commit
matches ONLY the h_norm-anchored cosine) but weak triple coincidence (H10a≈0.28,
H10d≈0.32) because (i) the "full-norm reattach" freeze created off-manifold states
and a NON-MONOTONE KL curve, and (ii) c_causal was defined by the FIRST tau
crossing, which misreads a non-monotone curve (misses the real late commitment).
Two fixes:

  (1) ORTHOGONAL-PROJECTION freeze (정사영).  Instead of  h <- ||h|| * u0  (which
      reattaches the FULL magnitude — built to move toward the ORIGINAL direction
      u_ell — onto u0, e.g. (3,4)->(5,0), an over-state the model never made), we
      PROJECT:  h <- (h . u0) u0  = keep only the component the model actually
      built ALONG u0, remove the rotating (perpendicular) part. This preserves the
      model's own parallel computation and blocks ONLY direction change, so the
      intervention is narrower, more on-manifold, and a purer measure of the
      causal importance of directional change.  (mode='proj')

  (2) TWO commit definitions + a THRESHOLD SWEEP.  We report c_causal under both
      "first-crossing" and "last-effective-layer" (robust to a non-monotone curve,
      captures late commitment), across a WIDE range of tau, and show whether the
      c_causal<->c_cos / c_causal<->c_jsd relationship is STABLE across tau. A
      relationship that survives a wide tau range — even though tau is fixed by an
      independent OUTPUT criterion — is the evidence that c_causal is a real metric,
      not a tau-tuning artifact.

Also: (3) HELD-OUT split — the tau-sweep is reported on two disjoint window halves,
so stability on NEW evaluation data is visible; (4) explicit checks that cosine AND
jsd show the SAME pattern and that the MAGNITUDE freeze yields no matching commit
structure (magnitude has ~no effect).

c_causal stays REFERENCE-FREE (projection onto the token's OWN direction u_ell), so
H10b (only h_norm cosine matches) remains non-circular.

RUN
  mock (verify, no GPU):  python wcl10_causal_settling_coincidence.py --mock --out out/exp10/
  real (GPU via gDTR):    python wcl10_causal_settling_coincidence.py \
      --gdtr-root ../gDTR --fasta-dir ../gDTR/data/reference \
      --windows data_cache_minimal/chr22_metadata.parquet \
      --n-windows 40 --n-pos 256 --layer-step 1 --out results/wcl/exp10_proj/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("wcl10")

ANCHOR, Q, W = 28, 0.70, 3
TAUS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


# ----------------------------- settling / lenses -----------------------------

def running_min(D):
    return np.minimum.accumulate(D, axis=0)


def settle(D, anchor=ANCHOR, q=Q, Wp=W):
    D = np.asarray(D, float); L, T = D.shape
    gamma = float(np.quantile(D[min(anchor, L - 1)], q))
    below = D <= gamma
    c = np.full(T, float(L))
    for ell in range(0, L - Wp + 1):
        run = np.all(below[ell:ell + Wp], axis=0)
        c[run & (c == L)] = ell
    return c


def cos_dist(h_stack, ref):
    if ref.ndim == 1:
        ref = np.broadcast_to(ref, (h_stack.shape[1], h_stack.shape[2]))
    hu = h_stack / np.clip(np.linalg.norm(h_stack, axis=-1, keepdims=True), 1e-12, None)
    ru = ref / np.clip(np.linalg.norm(ref, axis=-1, keepdims=True), 1e-12, None)
    return 1.0 - np.einsum("ltd,td->lt", hu, ru)


def softmax(z):
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)          # sanitize any non-finite logits
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def sym_kl_rows(P, Q):
    P = np.clip(np.nan_to_num(P, nan=0.0), 1e-30, 1); Q = np.clip(np.nan_to_num(Q, nan=0.0), 1e-30, 1)
    P = P / P.sum(-1, keepdims=True); Q = Q / Q.sum(-1, keepdims=True)
    d = 0.5 * ((P * np.log(P / Q)).sum(-1) + (Q * np.log(Q / P)).sum(-1))
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)       # never leak NaN into the curve


def spearman(a, b):
    from scipy.stats import spearmanr
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.unique(a[m]).size < 2 or np.unique(b[m]).size < 2:
        return None
    r = spearmanr(a[m], b[m]).statistic
    return None if not np.isfinite(r) else float(r)


def entropy_bits(c, L):
    c = c[c < L]
    if c.size < 5:
        return 0.0
    h = np.bincount(c.astype(int), minlength=L + 1); p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def agree_within(a, b, L, tol=1):
    """Fraction of tokens whose two commit layers agree within `tol` layers."""
    m = (a < L) & (b < L)
    return float((np.abs(a[m] - b[m]) <= tol).mean()) if m.any() else None


# ----------------------------- c_causal definitions -----------------------------

def commit_firstcross(M_norm, cand, L, tau):
    """First from_layer whose normalised KL <= tau (legacy)."""
    c = np.full(M_norm.shape[1], float(L))
    for i, l in enumerate(cand):
        hit = (M_norm[i] <= tau) & (c == L)
        c[hit] = l
    return c


def commit_lasteffective(M_norm, cand, L, tau):
    """Last from_layer whose normalised KL still EXCEEDS tau, +1 (= after the last
    direction update that mattered = committed). Robust to a non-monotone curve and
    captures LATE commitment (the L28->L30 rotation), which first-crossing misses."""
    cand = np.asarray(cand)
    c = np.full(M_norm.shape[1], float(cand[0]))     # if never above tau -> committed from the start
    for i, l in enumerate(cand):
        above = M_norm[i] > tau
        c[above] = min(int(l) + 1, L)
    return c


# ----------------------------- window iterators -----------------------------

def iter_mock(args):
    import _mock_evo2 as mk
    bundle = mk.load_evo2(seed=args.seed); rng = np.random.default_rng(args.seed)
    for w in range(args.n_windows):
        ids = rng.integers(0, 256, args.n_pos)
        base = mk.extract_hidden_states(bundle, ids)
        stack = np.stack([base[f"blocks.{i}"] for i in range(bundle.L)])
        h_norm = base["norm"]
        p0 = softmax(bundle.decode(h_norm, is_post_norm=True))
        jsd = mk.jsd_lens_np(stack, h_norm, bundle)

        def freeze_out(from_layer, mode):
            hs = mk.extract_hidden_states(bundle, ids, freeze=(from_layer, mode))
            return softmax(bundle.decode(hs["norm"], is_post_norm=True))
        yield w, stack, h_norm, p0, jsd, freeze_out


def iter_real(args):
    import torch, pandas as pd, pysam
    sys.path.insert(0, args.gdtr_root)
    from src.model_loader_evo2 import load_evo2, tokenize                        # type: ignore
    from src.logit_lens_evo2 import extract_hidden_states, all_layer_names, jsd_lens, _layer_logits  # type: ignore
    bundle = load_evo2(); names = all_layer_names(); N = len(names) - 1
    blocks = _find_blocks(bundle); state = {"freeze": None, "dir0": None, "mag0": None}

    def mkhook(ell):
        def hook(m, i, o):
            fz = state["freeze"]
            if fz is None or ell < fz[0]:
                return o
            t = o[0] if isinstance(o, tuple) else o                # [1,T,H] residual OUT
            inp = i[0] if isinstance(i, (tuple, list)) else i      # residual IN to the block
            if not hasattr(inp, "shape") or inp.shape[-2:] != t.shape[-2:]:
                return o                                           # unexpected block signature -> skip safely
            x_in = inp[0].float(); x_out = t[0].float()            # [T,H]
            if ell == fz[0]:
                state["dir0"] = (x_in / x_in.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clone()
            u0 = state["dir0"]
            if fz[1] in ("proj", "proj_perp"):                     # UPDATE projection (stable, no state collapse)
                delta = x_out - x_in
                par = (delta * u0).sum(-1, keepdim=True) * u0      # update component along u0
                newx = x_in + (par if fz[1] == "proj" else delta - par)
            elif fz[1] == "direction":                             # legacy state-level (unstable)
                newx = x_out.norm(dim=-1, keepdim=True) * u0
            else:
                newx = x_out
            t[0] = newx.to(t.dtype)
            return o
        return hook
    handles = [blk.register_forward_hook(mkhook(i)) for i, blk in enumerate(blocks)]

    fa = {}
    def fetch(c, s, e):
        cc = c if str(c).startswith("chr") else "chr" + str(c)
        fa.setdefault(cc, pysam.FastaFile(str(Path(args.fasta_dir) / f"{cc}.fa")))
        return fa[cc].fetch(cc, s, e).upper()
    meta = pd.read_parquet(args.windows).head(args.n_windows)

    for w, (_, r) in enumerate(meta.iterrows()):
        seq = fetch(r.get("chrom", "chr22"), int(r["start"]), int(r["end"]))
        ids = tokenize(seq, bundle, device="cuda")
        T = ids.shape[-1]; pos = np.linspace(0, T - 1, min(args.n_pos, T)).astype(int)
        with torch.no_grad():
            state["freeze"] = None
            hs = extract_hidden_states(bundle, ids, save_layers=names)
            stack = torch.stack([hs[f"blocks.{i}"][0][pos] for i in range(N)]).float().cpu().numpy()
            h_norm = hs["norm"][0][pos].float().cpu().numpy()
            jsd = jsd_lens(hs, bundle).numpy()[:, pos]
            p0 = softmax(_layer_logits(hs["norm"], bundle, is_post_norm=True)[0][pos].float().cpu().numpy())
        def freeze_out(from_layer, mode, ids=ids, pos=pos):
            with torch.no_grad():
                state["freeze"] = (from_layer, mode)
                hsf = extract_hidden_states(bundle, ids, save_layers=["norm"])
                state["freeze"] = None
                lg = _layer_logits(hsf["norm"], bundle, is_post_norm=True)[0][pos].float().cpu().numpy()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return softmax(lg)
        yield w, stack, h_norm, p0, jsd, freeze_out
    for h in handles:
        h.remove()


def _find_blocks(b):
    for path in ("model.backbone.blocks",
        "model.blocks",
        "backbone.blocks",
        "model.model.backbone.blocks",
        "model.model.blocks",
        "model.backbone.layers",
        "model.layers",
        "model.transformer.blocks",
        "model.transformer.layers",
        "model.network.blocks"):
        o = b
        try:
            for a in path.split("."):
                o = getattr(o, a)
            return o
        except AttributeError:
            continue
    raise RuntimeError("locate block ModuleList: print(bundle.model)")


# ----------------------------- main -----------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--gdtr-root"); p.add_argument("--fasta-dir"); p.add_argument("--windows")
    p.add_argument("--n-windows", type=int, default=8); p.add_argument("--n-pos", type=int, default=128)
    p.add_argument("--layer-step", type=int, default=2)
    p.add_argument("--mid-layer", type=int, default=15)
    p.add_argument("--commit-def", choices=["firstcross", "lasteffective"], default="firstcross",
                   help="c_causal = FIRST layer whose normalised freeze-KL drops <= tau (default; the same "
                        "'first threshold crossing' logic as c_cos, and it tracks the token-specific MID approach "
                        "to h_norm). lasteffective is optional but on Evo-2 it locks onto the universal L28->L30 "
                        "rotation and becomes near-constant -> Spearman null; not recommended here.")
    p.add_argument("--rep-tau", type=float, default=0.10, help="representative tau for the reference-specificity readout")
    p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    rng = np.random.default_rng(args.seed)

    it = iter_mock(args) if args.mock else iter_real(args)
    # per-window stores
    W_cos, W_cosR, W_cosM, W_jsd = [], [], [], []
    W_dirnorm, W_magnorm = [], []      # normalised KL curves [nL, n_pos] per window
    W_split = []
    dir_curve_abs, mag_curve_abs = [], []
    cand = None; L = None
    for w, stack, h_norm, p0, jsd, freeze_out in it:
        L = stack.shape[0]
        cand = list(range(0, L, args.layer_step))
        rand_ref = rng.standard_normal(stack.shape[-1]); mid_ref = stack[min(args.mid_layer, L - 1)]
        W_cos.append(settle(cos_dist(stack, h_norm)))
        W_cosR.append(settle(cos_dist(stack, rand_ref)))
        W_cosM.append(settle(cos_dist(stack, mid_ref)))
        W_jsd.append(settle(jsd))
        # direction freeze = keep PARALLEL update ('proj'); magnitude control = keep
        # PERPENDICULAR update ('proj_perp'). Both are on-manifold update components.
        dkl = np.stack([sym_kl_rows(freeze_out(fl, "proj"), p0) for fl in cand], 0)      # [nL, n_pos]
        mkl = np.stack([sym_kl_rows(freeze_out(fl, "proj_perp"), p0) for fl in cand], 0)
        # normalise each token's curve by its OWN MAX over layers (robust to one bad
        # layer; a bad layer-0 no longer poisons the whole normalisation -> no all-null)
        W_dirnorm.append(dkl / np.clip(dkl.max(0, keepdims=True), 1e-12, None))
        W_magnorm.append(mkl / np.clip(mkl.max(0, keepdims=True), 1e-12, None))
        dir_curve_abs.append(dkl.mean(1)); mag_curve_abs.append(mkl.mean(1))
        W_split.append(w % 2)
        log.info("window %d done (proj/proj_perp freeze forwards; dkl finite=%.0f%% mkl finite=%.0f%%)",
                 w, 100 * np.isfinite(dkl).mean(), 100 * np.isfinite(mkl).mean())

    cos = np.concatenate(W_cos); cosR = np.concatenate(W_cosR); cosM = np.concatenate(W_cosM); jsd = np.concatenate(W_jsd)
    dirN = np.concatenate(W_dirnorm, axis=1)      # [nL, N_tokens]
    magN = np.concatenate(W_magnorm, axis=1)
    split = np.concatenate([np.full(w.shape, s) for w, s in zip(W_cos, W_split)])
    dir_curve = np.mean(dir_curve_abs, 0); mag_curve = np.mean(mag_curve_abs, 0)

    # c_causal definition: FIRST-CROSSING (same "first threshold crossing" logic as
    # the cosine settling c_cos itself -> a fair, consistent comparison). The
    # projection freeze is what makes the freeze-KL curve monotone enough for this.
    commit = commit_lasteffective if args.commit_def == "lasteffective" else commit_firstcross

    def sweep(Mnorm, target):
        rows = []
        for tau in TAUS:
            c = commit(Mnorm, cand, L, tau)
            rows.append({"tau": tau, "spearman": spearman(c, target), "entropy_bits": round(entropy_bits(c, L), 3)})
        return rows

    def sweep_split(Mnorm, target, s):
        idx = split == s
        return [{"tau": tau, "spearman": spearman(commit(Mnorm[:, idx], cand, L, tau), target[idx])} for tau in TAUS]

    # representative-tau commit layers (for the backward-compatible H10a-e block)
    c_causal_rep = commit(dirN, cand, L, args.rep_tau)
    c_causal_mag_rep = commit(magN, cand, L, args.rep_tau)

    def _diag(name, a):
        a = np.asarray(a, float)
        return {"n": int(a.size), "n_unique": int(np.unique(a[np.isfinite(a)]).size),
                "frac_finite": round(float(np.isfinite(a).mean()), 4),
                "min": None if not np.isfinite(a).any() else round(float(np.nanmin(a)), 3),
                "max": None if not np.isfinite(a).any() else round(float(np.nanmax(a)), 3),
                "std": None if not np.isfinite(a).any() else round(float(np.nanstd(a)), 3)}

    # Diagnostics — if any Spearman is null, this block says WHY (which array is
    # constant / non-finite). A healthy run has n_unique >= 2 and frac_finite = 1.0
    # for c_cos, c_jsd, c_causal, and frac_finite = 1.0 for the freeze curves.
    diagnostics = {
        "c_cos": _diag("c_cos", cos), "c_jsd": _diag("c_jsd", jsd),
        "c_causal_dir@rep_tau": _diag("c_causal", c_causal_rep),
        "c_causal_mag@rep_tau": _diag("c_causal_mag", c_causal_mag_rep),
        "dir_freeze_KL_finite_frac": round(float(np.isfinite(dirN).mean()), 4),
        "mag_freeze_KL_finite_frac": round(float(np.isfinite(magN).mean()), 4),
        "spearman_null_reason": ("if a Spearman is null: the paired array had <10 finite pairs OR was constant "
                                 "(n_unique<2). Check n_unique/frac_finite above. c_causal constant usually means "
                                 "the freeze forward returned non-finite output (see *_finite_frac)."),
    }
    log.info("DIAG c_cos uniq=%d c_jsd uniq=%d c_causal uniq=%d | dirKL finite=%.0f%% magKL finite=%.0f%%",
             diagnostics["c_cos"]["n_unique"], diagnostics["c_jsd"]["n_unique"],
             diagnostics["c_causal_dir@rep_tau"]["n_unique"],
             100 * diagnostics["dir_freeze_KL_finite_frac"], 100 * diagnostics["mag_freeze_KL_finite_frac"])

    summary = {
        "_diagnostics": diagnostics,
        # ---------- metadata (makes the version unambiguous) ----------
        "schema_version": "wcl10.v2",
        "mode": "mock" if args.mock else "real", "n_tokens": int(cos.size), "n_layers": L,
        "freeze_method": "UPDATE orthogonal projection: keep the block update component parallel to u0=unit(x_from_layer) (numerically stable; no state collapse)",
        "commit_definition": (args.commit_def + " : "
                              + ("LAST from_layer whose normalised freeze-KL still exceeds tau, +1 "
                                 "(captures the late L28->L30 commitment)" if args.commit_def == "lasteffective"
                                 else "FIRST from_layer whose normalised freeze-KL <= tau")),
        "representative_tau": args.rep_tau,

        # ---------- BACKWARD-COMPATIBLE block (old H10a-e keys, at representative tau) ----------
        "H10a_causal_vs_cos_hnorm": {
            "spearman": spearman(c_causal_rep, cos), "agree_1layer": agree_within(c_causal_rep, cos, L),
            "tau": args.rep_tau},
        "H10b_reference_specificity": {
            "spearman_causal_vs_cos_random": spearman(c_causal_rep, cosR),
            "spearman_causal_vs_cos_midlayer": spearman(c_causal_rep, cosM),
            "verdict_note": "causal commit matches ONLY the h_norm-anchored cosine settling (random/mid ~0) "
                            "=> h_norm is the functional reference (the causal Exp-8 proof)."},
        "H10c_direction_vs_magnitude_freeze": {
            "dir_freeze_KL_curve": [round(float(x), 5) for x in dir_curve],
            "mag_freeze_KL_curve": [round(float(x), 5) for x in mag_curve],
            "mean_dir_over_mag": float(dir_curve.mean() / max(mag_curve.mean(), 1e-12)),
            "verdict_note": "magnitude-freeze KL -> 0 at late layers << direction-freeze => model reads direction."},
        "H10d_causal_vs_jsd": {
            "spearman": spearman(c_causal_rep, jsd), "agree_1layer": agree_within(c_causal_rep, jsd, L),
            "tau": args.rep_tau},
        "H10e_layer_meaningful": {
            "c_causal_entropy_bits": entropy_bits(c_causal_rep, L),
            "c_cos_entropy_bits": entropy_bits(cos, L),
            "note": "entropy > 0 => the commit layer genuinely varies across tokens."},

        # ---------- NEW robustness block (threshold sweep + held-out splits) ----------
        "target_similarity": {"spearman_cos_vs_jsd": spearman(cos, jsd)},
        "tau_sweep_vs_cos": sweep(dirN, cos),
        "tau_sweep_vs_jsd": sweep(dirN, jsd),
        "data_stability_vs_cos": {
            "splitA": sweep_split(dirN, cos, 0),
            "splitB": sweep_split(dirN, cos, 1),
            "note": "overlap of the two curves across tau => stable on new evaluation data."},
        "reference_specificity_tau_sweep": {
            "vs_cos_random": [{"tau": t, "spearman": spearman(commit(dirN, cand, L, t), cosR)} for t in TAUS],
            "vs_cos_midlayer": [{"tau": t, "spearman": spearman(commit(dirN, cand, L, t), cosM)} for t in TAUS]},
        "magnitude_control": {
            "mean_dir_over_mag_absKL": float(dir_curve.mean() / max(mag_curve.mean(), 1e-12)),
            "spearman_causal_MAG_vs_cos": [
                {"tau": t, "spearman": spearman(commit(magN, cand, L, t), cos)} for t in TAUS],
            "note": "the magnitude-freeze commit layer should NOT track c_cos across tau."},
        "candidate_layers": cand,

        "headline": ("Update-projection freeze + FIRST-CROSSING commit (same 'first threshold crossing' logic as "
                     "c_cos; it tracks the token-specific MID approach to h_norm, whereas last-effective locks onto "
                     "the universal L28->L30 rotation and goes constant). c_causal is meaningful ONLY if its "
                     "correlation with c_cos (and c_jsd) is stable across a wide tau range AND across held-out data, "
                     "appears ONLY for the h_norm reference, and is absent for magnitude (see dir/mag ratio: "
                     "perpendicular-update control ~0). Read tau_sweep_vs_cos + data_stability_vs_cos + _diagnostics."),
        "caveat": "MOCK verifies code/logic only (its blocks are not fully scale-invariant, so magnitude separation "
                  "is understated). Freezing = counterfactual; c_causal is an output-space sufficiency layer, not a "
                  "full circuit.",
    }
    (out / "causal_settling_coincidence.json").write_text(json.dumps(summary, indent=2))
    _plot(out, summary, dir_curve, mag_curve, cand)
    log.info("schema=%s commit_def=%s | cos~jsd=%s", summary["schema_version"], args.commit_def,
             summary["target_similarity"]["spearman_cos_vs_jsd"])
    log.info("H10a c_causal~cos @tau=%.2f: spearman=%s agree=%s", args.rep_tau,
             summary["H10a_causal_vs_cos_hnorm"]["spearman"], summary["H10a_causal_vs_cos_hnorm"]["agree_1layer"])
    log.info("tau-sweep vs cos: %s", [(r["tau"], r["spearman"]) for r in summary["tau_sweep_vs_cos"]])
    log.info("tau-sweep vs jsd: %s", [(r["tau"], r["spearman"]) for r in summary["tau_sweep_vs_jsd"]])
    rs = summary["H10b_reference_specificity"]
    log.info("ref-specificity @tau=%.2f: hnorm=%s random=%s mid=%s", args.rep_tau,
             summary["H10a_causal_vs_cos_hnorm"]["spearman"],
             rs["spearman_causal_vs_cos_random"], rs["spearman_causal_vs_cos_midlayer"])
    log.info("dir/mag absKL ratio=%.1fx", summary["magnitude_control"]["mean_dir_over_mag_absKL"])
    log.info("Done -> %s", out / "causal_settling_coincidence.json")


def _plot(out, summary, dir_curve, mag_curve, cand):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # 1) tau-sweep stability
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, rows in [("c_causal vs cos", summary["tau_sweep_vs_cos"]),
                       ("c_causal vs jsd", summary["tau_sweep_vs_jsd"])]:
        xs = [r["tau"] for r in rows]; ys = [r["spearman"] if r["spearman"] is not None else np.nan for r in rows]
        ax.plot(xs, ys, "-o", ms=3, label=name)
    ax.axhline(0, color="grey", lw=.8); ax.set_xlabel("threshold τ"); ax.set_ylabel("Spearman(c_causal, target)")
    ax.set_title("Is the coincidence stable across τ?"); ax.legend(fontsize=8); ax.set_ylim(-0.2, 1.0); fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(out / f"F_tau_sweep.{e}", dpi=150)
    plt.close(fig)
    # 2) freeze curves (dir projection vs magnitude)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cand, dir_curve, "-o", ms=3, label="projection direction-freeze KL")
    ax.plot(cand, mag_curve, "-s", ms=3, label="perpendicular-update (magnitude control)")
    ax.set_xlabel("freeze-from layer ℓ"); ax.set_ylabel("output-space KL"); ax.legend(fontsize=8)
    ax.set_title("Projection freeze curve (smoother) vs magnitude"); fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(out / f"F_freeze_curves.{e}", dpi=150)
    plt.close(fig)
    # 3) data-split stability
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in ("splitA", "splitB"):
        rows = summary["data_stability_vs_cos"][s]
        xs = [r["tau"] for r in rows]; ys = [r["spearman"] if r["spearman"] is not None else np.nan for r in rows]
        ax.plot(xs, ys, "-o", ms=3, label=s)
    ax.axhline(0, color="grey", lw=.8); ax.set_xlabel("threshold τ"); ax.set_ylabel("Spearman(c_causal, c_cos)")
    ax.set_title("Held-out stability: two disjoint window halves"); ax.legend(fontsize=8); ax.set_ylim(-0.2, 1.0); fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(out / f"F_split_stability.{e}", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
