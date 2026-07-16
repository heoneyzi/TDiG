#!/usr/bin/env python3
"""wcl04 v2 — Direction vs Magnitude causal patching in OUTPUT (distribution) space,
at EQUAL intervention budget, above an on-manifold NULL floor.

=================================================================================
WHAT CHANGED vs the version you ran (and WHY it is now valid)
=================================================================================
Your run measured `downstream_shift_norm` = an L2 shift of the HIDDEN STATE, with
modes full/direction/magnitude and no budget control. Diagnostic on your CSV:
`direction ≈ full` (dir/full = 1.000, spearman 0.998) and magnitude ≈ 0.03*full.
That result is confounded and near-trivial for two reasons:
  * hidden-state L2 measures "how much the representation moved," NOT "how much the
    PREDICTION changed" — but GDTR's question is about the model's output;
  * a single-nucleotide variant's difference is ~97% directional, so the direction
    patch is a MUCH bigger perturbation than the magnitude patch — a bigger shift
    downstream is then trivial ("direction is the bigger patch"), not evidence the
    model prefers direction.

Three fixes make it a valid causal test:
  (1) OUTPUT-SPACE METRIC. effect = symmetric KL of the next-token distribution
      (patched vs unpatched), read through the model's own readout. Because the
      readout is RMSNorm-invariant, this metric is not pre-biased toward either
      intervention.
  (2) EQUAL BUDGET. Instead of the ref/alt-defined patches (directional by
      construction), apply CONTROLLED moves of the SAME size B = ||h_ref - h_alt||:
        DIRECTION : move h_alt by B toward the ref DIRECTION (angular move)
        MAGNITUDE : move h_alt by B along its OWN direction (radial move)
        NULL      : move h_alt by B toward a real other-position state (on-manifold)
      Now any difference in output KL is due to WHERE the move points, not its size.
  (3) NULL FLOOR + multi-depth. Read direction/magnitude ABOVE the null, at several
      layers, so magnitude has downstream nonlinear blocks to act on (non-trivial).

Valid interpretation: if, dollar-for-dollar, an angular move changes the model's
OUTPUT far more than a radial move (and above the null floor), the model's
computation is genuinely more sensitive to direction than magnitude — the causal
support a direction-only lens needs. (Still a sensitivity claim, not a full circuit.)

Requires fasta_utils.py (robust reference loader) in the same folder.

RUN
  mock (verify, no GPU):  python wcl04_v2_output_space.py --mock --out out/exp4_v2/
  real (GPU via gDTR):    python wcl04_v2_output_space.py \
      --gdtr-root ../gDTR --variants data_cache_minimal/variant_scalars.parquet \
      --fasta-dir ../gDTR/data/reference --out results/wcl/exp4_v2/ \
      --layers 12 20 24 27 --n-per-class 30
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("wcl04v2")


def softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True); e = np.exp(z); return e / e.sum(axis, keepdims=True)


def sym_kl(p, q):
    p = np.clip(p, 1e-30, 1); q = np.clip(q, 1e-30, 1)
    return float(0.5 * ((p * np.log(p / q)).sum(-1) + (q * np.log(q / p)).sum(-1)).mean())


def equal_budget_patch(h_alt_v, h_ref_v, mode, B, null_pool=None, rng=None):
    """Return a patched vector = h_alt_v + (a move of size ~B), where the move
    DIRECTION depends on `mode`. All modes have ||patched - h_alt_v|| == B, so
    they are matched in intervention size and differ only in where they point."""
    na = np.linalg.norm(h_alt_v) + 1e-12
    if mode == "direction":
        target = na * h_ref_v / (np.linalg.norm(h_ref_v) + 1e-12)     # ref direction, alt norm
        d = target - h_alt_v
    elif mode == "magnitude":
        sign = 1.0 if np.linalg.norm(h_ref_v) > na else -1.0
        d = sign * h_alt_v / na                                        # pure radial (own direction)
    elif mode in {"null", "on_manifold_null"}:
        other = null_pool[rng.integers(0, null_pool.shape[0])]
        d = other - h_alt_v                                            # toward a real other state
    elif mode == "full":
        return h_ref_v                                                 # reference (size B by def.)
    else:
        raise ValueError(mode)
    nd = np.linalg.norm(d) + 1e-12
    return h_alt_v + B * d / nd                                        # exact size B


def measure_variant(forward_dist, h_ref_stack, h_alt_stack, positions, layers, modes, rng):
    """forward_dist(patch)-> dist array [len(positions), V] (patch=(layer,pos,vec) or None).
    positions[0] is the variant position; the rest are downstream held-out positions."""
    vi = positions[0]
    base = forward_dist(None)
    recs = []
    for ell in layers:
        hr = h_ref_stack[ell, vi]; ha = h_alt_stack[ell, vi]
        B = float(np.linalg.norm(hr - ha))                            # equal budget = full-patch size
        if B < 1e-9:
            continue
        natural = float(np.median(np.linalg.norm(h_alt_stack[ell], axis=-1)))
        for mode in modes:
            vec = equal_budget_patch(ha, hr, mode, B, null_pool=h_alt_stack[ell], rng=rng)
            eff = sym_kl(forward_dist((ell, vi, vec)), base)          # OUTPUT-space effect
            recs.append({"layer": int(ell), "mode": mode, "budget_B": B,
                         "effect_symKL": eff,
                         "offmanifold_norm_ratio": float(np.linalg.norm(vec) / (natural + 1e-12))})
    return recs


# ----------------------------- data acquisition -----------------------------

def run_mock(args, modes, rng):
    import _mock_evo2 as mk
    bundle = mk.load_evo2(seed=args.seed); r = np.random.default_rng(args.seed)
    recs = []
    for gi in range(2 * args.n_per_class):
        T = args.mock_T
        ids_ref = r.integers(0, 256, T); vi = r.integers(T // 4, 3 * T // 4)
        ids_alt = ids_ref.copy(); ids_alt[vi] = (ids_alt[vi] + r.integers(1, 255)) % 256
        cat = "P_LP" if gi < args.n_per_class else "B_LB"
        hr = mk.extract_hidden_states(bundle, ids_ref)
        ha = mk.extract_hidden_states(bundle, ids_alt)
        h_ref_stack = np.stack([hr[f"blocks.{i}"] for i in range(bundle.L)])
        h_alt_stack = np.stack([ha[f"blocks.{i}"] for i in range(bundle.L)])
        pos = [vi] + [min(vi + k, T - 1) for k in (1, 2, 3)]
        def fdist(patch):
            hs = mk.extract_hidden_states(bundle, ids_alt, patch=patch)
            return softmax(bundle.decode(hs["norm"][pos], is_post_norm=True), -1)
        for rec in measure_variant(fdist, h_ref_stack, h_alt_stack, pos, args.layers, modes, rng):
            rec.update({"variant": gi, "category": cat}); recs.append(rec)
    return recs


def _inline_fasta_loader():
    """Self-contained robust reference loader (no fasta_utils dependency): handles
    chrom '2' vs contig 'chr2', a single whole-genome FASTA OR a per-chromosome
    directory, and skips a missing contig instead of crashing."""
    import os
    import pysam

    def cands(chrom):
        c = str(chrom); bare = c[3:] if c.lower().startswith("chr") else c
        return [f"chr{bare}", bare, c, f"chr{c}"]

    def get_fasta_loader(path):
        path = str(path)
        if os.path.isfile(path):
            fa = pysam.FastaFile(path); refs = set(fa.references)
            def fetch(chrom, s, e):
                for cc in cands(chrom):
                    if cc in refs:
                        return fa.fetch(cc, s, e).upper()
                return None
            return fetch
        cache = {}
        def fetch(chrom, s, e):
            for name in cands(chrom):
                p = os.path.join(path, f"{name}.fa")
                if os.path.exists(p):
                    if p not in cache:
                        cache[p] = pysam.FastaFile(p)
                    fh = cache[p]
                    return fh.fetch(fh.references[0], s, e).upper()
            return None
        return fetch

    def build_seq(fetch, chrom, pos, ref, alt, ctx_kb=3):
        start = max(0, pos - 1 - ctx_kb * 1000)
        seq = fetch(chrom, start, pos - 1 + len(ref) + ctx_kb * 1000)
        if seq is None:
            return None, None, None
        vi = pos - 1 - start
        if seq[vi:vi + len(ref)] != ref:
            return None, None, None
        return seq, seq[:vi] + alt + seq[vi + len(ref):], vi

    return get_fasta_loader, build_seq


def run_real(args, modes, rng):
    import torch, pandas as pd
    sys.path.insert(0, args.gdtr_root)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.model_loader_evo2 import load_evo2, tokenize                       # type: ignore
    from src.logit_lens_evo2 import extract_hidden_states, all_layer_names, _layer_logits  # type: ignore
    get_fasta_loader, build_seq = _inline_fasta_loader()                         # self-contained

    bundle = load_evo2(); names = all_layer_names(); N = len(names) - 1
    def find_blocks(b):
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
    blocks = find_blocks(bundle); state = {"patch": None}
    def mkhook(ell):
        def hook(m, i, o):
            if state["patch"] and state["patch"][0] == ell:
                _, pos, vec = state["patch"]
                t = o[0] if isinstance(o, tuple) else o
                t[0, pos, :] = torch.from_numpy(np.asarray(vec)).to(t.dtype).to(t.device)
                return o
        return hook
    handles = [blk.register_forward_hook(mkhook(i)) for i, blk in enumerate(blocks)]

    fetch = get_fasta_loader(args.fasta_dir)
    df = pd.read_parquet(args.variants); dfb = df[df.category.isin({"P_LP", "B_LB"})]
    samp = pd.concat([dfb[dfb.category == "P_LP"].sample(args.n_per_class, random_state=args.seed),
                      dfb[dfb.category == "B_LB"].sample(args.n_per_class, random_state=args.seed)])
    recs = []
    for gi, (_, row) in enumerate(samp.iterrows()):
        if len(row.ref) != 1 or len(row.alt) != 1:
            continue
        seq_ref, seq_alt, vi = build_seq(fetch, row.chrom, int(row.pos), row.ref.upper(), row.alt.upper())
        if seq_ref is None:
            continue
        ids_ref = tokenize(seq_ref, bundle, device="cuda"); ids_alt = tokenize(seq_alt, bundle, device="cuda")
        T = ids_alt.shape[-1]; pos = [vi] + [min(vi + k, T - 1) for k in (1, 2, 3)]
        with torch.no_grad():
            hr = extract_hidden_states(bundle, ids_ref, save_layers=names)
            ha = extract_hidden_states(bundle, ids_alt, save_layers=names)
            h_ref_stack = torch.stack([hr[f"blocks.{i}"][0] for i in range(N)]).float().cpu().numpy()
            h_alt_stack = torch.stack([ha[f"blocks.{i}"][0] for i in range(N)]).float().cpu().numpy()
            def fdist(patch):
                state["patch"] = patch
                hs = extract_hidden_states(bundle, ids_alt, save_layers=["norm"])
                state["patch"] = None
                lg = _layer_logits(hs["norm"], bundle, is_post_norm=True)[0, pos].float().cpu().numpy()
                return softmax(lg, -1)
            for rec in measure_variant(fdist, h_ref_stack, h_alt_stack, pos, args.layers, modes, rng):
                rec.update({"variant": gi, "category": row.category}); recs.append(rec)
    for h in handles:
        h.remove()
    return recs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--gdtr-root"); p.add_argument("--variants"); p.add_argument("--fasta-dir")
    p.add_argument("--layers", type=int, nargs="+", default=[12, 20, 24, 27])
    p.add_argument("--n-per-class", type=int, default=20); p.add_argument("--mock-T", type=int, default=64)
    p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    modes = ["direction", "magnitude", "on_manifold_null", "full"]

    recs = run_mock(args, modes, rng) if args.mock else run_real(args, modes, rng)
    import pandas as pd
    from scipy.stats import wilcoxon
    df = pd.DataFrame(recs)
    # Raw per-variant results. Use a non-NA-like label for the null control so that
    # pandas/Excel do not silently interpret the string "null" as a missing value.
    df.to_csv(out / "patching_results.csv", index=False)

    # Save the same layer/mode medians printed to the terminal as a separate CSV.
    summary_df = (
        df.groupby(["layer", "mode"], as_index=False)
          .agg(
              n=("effect_symKL", "count"),
              median_effect_symKL=("effect_symKL", "median"),
              mean_effect_symKL=("effect_symKL", "mean"),
              std_effect_symKL=("effect_symKL", "std"),
              median_budget_B=("budget_B", "median"),
              median_offmanifold_norm_ratio=("offmanifold_norm_ratio", "median"),
          )
    )
    summary_df.to_csv(out / "patching_results_summary.csv", index=False)

    verdict = {}
    for ell in args.layers:
        piv = df[df.layer == ell].pivot_table(index="variant", columns="mode", values="effect_symKL").dropna()
        if len(piv) < 6:
            continue
        row = {"n": int(len(piv)),
               "median_direction": float(piv["direction"].median()),
               "median_magnitude": float(piv["magnitude"].median()),
               "median_null": float(piv["on_manifold_null"].median()),
               "direction_over_magnitude": float(piv["direction"].median() / max(piv["magnitude"].median(), 1e-12)),
               "direction_above_null": bool(piv["direction"].median() > piv["on_manifold_null"].median()),
               "magnitude_at_or_below_null": bool(piv["magnitude"].median() <= piv["on_manifold_null"].median())}
        try:
            row["wilcoxon_dir_vs_mag_p"] = float(wilcoxon(piv["direction"], piv["magnitude"]).pvalue)
        except Exception:
            row["wilcoxon_dir_vs_mag_p"] = None
        verdict[f"L{ell}"] = row
        log.info("L%-2d dir=%.4f mag=%.4f null=%.4f | dir/mag=%.1fx dir>null=%s p=%s",
                 ell, row["median_direction"], row["median_magnitude"], row["median_null"],
                 row["direction_over_magnitude"], row["direction_above_null"], row.get("wilcoxon_dir_vs_mag_p"))

    summary = {"mode": "mock" if args.mock else "real", "layers": args.layers,
               "metric": "symmetric KL of next-token distribution (OUTPUT space)",
               "interventions": "EQUAL budget B=||h_ref-h_alt||: angular vs radial vs on-manifold null",
               "per_layer": verdict,
               "read_as": "direction >> magnitude AND direction > null, at equal budget, in output space, "
                          "means the model's computation is causally more sensitive to angular than radial "
                          "moves of the residual stream. This is the valid version of Exp 4.",
               "caveat": "MOCK verifies code only. Patching = counterfactual activations; claim is output-space "
                         "sensitivity above an on-manifold null, not a full circuit."}
    (out / "causal_effect_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Done -> %s", out / "causal_effect_summary.json")
    log.info("Raw CSV -> %s", out / "patching_results.csv")
    log.info("Summary CSV -> %s", out / "patching_results_summary.csv")


if __name__ == "__main__":
    main()
