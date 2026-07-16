#!/usr/bin/env python3
"""wcl04 — Experiment 4 (STRETCH, GPU): direction- vs magnitude-isolated causal
activation patching. Upgrades Thesis A from correlational to causal.

This is the piece TDiG's scripts/32_activation_patching.py explicitly deferred
("For TRUE causal patching, would need model hooks — deferred"; its
`patched_forward_simple` returns None). It requires WRITE hooks on Evo 2's
StripedHyena2 blocks, implemented as a NON-INVASIVE wrapper around gDTR's
inference path (src/model_loader_evo2.py, src/logit_lens_evo2.py) — never edit
the frozen loader in place.

Design (pre-registered):
  For matched ref/alt ClinVar pairs, at the canonical tap L*=29 and flanking
  attention taps (24, 27), run three interventions and propagate each forward:
    FULL           h_alt[ell] <- h_ref[ell]
    DIRECTION-ONLY h_alt[ell] <- ‖h_alt[ell]‖ * unit(h_ref[ell])   (ref direction, alt magnitude)
    MAGNITUDE-ONLY h_alt[ell] <- ‖h_ref[ell]‖ * unit(h_alt[ell])   (alt direction, ref magnitude)
  Measure downstream effect = shift in next-token log-likelihood at held-out
  positions (and final-layer ΔD_cos). Paired Wilcoxon: direction-effect vs
  magnitude-effect, per layer, normalised by ‖h_alt[ell]-h_ref[ell]‖ so both
  interventions move the representation by a comparable budget.

H4  at L*=29 the direction-only patch produces a larger downstream shift than
    the magnitude-only patch (model computation depends more on direction).

REQUIRES: gDTR repo on sys.path, Evo 2 7B weights, 1x H200 (~0.3-3 GPU-h).
This file is the scaffold + the hook implementation stub; fill `HookedEvo2`
against your installed evo2/vortex version (block module paths differ by build).

Run:
    python wcl04_direction_magnitude_patch.py \
        --gdtr-root /path/to/gDTR \
        --variants data_cache_minimal/variant_scalars.parquet \
        --fasta-dir /path/to/reference --out results/wcl/exp4/ \
        --layers 24 27 29 --n-per-class 30
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("wcl04")


class HookedEvo2:
    """Non-invasive wrapper adding forward WRITE hooks at chosen block outputs.

    Implementation note: locate the StripedHyena2 block ModuleList on the loaded
    bundle (commonly `bundle.model.backbone.blocks` or similar — inspect once
    with `print(bundle.model)`), then register_forward_hook that overwrites the
    residual-stream output tensor at (layer, position) with a supplied vector.
    The Hyena long-conv path may not accept mid-sequence overwrite as cleanly as
    attention blocks; per §9.5 failure-modes, scope to attention taps (24, 27)
    first, then attempt L29 (hcm)."""

    def __init__(self, bundle):
        self.bundle = bundle
        self.blocks = self._find_blocks(bundle)
        self._patch = {}   # {layer: (pos, vector)}
        self._handles = []

    @staticmethod
    def _find_blocks(bundle):
        import torch.nn as nn

    # 먼저 흔한 경로들을 직접 확인
        candidate_paths = (
        "model.backbone.blocks",
        "model.blocks",
        "backbone.blocks",
        "model.model.backbone.blocks",
        "model.model.blocks",
        "model.backbone.layers",
        "model.layers",
        "model.transformer.blocks",
        "model.transformer.layers",
        "model.network.blocks",
        )

        for path in candidate_paths:
            obj = bundle
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)

                if isinstance(obj, (nn.ModuleList, nn.Sequential)):
                    print(f"[INFO] Found blocks at: {path}")
                    print(f"[INFO] Number of blocks: {len(obj)}")
                    return obj

            except AttributeError:
                continue

    # 직접 경로에서 못 찾으면 모든 named_modules 검색
        roots = []

        if isinstance(bundle, nn.Module):
            roots.append(("bundle", bundle))

        if hasattr(bundle, "model") and isinstance(bundle.model, nn.Module):
            roots.append(("bundle.model", bundle.model))

        for root_name, root in roots:
            candidates = []

            for name, module in root.named_modules():
                if isinstance(module, (nn.ModuleList, nn.Sequential)):
                    try:
                        length = len(module)
                    except TypeError:
                        continue

                    if length >= 20:
                        candidates.append((name, module))

            if candidates:
                print("[INFO] Candidate block containers:")

                for name, module in candidates:
                    print(
                        f"  {root_name}.{name}: "
                        f"{type(module).__name__}, len={len(module)}"
                    )

                # Evo2 7B는 현재 로그상 blocks.0 ~ blocks.31이 있으므로
                # 길이가 32인 container를 우선 선택
                for name, module in candidates:
                    if len(module) == 32:
                        print(
                            f"[INFO] Selected blocks: "
                            f"{root_name}.{name}"
                        )
                        return module

                # 32개짜리가 없으면 가장 긴 container 선택
                name, module = max(candidates, key=lambda x: len(x[1]))

                print(
                    f"[WARNING] No 32-layer ModuleList found. "
                    f"Using longest candidate: "
                    f"{root_name}.{name}, len={len(module)}"
                )
                return module

        raise RuntimeError(
            "Could not locate Evo2 block container. "
            "Inspect bundle and bundle.model with print()."
        )
    
    def set_patch(self, layer, pos, vector):
        self._patch = {layer: (pos, vector)}

    def clear(self):
        self._patch = {}

    def _make_hook(self, layer):
        def hook(module, inp, out):
            if layer in self._patch:
                pos, vec = self._patch[layer]
                # out may be a tuple; residual stream is typically out[0]
                t = out[0] if isinstance(out, tuple) else out
                t[0, pos, :] = vec.to(t.dtype).to(t.device)
                return out
        return hook

    def __enter__(self):
        for ell, blk in enumerate(self.blocks):
            self._handles.append(blk.register_forward_hook(self._make_hook(ell)))
        return self

    def __exit__(self, *a):
        for h in self._handles:
            h.remove()
        self._handles = []


def isolate(h_ref_v, h_alt_v, mode):
    import torch
    nr = h_ref_v.norm(); na = h_alt_v.norm()
    if mode == "full":
        return h_ref_v
    if mode == "direction":                     # ref direction, alt magnitude
        return na * h_ref_v / nr.clamp_min(1e-12)
    if mode == "magnitude":                     # alt direction, ref magnitude
        return nr * h_alt_v / na.clamp_min(1e-12)
    raise ValueError(mode)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gdtr-root", required=True)
    p.add_argument("--variants", required=True)
    p.add_argument("--fasta-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[24, 27, 29])
    p.add_argument("--n-per-class", type=int, default=30)
    p.add_argument("--held-out", type=int, default=5, help="downstream positions for LL shift")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, args.gdtr_root)

    import pandas as pd
    import torch
    from src.model_loader_evo2 import load_evo2, tokenize          # type: ignore
    from src.logit_lens_evo2 import extract_hidden_states, all_layer_names  # type: ignore

    # reuse 32_activation_patching's fasta + build_seq helpers (import or copy)
    sys.path.insert(0, str(Path(args.gdtr_root).parent / "TDiG" / "scripts"))
    try:
        from importlib import import_module
        ap = import_module("32_activation_patching")
        get_fasta_loader, build_seq = ap.get_fasta_loader, ap.build_seq
    except Exception:
        log.warning("could not import 32_activation_patching helpers — using inline fallback")
        raise

    df = pd.read_parquet(args.variants)
    dfb = df[df.category.isin({"P_LP", "B_LB"})]
    samp = pd.concat([dfb[dfb.category == "P_LP"].sample(args.n_per_class, random_state=args.seed),
                      dfb[dfb.category == "B_LB"].sample(args.n_per_class, random_state=args.seed)])
    bundle = load_evo2()
    fetch = get_fasta_loader(args.fasta_dir)
    layer_names = all_layer_names()

    records = []
    with HookedEvo2(bundle) as hooked:
        for _, row in samp.iterrows():
            if len(row.ref) != 1 or len(row.alt) != 1:
                continue
            seq_ref, seq_alt, vi = build_seq(fetch, row.chrom, int(row.pos), row.ref.upper(), row.alt.upper())
            if seq_ref is None:
                continue
            ids_ref = tokenize(seq_ref, bundle, device="cuda")
            ids_alt = tokenize(seq_alt, bundle, device="cuda")
            with torch.no_grad():
                hs_ref = extract_hidden_states(bundle, ids_ref, save_layers=layer_names)
                hs_alt = extract_hidden_states(bundle, ids_alt, save_layers=layer_names)
                base_alt_final = hs_alt[layer_names[-1]][0, vi].float()
                for ell in args.layers:
                    hr = hs_ref[f"blocks.{ell}"][0, vi].float()
                    ha = hs_alt[f"blocks.{ell}"][0, vi].float()
                    budget = (ha - hr).norm().item() + 1e-12
                    for mode in ("full", "direction", "magnitude"):
                        vec = isolate(hr, ha, mode)
                        hooked.set_patch(ell, vi, vec)
                        hs_p = extract_hidden_states(bundle, ids_alt, save_layers=layer_names)
                        hooked.clear()
                        final = hs_p[layer_names[-1]][0, vi].float()
                        shift = (final - base_alt_final).norm().item() / budget
                        records.append({"gene": row.gene, "category": row.category,
                                        "layer": ell, "mode": mode, "downstream_shift_norm": shift})

    res = pd.DataFrame(records)
    res.to_csv(out / "patching_results.csv", index=False)
    # paired Wilcoxon per layer: direction vs magnitude
    from scipy.stats import wilcoxon
    verdict = {}
    for ell in args.layers:
        sub = res[res.layer == ell].pivot_table(index=["gene", "category"], columns="mode",
                                                 values="downstream_shift_norm")
        sub = sub.dropna()
        if len(sub) >= 6:
            stat, pval = wilcoxon(sub["direction"], sub["magnitude"])
            verdict[f"L{ell}"] = {"n": int(len(sub)),
                                  "median_direction": float(sub["direction"].median()),
                                  "median_magnitude": float(sub["magnitude"].median()),
                                  "wilcoxon_p": float(pval),
                                  "direction_dominates": bool(sub["direction"].median() > sub["magnitude"].median())}
    (out / "causal_effect_summary.json").write_text(json.dumps(verdict, indent=2))
    log.info("Done -> %s", verdict)


if __name__ == "__main__":
    main()
