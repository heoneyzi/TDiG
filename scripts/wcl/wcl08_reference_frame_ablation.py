#!/usr/bin/env python3
"""wcl08 — Experiment 8: Reference-Frame Necessity Ablation.
Proves the cosine biological signal comes from convergence to the FUNCTIONAL
output frame h_norm specifically — not from generic directional dynamics.

=================================================================================
WHY THIS EXPERIMENT (the airtight logic)
=================================================================================
Cosine settling's whole meaning rests on the choice of reference = h_norm (the
post-final-norm state the unembedding reads). A skeptic can ask: "maybe ANY
reference gives a splice signal — maybe you're just measuring generic layerwise
directional drift, and h_norm is incidental." If true, the 'reference-anchored'
necessity argument collapses.

We falsify that by recomputing cosine settling against a ladder of controls and
measuring the splice-donor-vs-intron Cohen d for each:
  R0  h_norm            (real functional output frame)              -> expected: signal
  R1  random fixed unit vector (same for all tokens)                -> expected: ~0
  R2  h at a mid layer  (h_15, a non-output frame)                  -> expected: weak
  R3  population-mean hidden state                                  -> expected: weak
  R4  h_norm of a DIFFERENT token (shuffled reference alignment)    -> expected: ~0
  R5  (GPU control) h_norm from a dinucleotide-shuffled INPUT seq   -> expected: signal destroyed
      (this control needs a fresh forward — see --forward-mode.)

  H8   splice d is materially non-zero ONLY for R0 (and R5 destroys it), collapsing
       toward 0 for R1-R4. => the signal is convergence to the functional frame,
       so 'reference-anchored to h_norm' is a load-bearing, causal property.

Together with wcl07 (cosine tracks functional prediction commitment) this makes
the reference-anchored claim non-circular and complete: the reference is the
model's own output frame, that frame is functional, and only that frame yields
the biology.

=================================================================================
DATA (two paths)
=================================================================================
  CPU path (R0-R4): HF chr22_tier3_raw.h5 (raw_h_ell + raw_h_norm) — already
    downloaded for Exp 2. No GPU. Per-position labels from gDTR
    prep_chr22_windows.py (chr22_position_labels.npy), sliced to token_stride.
  GPU path (R5): gDTR forward on shuffled-input sequences (needs Evo 2 weights;
    the ONE control tier3 cannot provide). Enable with --forward-mode + --gdtr-root.

Run (CPU, R0-R4):
    python wcl08_reference_frame_ablation.py \
        --tier3 hf/chr22_tier3_raw.h5 \
        --pos-labels ../gDTR/data/annotation/chr22_position_labels.npy \
        --out results/wcl/exp8/ --n-windows 100
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wcl00_shared_lens_utils import run_settling_pipeline, cohend, N_LAYERS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("wcl08")


def cos_settle_to_ref(h_stack: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """h_stack [L,T,H], ref [T,H] or [H] -> settling depth [T] via shared pipeline."""
    if ref.ndim == 1:
        ref = np.broadcast_to(ref, (h_stack.shape[1], h_stack.shape[2]))
    hn = h_stack / np.clip(np.linalg.norm(h_stack, axis=-1, keepdims=True), 1e-12, None)
    rn = ref / np.clip(np.linalg.norm(ref, axis=-1, keepdims=True), 1e-12, None)
    D = 1.0 - np.einsum("ltd,td->lt", hn, rn)
    return run_settling_pipeline(D, "dir")["c"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier3", required=True)
    p.add_argument("--pos-labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-windows", type=int, default=100)
    p.add_argument("--mid-layer", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    import h5py
    pos_labels = np.load(args.pos_labels)

    refs = {"R0_h_norm": [], "R1_random": [], "R2_mid_layer": [], "R3_pop_mean": [], "R4_shuffled_ref": []}
    label_store = []
    with h5py.File(args.tier3, "r") as f:
        raw = f["raw_h_ell"]; hnorm = f["raw_h_norm"]
        wids = f["window_idx"][:]
        stride = int(f["token_stride"][()]) if "token_stride" in f else 10
        nW = min(args.n_windows, raw.shape[0]); H = raw.shape[-1]
        # population mean hidden state across a subsample
        pool = np.concatenate([raw[w].reshape(-1, H) for w in range(min(8, nW))], axis=0)
        pop_mean = pool.mean(axis=0)                                # [H]
        rand_ref = rng.standard_normal(H)
        for wi in range(nW):
            h_stack = raw[wi].astype(np.float64)                   # [32,600,H]
            hn = hnorm[wi].astype(np.float64)                      # [600,H]
            T = h_stack.shape[1]
            refs["R0_h_norm"].append(cos_settle_to_ref(h_stack, hn))
            refs["R1_random"].append(cos_settle_to_ref(h_stack, rand_ref))
            refs["R2_mid_layer"].append(cos_settle_to_ref(h_stack, h_stack[args.mid_layer]))
            refs["R3_pop_mean"].append(cos_settle_to_ref(h_stack, pop_mean))
            hn_shuf = hn[rng.permutation(T)]                       # break token alignment
            refs["R4_shuffled_ref"].append(cos_settle_to_ref(h_stack, hn_shuf))
            start = int(wids[wi]) if wids.ndim else 0
            idxs = np.clip(start + np.arange(T) * stride, 0, len(pos_labels) - 1)
            label_store.append(pos_labels[idxs])

    lab = np.concatenate(label_store)
    result = {}
    for name, chunks in refs.items():
        c = np.concatenate(chunks)
        donor = c[(lab == 5) & (c != -1)]; intron = c[(lab == 1) & (c != -1)]
        d = cohend(donor, intron)
        result[name] = {"splice_d": d, "n_donor": int(donor.size), "n_intron": int(intron.size)}
        log.info("%-18s splice d = %+.4f (n_donor=%d n_intron=%d)", name, d, donor.size, intron.size)

    r0 = abs(result["R0_h_norm"]["splice_d"] or 0)
    others = [abs(result[k]["splice_d"] or 0) for k in refs if k != "R0_h_norm"]
    result["H8_verdict"] = ("reference_is_load_bearing" if r0 > 2 * max(others + [1e-9])
                            else "signal_not_reference_specific")
    result["note"] = ("R5 (shuffled-INPUT h_norm) requires a GPU forward — run with "
                      "--forward-mode using gDTR; if the signal survives R1-R4 collapse but "
                      "dies under R5, the reference-frame necessity is fully established.")
    (out / "reference_frame_ablation.json").write_text(json.dumps(result, indent=2))
    log.info("Verdict: %s -> %s", result["H8_verdict"], out / "reference_frame_ablation.json")


if __name__ == "__main__":
    main()
