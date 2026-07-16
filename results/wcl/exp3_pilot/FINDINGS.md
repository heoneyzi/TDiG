# Experiment 3 — Stage 0 pilot findings (CPU-only, existing cache)

Source: TDiG `variant_scalars.parquet` (8008 variants: 3514 P/LP,
4494 B/LB). See script docstring for why these numbers differ from
the paper's published Table A8 (different pipeline, same qualitative cohort).

## H3a — complementarity

| Feature | AUROC |
|---|---:|
| D_cos (32-d, direction) | 0.9383 |
| M_l2 (32-d, magnitude) | 0.9264 |
| D + M (64-d, combined) | 0.9512 |

Bootstrap (1000x) paired AUROC differences:
- D+M vs. M alone: +0.0248 (95% CI [0.0217522262051029, 0.028240514855719923])
- D+M vs. D alone: +0.0129 (95% CI [0.010645954636896238, 0.01529899411971274])

Both CIs exclude 0 in this pilot: **direction is not redundant with magnitude**,
even though magnitude alone is a strong classifier — H3a direction confirmed.

## H3b/H3c — argmax-layer localization ("where does the peak disruption sit?")

| | argmax(\|D_cos\|) | argmax(\|M_l2\|) |
|---|---:|---:|
| mean layer | 5.83 | 30.00 |
| std | 10.91 | **0.00** |
| top layers | {0: 5961, 27: 1367, 6: 183, 4: 168, 28: 132} | **100% at layer 30** |

Every single variant's largest \|Δh_L2\| falls at exactly layer 30 (mean ΔH_L2
at L30 = 1.459e+12, max = 1.017e+13 — an
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
