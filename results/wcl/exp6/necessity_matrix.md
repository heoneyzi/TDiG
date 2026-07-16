# Lens Necessity Matrix (TDiG v2 protocol, chr22; transfer to chr17)

Source: TDiG `results/` Stage-6 CSVs. Every lens scored on the criteria the
settling-depth construct requires. Cosine is **not** the strongest biological
discriminator — the reference-free trajectory lens is — so the necessity claim
is about *construct match* (bounded, reference-anchored, well-posed threshold
crossing to the output-ready frame), not effect-size superiority.

| Lens | bounded | ref-anchored | well-posed | biol. d (donor−intron) | chr22→chr17 | settling AUROC |
|---|:--:|:--:|:--:|--:|--:|--:|
| Cosine / direction (paper's lens) | ✓ | ✓ | ✓ | -0.099 | 105% | 0.541 |
| Magnitude ratio (r−1) | ✗ | ✓ | ✓ | +0.174 | 108% | 0.552 |
| Whitened Mahalanobis distance | ✗ | ✓ | ✓ | -0.155 | 117% | 0.629 |
| Path tortuosity | ✗ | ✗ | ✓ | -0.437 | 97% | nan |
| Trajectory velocity+curvature | ✗ | ✗ | ✓ | -0.889 | 90% | 0.558 |

**Reading of the matrix.**
- Magnitude (M2) and whitened distance (M4) collapse to a degenerate cell under
  two of three reference conventions and, where defined, carry the wrong-sign / weak
  biological signal — they are not well-posed settling constructs.
- Trajectory (M3_geo) is the strongest biological discriminator and transfers well,
  BUT is reference-free: it measures trajectory smoothness, not commitment to the
  model's output-ready frame — a different question (paper Def-2/Def-3 split).
- Cosine (M1) is the only bounded, reference-anchored, non-degenerate lens whose
  threshold crossing is the paper's exact construct; its weaker raw d is the cost of
  measuring the *right* target rather than the easiest-to-separate one.

*Rogue-dimension (Exp 2): top-8 variance share at L29 = 0.007573627191469916.*
*Incremental AUROC (Exp 3b): D+M vs M = 0.009959725430868479.*
