# ADR 0006: deterministic cyclic calendar arithmetic

The initial Windows/Linux rebuild agreed on labels, scaling, and most feature IDs. Training/test feature aggregates differed because platform math libraries can differ at rounding boundaries in sin/cos. Keeping a frozen SHA check exposed this instead of hiding it behind a numeric tolerance.

Calendar phases now use integer slot numerators and a fixed Decimal calculation at precision 40, a declared pi constant, and twelve-decimal quantisation. A bounded cache reuses the 35,143 relevant cycle phases. Negative zero is normalised. This affects deterministic calendar metadata, not the definition of demand or source truth.

The complete freeze and baseline artifacts were regenerated for review. The review workflow writes only to its ephemeral workspace and artifacts; it cannot push. The normal live-data job still requires exact equality with the committed dataset manifest and annual source hashes. Prediction metrics retain their separate 1e-6 MW cross-BLAS comparison.

The calendar portability workflow fingerprints every phase for denominators 7, 48, 17,520, and 17,568. Windows must match Linux exactly. The manifest declares `fixed_decimal_cyclic_v1`. Earlier versions remain available in Git history.

This is a reproducibility choice, not an additional forecasting method.
