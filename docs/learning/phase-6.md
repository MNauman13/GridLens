# Phase 6: the complete small Transformer

`models/transformer.py` builds on the checked manual attention operators. It never delegates attention or the architecture to `torch.nn.Transformer`.

An encoder block computes attention, adds the input residual, normalizes, then applies a position-wise two-layer ReLU network with another residual and normalization. This is the original **post-norm** order. Decoder blocks add a separate encoder cross-attention sublayer between masked self-attention and the feed-forward network. Source and target lengths may differ.

```mermaid
flowchart TD
  S[Source embeddings and positions] --> E[Encoder blocks]
  T[Shifted target embeddings] --> M[Causal self-attention]
  E --> C[Encoder cross-attention]
  M --> C
  C --> F[Feed-forward and vocabulary logits]
  F --> G[Generated prefix at inference]
  G --> M
```

Tokens reserve PAD=0, BOS=1 and EOS=2. Training predicts `reverse(source), EOS` from `BOS, reverse(source)`; the causal mask blocks later decoder inputs, and source padding is hidden from encoder and cross-attention keys. PAD targets do not contribute to cross-entropy. Embeddings are scaled by sqrt(width), followed by sinusoidal positions and dropout. Shared source/target vocabularies support tying the output matrix to the embedding matrix.

Generation has no target argument. It encodes the source once and repeatedly decodes the generated prefix, suppressing BOS/PAD until EOS. Finished rows are padded and a hard maximum length bounds execution. Evaluation requires `eval()` to disable dropout.

The reproduction uses the planned 128-wide, four-head, two-encoder/two-decoder model with 512-wide feed-forward layers. Its bounded CPU run uses 4,000 batches of 64 random sequences, lengths 3–8, seed 17, Adam (0.9,0.98), epsilon 1e-9, gradient clipping and the original schedule shape with a deliberately shortened 400-step warmup and 0.5 multiplier. Matrices use Xavier initialization so positions are not overwhelmed by default embedding variance. Every held-out sequence is explicitly excluded from training. A separate eight-sequence fixture overfits for 500 steps before resetting the full model and RNG. The run evaluates 256 held-out same-length sequences and 128 longer sequences.

The accepted run reaches **100%** exact accuracy on the tiny fixture and same-length held-out set, but **0%** on lengths 9–12. That failure is part of the result. The initial failed 3.5% diagnostic remains in `reports/reversal-initial-diagnostic.json`; accepted evidence is `reports/reversal-validation-v1.json`. Success on this toy task does not reproduce translation results.

Tests check causal prefix invariance, source-padding invariance, unequal lengths, finite gradients, weight reload equality, generation bounds and schedule shape. The trained checkpoint is an Actions artifact, not a Git-embedded binary. The report records exact sequence accuracy, loss history, runtime and reload verification. Do not infer success from code presence: consult the measured report.

Open `/learn/transformer` for the shifted-input and generated-prefix walkthrough.
