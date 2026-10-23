# Phase 5 results

Implemented scaled dot-product attention, learned multi-head Q/K/V/output projections, boolean causal/padding masks, and sinusoidal positional buffers. The implementation uses tensor primitives; PyTorch's built-in attention is used only as an independent oracle in tests.

Critical CPU tests passed on [the attention-core CI run](https://github.com/MNauman13/GridLens/actions/runs/40016626636): hand arithmetic, self/cross parity, gradient parity, finite gradients, numerical gradcheck, future/padding invariance, invalid masks and fully blocked queries, known odd-width positions, offsets, state reload, and half-precision accumulation. CUDA is conditionally checked and was unavailable on the hosted CPU runner.

The lesson at `/learn/attention` has query/head controls, causal masking, exact scores/weights/value mixtures, calculation playback, and a separate positional animation. An independent calculator fixture is checked against the actual PyTorch modules in CI. Browser checks compare rendered values and controls at desktop/mobile sizes.

The complete encoder-decoder is the next planned phase. No trained Transformer or sequence-quality result is claimed here. See the [walkthrough](../learning/phase-5.md) and final [verification report](phases-3-5.md).
