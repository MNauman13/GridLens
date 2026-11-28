# ADR 0007: reproducible upgrade baseline and a required live release

Status: accepted for I0 implementation, 24 November 2026.

The accepted Phase 8 is a bounded pilot. Its API serves saved examples and the linked neural artifact has expired from the run's artifact listing. We need a runnable, versioned baseline before adding inference optimisations.

Preserve `gb_nd_day_ahead_v1`, source hashes, split membership, training-only scaling, seed 17 and the original recovery training budget. Regenerate the existing numerical Transformer and selected patch-48 TimeXer when original bundles cannot be recovered. Regenerated weights get their actual new identities; original reports and the original champion are not overwritten. Differences from published pilot predictions are recorded and are not attributed to an optimisation. Model selection and 2024 scoring remain deferred.

An authoritative manual attention/reference path remains mandatory. Optional accelerated adapters may use established kernels/export runtimes only alongside that reference, with explicit mask conversion, numerical parity and separately versioned runtime identities. This changes the earlier blanket restriction on using convenience execution operators for the optional accelerated path; it does not permit replacing the manual research implementation or copying unlicensed reference code.

Public anonymous real-inference deployment is now an explicit delivery goal, rather than an optional post-R2 choice. I1 produces the bounded point-inference preview; later increments add calibrated replay and hardened operations. Public deployment must not expose locked-test evaluation before its declared release gate. This does not authorise unbounded paid compute.

`configs/upgrade_experiment_v2.json` freezes baseline pins, 32 validation origins, warmup/repeat budgets, numerical tolerances, CPU settings and future trial/service targets. The additional cold-process protocol uses 20 fresh interpreters per model and explicitly distinguishes interpreter startup, verified model load and first prediction. Host file caches may remain warm.

Keep original Phase 9-14 methodological responsibilities, but use I0-I9 for the upgrade sequence so an early service slice and inference work can precede completion of interval methods. I0 introduces no new architecture, cache optimisation or accuracy-superiority claim.

Model bundles must have a verified durable copy outside expiring CI artifacts. Archive only explicit deployable files and evidence, omit raw datasets/checkpoints/secrets, and keep binary artifacts out of Git. Regeneration remains documented so another developer does not need access to a private retained archive.
