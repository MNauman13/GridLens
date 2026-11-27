# End-to-end completion work

Requested 27 November 2026. This checklist records executable evidence rather than marking design work complete.

- I0/I1 and fast-attention I2 are committed and publicly deployed.
- KV-cache I2 code passes initial correctness tests. Its interrupted timing run is not accepted evidence. The completed replacement retains 11,000 timings and establishes 1.54× B1 median speed-up for manual full caching, with 0.022935 MW maximum drift.
- I3 will compare exact ONNX CPU, dynamic INT8 and eager execution, with startup, precision and batch/thread scopes separated.
- I4 plan is frozen in configs/release_study_v1.json. Four neural configurations, three seeds, full 2022 validation, 50-epoch cap and patience eight. No 2024 selection.
- I5 requires fixed/rolling residual intervals, independently implemented NCC losses/control/TTA, matured feedback and replay restoration.
- I6 durable queue and failure/ownership tests pass on SQLite and real PostgreSQL. The 15-minute offered-load campaign is running. Scheduled capture and operational release checks remain.
- I7 requires real compute, forecast/interval/replay/comparison workflows and accessible browser verification.
- I8 remains locked until a final point/runtime/calibration/evaluation manifest has been committed before test inspection.
- I9 requires public smoke, archive retrieval, restore/rollback and a monitoring record. A seven-day availability observation cannot be manufactured during an implementation session.

Every completed increment must have a measured report, raw-result lineage, passing relevant tests and a clear account of unsupported or time-dependent claims.
