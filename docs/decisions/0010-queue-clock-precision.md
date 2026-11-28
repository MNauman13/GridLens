# Queue timestamp precision

The first Compose campaign built and started both containers but failed its fresh-compute smoke check. PostgreSQL `REAL` stores float32 values. At current Unix epochs, adjacent representable values are about 128 seconds apart, so a five- or ten-second admission/lease boundary cannot be represented correctly. SQLite `REAL` uses double precision, which hid the problem during the local HTTP load campaign. The initial PostgreSQL unit clocks near 1000 seconds also missed it.

All queue timestamp columns now use `DOUBLE PRECISION`. Startup widens any legacy PostgreSQL float32 columns under the existing advisory transaction lock. Widening does not reconstruct precision already lost in historical values. The queue contract tests now use fractional current-scale Unix epochs. Additional tests verify migration, exact lease boundaries, process-death fencing and real-clock PostgreSQL model computation.

The regression campaign failed all six old PostgreSQL queue tests at the larger epoch before the fix, then passed eight integration checks after the fix, including migration, a real-clock worker and dump/restore equality. Compose is rerun independently.

This operational repair follows the frozen 2024 model evaluation. It changes queue persistence types, not point weights, model operators, calibration state, selection or test metrics. The retained evaluation still references the exact pre-test commit `cc39b120` for its model-scoring provenance. No forecast candidate is reselected or refitted because of the infrastructure fix.
