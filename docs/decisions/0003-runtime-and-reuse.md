# ADR 0003: CPU-first development and independent research implementation

Status: accepted for initial development; dependency resolution belongs to Phase 1.

The inspected runtime is Windows AMD64 with Python 3.12.7, Node 22.12.0, Git 2.52.0, and uv 0.11.21. Eight logical CPUs are visible. The GPU probe reports a GeForce MX130 with 2,048 MiB, but PyTorch is not installed and compatible tensor execution is unverified. Physical RAM could not be measured through the permitted audit APIs; no capacity is guessed.

Start on CPU using tiny fixtures. We do not base the full training plan on this GPU. Use optional GPU compute only after a separately measured compatibility/throughput check and an explicit budget decision.

The chosen interpreter has no torch, NumPy, pandas, pytest, Pydantic, FastAPI, or tzdata distribution. Phase 0 uses standard-library checks. Phase 1 creates the project environment and locks tested dependencies. On Windows, install tzdata before any Europe/London source conversion; UTC-only contract checks do not need it.

The experiment plan limits trials, epochs, seeds, and wall-clock time, and disables paid compute by default. A missing runtime capability is recorded honestly, not treated as a passing deployment check.

Both research reference repositories lack a located repository-wide licence grant at the pinned trees. Implement published methods independently. Keep provider attribution and research citations separate from project-code licensing.
