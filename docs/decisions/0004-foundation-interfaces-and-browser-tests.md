# ADR 0004: foundation boundaries and repeatable browser verification

Status: implemented; verification results are recorded in the Phase 1 progress report.

## Package boundaries

Use a `src/gridlens` Python package and an installed `gridlens` command. This catches missing package files and packaging assumptions earlier than running everything from notebooks. Phase 1 has a synthetic contract fixture, not a trained forecast.

Pydantic records reject unknown fields and non-finite numeric values. Input windows enforce the issue time, cutoff, seven-day context, one-day gap, and target boundaries. Point forecasts require 48 values. Interval forecasts require valid ordered bands and nesting. Feedback records require a completed target interval and explicit UTC availability.

Model interfaces accept a dataset iterator rather than making a model fetch source data. The point model and calibrator are separate protocols. A calibrator receives matured feedback through an explicit `observe` method, so later implementations can enforce event-time eligibility rather than updating in notebook row order.

JSON Schema exports make field structure inspectable. Python cross-field validators remain authoritative for temporal arithmetic and interval nesting; JSON Schema alone does not encode every runtime invariant.

## Artifact identity

An artifact manifest contains relative paths, byte sizes, and SHA-256 hashes. The manifest ID is the hash of canonical JSON. Verification fails on changed bytes, missing files, duplicate paths, traversal, or foreign absolute-path syntax. Generated weights remain outside Git.

## Browser architecture

The browser opens Next.js on port 3000. Its same-origin `/api/fixture` route forwards to the Python API on port 8000. Only the Next.js server reads `GRIDLENS_API_URL`. The browser receives neither provider credentials nor a configurable server URL.

The foundation page distinguishes unverified, loading, verified, and error states. It never displays synthetic inputs as measured forecasting performance. The step-through timeline is schematic, not a proportionally scaled duration chart.

## Two verification environments

Local verification requires the server process, HTTP client, and browser to reach the same loopback interface. In this restricted task environment a direct loopback request failed with Windows socket error 10013, and the in-app browser timed out. Starting a server is not proof that this route works.

Playwright's `webServer` configuration starts the API and web app, waits for their health URLs, runs browser assertions at desktop/mobile sizes, checks JavaScript errors and horizontal overflow, and retains screenshots/traces. Run this normally on a developer machine or GitHub-hosted CI. Do not weaken local network protections or expose a public tunnel to work around a restriction.

GitHub CI has read-only repository permissions and a 20-minute job limit. It does not push changes or store credentials. During bootstrap it emits only allowlisted dependency-lock files for review; after those are committed, install commands enforce them.

## Runtime separation

The core/API environment is lightweight. PyTorch is an explicit `ml` extra resolved from the official CPU index, so a web test does not install CUDA libraries. A separate CPU forward/backward smoke step verifies that ML dependency set. GPU support is a later hardware-specific configuration, not inferred from the visible MX130.
