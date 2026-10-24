# Neural pilot model card

## Intended demonstration

GridLens demonstrates an auditable day-ahead electricity forecasting pipeline, manual PyTorch attention and published TimeXer operators. It forecasts NESO National Demand in MW, not an interchangeable generic electricity series. It is an application and reproduction project, not a new research method.

The issue is 09:30 UTC. Inputs contain 336 half-hours through midnight at the start of the issue day, and outputs contain 48 half-hours of the following UTC day. The intervening 48 issue-day slots are unobserved. Actual target-day values never become inference inputs.

## Data and evaluation

Only frozen 2018–2021 training origins fit weights and global scaling. Label-maturity embargoes retain 1,460 training and 364 validation origins. Validation covers 2022. Source hashes, feature hashes, evidence mode and excluded boundary dates are recorded in the dataset freeze. This is retrospective revised data, not proof of historical publication availability. No model selection uses 2024 scores.

Inputs contain historical demand, wind, solar, missing flags, feature ages and deterministic calendar encodings. Only two-slot causal forward fills are allowed in history; targets are not filled. Target calendar features are known in advance. No future weather forecasts, demand observations, wind or solar measurements are used.

## Models and selection

The original token Transformer has manual multi-head attention, post-norm blocks, sinusoidal positions, shifted targets and bounded generation. A seeded reversal reproduction verifies learnability on explicitly disjoint sequences. Its poor longer-length performance remains disclosed.

Numeric Transformer and LSTM pilots generate autoregressively at validation. Their teacher-forced training creates exposure bias. TimeXer uses endogenous patches, one global token and external variable tokens, followed by a GridLens gap/calendar adapter. The gap-free reference route and application changes are documented separately.

Eight neural candidates comprise two numerical architectures and six TimeXer configurations. Each receives seed 17 and at most 12 epochs; patience is four. Candidate selection and early stopping both use the development validation period. This is a bounded pilot, not the full 17/42/101 multi-seed plan. Maximum-epoch/patience settings are smaller than the full experiment budget and are reported as such.

The champion minimizes full-validation RMSE among these candidates and five frozen baseline candidates. `configs/point_champion_v1.json` records the identity, source/data/report references, configuration and exact bundle manifest hash. The result must be interpreted with selection uncertainty; it is not an unbiased held-out test estimate.

## Measured evidence and uncertainty

`reports/neural-validation-v1.json` contains MAE, RMSE, per-slot errors, peak errors, best-epoch histories and measured CPU fit and inference times. Full aligned predictions and actuals are in the experiment artifact. Latency uses five warmed single-origin calls and is environment-specific; it is not a service load benchmark.

Paired circular moving-day-block bootstrap intervals use 2,000 replicates, seed 17 and 7/14-day block sensitivity. They preserve paired days and short-range dependence better than treating half-hours independently. They are conditional on the fitted seed and validation selection; they do not measure seed or selection uncertainty.

## Serving and limitations

The web/API slice serves integrity-checked, immutable validation examples. `NeuralBundle` loads actual weights for arbitrary complete eligible feature snapshots. Requests do not train models. The output contract enforces finite values and exact horizon length; it does not clip regression predictions or guarantee nonnegative demand.

This phase does not provide calibrated intervals, online feedback, weather-forecast inputs, production deployment, or evidence of future accuracy. The broader patch-length-12 and multi-seed comparisons remain deferred. Remaining project phases implement calibration, replay, operational integration and final locked evaluation.

Weights and checkpoints are available in linked GitHub Actions artifacts, whose expiry is recorded. The frozen source and data allow regeneration; download the artifact for long-term archival. No checkpoint binary or raw data is committed to Git.
