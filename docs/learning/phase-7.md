# Phase 7: numerical forecasting and training

The numeric encoder projects each of the 336 historical 15-feature rows into model width. The decoder receives a previous demand value and six known calendar features. Its scalar output is normalized demand, converted back to MW using statistics fitted on training contexts only.

History positions are 0–335. Positions 336–383 represent the unobserved issue-day gap; decoder positions begin at 384. The first numeric decoder input is the last available context demand, explicitly a seed across the gap rather than an observed target-day value. Later training inputs use shifted targets. Validation and inference repeatedly feed back previous predictions. The public forward method rejects teacher-forcing labels while in evaluation mode.

The LSTM uses the identical tensor input and output interface, training origins, target calendar and scaling. Its encoder state initializes a recurrent decoder. It also evaluates by generated rollout. Neither model consumes future wind, solar or demand.

`python -m scripts.train_neural` is a bounded CPU pilot, not the planned full multi-seed experiment. It uses seed 17, Adam at 0.001, batch 32, a maximum of 12 epochs and patience 4. Each epoch measures full 364-origin validation RMSE in inference mode. The minimum validation epoch supplies the final weights. The final report includes all 48-slot errors, peak errors, measured single-origin latency, configuration, dataset identity and loss history. Numeric teacher forcing can create exposure bias; poor rollout results remain in the report.

Checkpoints contain model and optimizer state, epoch, best state, patience counter, history and Torch RNG. Their content manifest is checked before loading; an identity hash rejects changes to dataset/configuration/seed/epoch budget. `--resume` continues a matching run. Final bundles contain a state dictionary, metadata and SHA-256 manifest. Loading uses `weights_only=True`; evaluation output equality is checked after reload. Checkpoints and bundles are retained as Actions artifacts for 30 days and should be downloaded for durable experiment archival.

Train and validation come from the approved frozen data report, including label-maturity exclusions. Preparing frozen membership does not score the 2024 role. Training and selection touch only `point_train` and `point_validation`. Published forecast examples are retrospective validation cases, not live forecasts.
