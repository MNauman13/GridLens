# Neural experiment operations

1. Install the frozen ML and development environment using the README commands.
2. Ingest the pinned 2017–2024 annual sources. The dataset builder must match the approved freeze exactly, including calendar encoding, label-maturity exclusions and scaling.
3. Run `python scripts/train_reversal.py` for the toy reproduction and `python -m scripts.train_neural` for the bounded electricity pilot. Both have fixed default budgets; changing configuration requires a new reviewed report.
4. Use `--resume` only for a matching neural run. The identity incorporates configuration, source hashes, data freeze, seed, epoch budget and Torch version. The sampler/dropout use Torch RNG, whose state is restored with model and optimizer state. A completed early-stopped run does not resume extra epochs.
5. Preserve `reports/local/neural/`: every run directory contains a resumable checkpoint, checkpoint manifest, best weights, metadata and final bundle manifest. `validation-predictions.json` contains aligned dates, observations and every candidate's full forecast array. Reports and public examples are separate JSON files.
6. Review the full validation report before updating the public examples or champion. `scripts/check_neural_reference.py` checks the committed decision, report digest, frozen counts, forecast lineage and reversal learning gates. Never populate the champion using the 2024 score.
7. Download the exact linked Actions artifact before its recorded expiry. The current workflow requests 90-day retention for future runs; the recorded experiment may have the earlier 30-day retention. Git stores small reports and examples, not binary weights or raw datasets.

Load a downloaded bundle in Python:

```python
from pathlib import Path
from gridlens.models.bundle import NeuralBundle

model = NeuralBundle.load_bundle(Path("path/to/run-directory"))
point = model.predict(eligible_feature_snapshot)
assert len(point.prediction_mw) == 48
```

`eligible_feature_snapshot` must come from the frozen causal data adapter. The demand-only `InputWindow` is insufficient to represent all historical features. Predictions return the shared `PointForecast` contract; they do not train or update weights.

The hosted workflow emits small reviewable report/example JSON chunks in its logs and uploads binary bundles as artifacts. It has read-only repository permissions and cannot silently overwrite committed reference evidence. Promoting a run means reviewing and committing the generated small files, retaining the exact artifact link and manifest hash.

The default pilot is exploratory: one seed, at most 12 epochs, patience four. The full experiment manifest's three seeds and larger training schedule are not claimed complete. Repeating diagnostics after implementation corrections is disclosed; it does not introduce more than six distinct TimeXer configurations. Broader hyperparameter search requires revising the declared budget.
