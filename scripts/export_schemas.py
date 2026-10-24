"""Export inspectable JSON Schemas from the authoritative Python contracts."""
import argparse
import json
import sys
from pathlib import Path

from gridlens.artifacts import ArtifactManifest
from gridlens.contracts import BaselineForecast, DatasetManifest, InputWindow, IntervalForecast, MaturedFeedback, PointForecast, TrainingExample
from gridlens.saved_neural import SavedNeuralForecast

ROOT = Path(__file__).resolve().parents[1]
MODELS = [DatasetManifest, InputWindow, PointForecast, IntervalForecast, MaturedFeedback, TrainingExample, ArtifactManifest, BaselineForecast, SavedNeuralForecast]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    stale = []
    for model in MODELS:
        path = ROOT / 'schemas' / (model.__name__ + '.json')
        content = json.dumps(model.model_json_schema(), sort_keys=True, indent=2) + '\n'
        if args.check:
            if not path.exists() or json.loads(path.read_text(encoding='utf-8')) != json.loads(content):
                stale.append(path.name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8', newline='\n')
    if stale:
        print('Stale schemas: ' + ', '.join(stale), file=sys.stderr)
        raise SystemExit(1)
    print('Contract schemas verified' if args.check else 'Contract schemas exported')


if __name__ == '__main__':
    main()
