"""Check committed neural evidence, champion decision and public forecast lineage."""
import json
from pathlib import Path

from gridlens.data.store import content_id
from gridlens.saved_neural import SavedNeuralForecast

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT/'reports/neural-validation-v1.json').read_text())
    freeze = json.loads((ROOT/'reports/dataset-freeze-v1.json').read_text())
    baseline = json.loads((ROOT/'reports/baseline-validation-v1.json').read_text())
    champion = json.loads((ROOT/'configs/point_champion_v1.json').read_text())
    if report['report_id'] != content_id({k:v for k,v in report.items() if k != 'report_id'}):
        raise ValueError('Neural report integrity mismatch')
    assert report['dataset_manifest_id'] == freeze['manifest_id'] == champion['dataset_manifest_id']
    assert report['training_origins'] == len(freeze['splits']['point_train']['target_dates'])
    assert report['validation_origins'] == len(freeze['splits']['point_validation']['target_dates'])
    assert report['locked_test_evaluated'] is False
    assert report['timexer_trials'] <= 6 and report['numeric_trials'] <= 3
    scores = {r['identity']:r['metrics']['rmse_mw'] for r in report['results']}
    scores.update({r['model_version']:r['metrics']['rmse_mw'] for r in baseline['results']})
    selected = min(scores,key=lambda identity:(scores[identity],identity))
    assert selected == report['champion']['identity'] == champion['identity']
    assert champion['validation_report_id'] == report['report_id']
    for kind,payload in json.loads((ROOT/'src/gridlens/resources/neural_forecasts_v1.json').read_text()).items():
        forecast = SavedNeuralForecast.model_validate(payload)
        assert forecast.validation_report_id == report['report_id']
        assert forecast.dataset_manifest_id == freeze['manifest_id']
        assert forecast.architecture['kind'] == kind
        assert forecast.model_id in scores
    reversal = json.loads((ROOT/'reports/reversal-validation-v1.json').read_text())
    assert reversal['training_excludes_all_held_out_sequences'] is True
    assert reversal['exact_accuracy'] >= .95
    assert reversal['tiny_overfit_exact_accuracy'] >= .95
    print('Neural report, champion, forecast lineage and disjoint reversal gates verified')


if __name__ == '__main__':
    main()
