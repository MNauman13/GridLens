"""Turn the frozen forecasting contract into concrete, timezone-aware boundaries.

This is a verification tool, not the production feature sampler (Phase 3).
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / 'configs' / 'forecast_contract_v1.json'


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict:
    """Load and validate the versioned contract."""
    contract = json.loads(path.read_text(encoding='utf-8'))
    validate_contract(contract)
    return contract


def validate_contract(contract: dict) -> None:
    """Reject incompatible timing or target changes under the frozen v1 ID."""
    expected = {
        'contract_version': 'gb_nd_day_ahead_v1',
        'target': 'NESO_ND', 'unit': 'MW',
        'storage_timezone': 'UTC', 'display_timezone': 'Europe/London',
        'interval_minutes': 30, 'interval_semantics': '[start,end)',
        'issue_time_utc': '09:30:00',
        'context_end': 'issue_day_00:00_UTC_exclusive',
        'context_intervals': 336, 'gap_intervals': 48,
        'target_day_offset': 1, 'horizon_intervals': 48,
        'origin_frequency': 'daily',
        'nominal_coverage_levels': [0.8, 0.9, 0.95],
        'coverage_scope': 'marginal_per_output_slot',
        'historical_observations_rule': 'interval_end <= context_end_exclusive',
        'captured_availability_rule': 'first_seen_at <= issued_at',
        'forecast_covariate_rule': 'source_issued_at <= issued_at',
        'incomplete_input_policy': 'fail_without_moving_cutoff',
        'predictions_immutable': True,
    }
    if not isinstance(contract, dict) or contract.keys() != expected.keys():
        raise ValueError('The v1 contract has missing or unknown fields')
    for key, value in expected.items():
        if type(contract[key]) is not type(value) or contract[key] != value:
            raise ValueError(f'Incompatible v1 field: {key}; create a new contract version')


def origin_boundaries(issue_date: date, contract: dict) -> dict:
    """Derive a daily context, deliberate gap, and 48 target intervals."""
    if type(issue_date) is not date:
        raise TypeError('issue_date must be a date, not a naive or local datetime')
    # Validate mappings supplied by callers, not only mappings loaded from disk.
    validate_contract(contract)
    midnight = datetime.combine(issue_date, time(), tzinfo=timezone.utc)
    issued_at = datetime.combine(issue_date, time.fromisoformat(contract['issue_time_utc']), tzinfo=timezone.utc)
    step = timedelta(minutes=contract['interval_minutes'])
    context_start = midnight - contract['context_intervals'] * step
    target_start = midnight + timedelta(days=contract['target_day_offset'])
    points = []
    for slot in range(contract['horizon_intervals']):
        start = target_start + slot * step
        points.append({
            'slot': slot, 'interval_start': start, 'interval_end': start + step,
            'lead_minutes': int((start - issued_at).total_seconds() // 60),
            'weekly_baseline_start': start - timedelta(days=7),
            'last_available_day_baseline_start': midnight - timedelta(days=1) + slot * step,
        })
    return {
        'contract_version': contract['contract_version'],
        'issued_at': issued_at, 'context_start': context_start,
        'context_end_exclusive': midnight,
        'last_observed_interval_start': midnight - step,
        'gap_intervals': int((target_start - midnight) / step),
        'target_start': target_start,
        'target_end_exclusive': target_start + len(points) * step,
        'points': points,
    }


def serialise(value):
    """Use explicit Z timestamps in the human-readable JSON example."""
    if isinstance(value, datetime):
        return value.isoformat().replace('+00:00', 'Z')
    raise TypeError(f'Cannot serialise {type(value).__name__}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--issue-date', type=date.fromisoformat, default=date(2024, 2, 10))
    parser.add_argument('--contract', type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    result = origin_boundaries(args.issue_date, load_contract(args.contract))
    print(json.dumps(result, default=serialise, indent=2))


if __name__ == '__main__':
    main()
