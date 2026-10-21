"""Verify Phase 0 artifacts; keep data/runtime readiness separate from design checks."""
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from uuid import UUID

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.forecast_contract import ROOT, load_contract, origin_boundaries


def check():
    contract = load_contract()
    plan = json.loads((ROOT / 'configs/experiment_manifest_v1.json').read_text())
    catalog = json.loads((ROOT / 'configs/source_catalog_v1.json').read_text())
    pins = json.loads((ROOT / 'configs/research_pins.json').read_text())
    if plan['contract_version'] != contract['contract_version']:
        raise ValueError('Experiment/forecast contract mismatch')
    previous_end = None
    roles = []
    for split in plan['splits']:
        start, end = date.fromisoformat(split['start']), date.fromisoformat(split['end'])
        if start > end or (previous_end and start != previous_end + timedelta(days=1)):
            raise ValueError('Split overlap, gap, or inverted range')
        roles.append(split['role'])
        previous_end = end
    expected_roles = ['point_train','point_validation','calibrator_train','calibrator_validation','calibration_initialisation','locked_test']
    if roles != expected_roles or plan['test_locked'] is not True:
        raise ValueError('Unexpected split roles or unlocked test')
    first_target = date.fromisoformat(plan['splits'][0]['start'])
    origin = origin_boundaries(first_target - timedelta(days=1), contract)
    if date.fromisoformat(plan['input_warmup_start']) > origin['context_start'].date():
        raise ValueError('Warmup starts after required context')
    if plan['budget']['paid_compute_enabled'] is not False:
        raise ValueError('Paid compute must remain disabled in Phase 0')
    if {r['year'] for r in catalog['annual_resources']} != set(range(2017,2025)):
        raise ValueError('Missing planned annual source')
    for source in catalog['annual_resources'] + [catalog['daily_resource']]:
        UUID(source['resource_id'])
    if catalog['raw_bytes_audited'] or catalog['full_coverage_verified']:
        raise ValueError('Phase 0 must not overstate the preview audit')
    if {ref['method'] for ref in pins['references']} != {'timexer','ncc'}:
        raise ValueError('Expected exactly the two selected additional methods')
    for ref in pins['references']:
        if not re.fullmatch('[0-9a-f]{40}', ref['reference_commit']):
            raise ValueError('Unpinned research reference')
        if ref['recursive_tree_truncated']:
            raise ValueError('Incomplete licence tree audit')
    missing = []
    for document in (ROOT / 'docs').rglob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', document.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','#','mailto:','sandbox:')):
                continue
            target_path = target.split('#',1)[0]
            if target_path and not (document.parent / target_path).exists():
                missing.append(str(document.relative_to(ROOT)) + ': ' + target)
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    for target in re.findall(r'\]\(([^)]+)\)', readme):
        if not target.startswith(('https:','http:','#')) and not (ROOT / target).exists():
            missing.append('README.md: ' + target)
    if missing:
        raise ValueError('Broken local links: ' + '; '.join(missing))
    return {
        'phase0_contract_and_artifact_checks': 'passed',
        'forecast_contract': contract['contract_version'],
        'research_references': len(pins['references']),
        'registered_annual_sources': len(catalog['annual_resources']),
        'full_source_quality_audit': 'pending_phase_2',
        'runtime_checks': 'not_evaluated_by_this_phase0_command',
        'training_ready': False,
    }


if __name__ == '__main__':
    try:
        print(json.dumps(check(), indent=2))
    except (ValueError, KeyError, TypeError, OSError) as error:
        print('Phase 0 check failed: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
