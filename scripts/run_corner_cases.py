"""Run bounded physical RGB-D fault cases and report measured recovery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navigation.runtime import core_hashes


def fault(kind, start, end):
    return {'kind': kind, 'start_s': start, 'end_s': end}


CASES = {
    'inner_only_cold': {'duration': 28, 'faults': [fault('hide_outer_line', 0, 20)]},
    'outer_only_cold': {'duration': 28, 'faults': [fault('hide_inner_line', 0, 20)]},
    'inner_only_curve': {'duration': 50, 'faults': [fault('hide_outer_line', 20, 40)]},
    'outer_only_curve': {'duration': 50, 'faults': [fault('hide_inner_line', 20, 40)]},
    'inner_only_small_curve': {'duration': 25, 'spawn': [.20, 1.25, np.pi],
                             'faults': [fault('hide_outer_line', 0, 18)]},
    'outer_only_small_curve': {'duration': 25, 'spawn': [.20, 1.25, np.pi],
                             'faults': [fault('hide_inner_line', 0, 18)]},
    'rgbd_loss_straight': {'duration': 18, 'faults': [fault('rgbd_black', 6, 8)]},
    'rgbd_loss_boxes': {'duration': 35, 'spawn': [2.55, 1.25, np.pi],
                        'faults': [fault('rgbd_black', 8, 9)]},
    'inner_only_boxes': {'duration': 35, 'spawn': [2.55, 1.25, np.pi],
                        'faults': [fault('hide_outer_line', 3, 20)]},
    'depth_holes_straight': {'duration': 20, 'faults': [fault('depth_holes', 6, 12)]},
    'heading_30_degrees': {'duration': 20, 'spawn': [.55, -1.25, np.pi/6], 'faults': []},
    'heading_minus_30_degrees': {'duration': 20, 'spawn': [.55, -1.25, -np.pi/6], 'faults': []},
}


def analyze(output, name):
    run = output/name
    summary = json.loads((run/'summary.json').read_text())
    observations = json.loads((run/'observations.json').read_text())
    telemetry = np.genfromtxt(run/'telemetry.csv', delimiter=',', names=True)
    times = telemetry['t']
    active = [row for row in observations if row.get('sensor_faults')]
    result = {
        'status': summary['status'], 'time_s': summary['sim_time_s'],
        'progress_m': summary['truth_progress_evaluation_only_m'],
        'event_detail': summary['event_detail'], 'event_time_s': summary['event_time_s'],
        'core_matches_current': summary['navigation_core_sha256'] == core_hashes(),
        'fault_frames': len(active),
        'modified_pixels': sum(item['altered_pixels'] for row in active for item in row['sensor_faults']),
        'valid_lane_frames_during_fault': sum(row['lane'] is not None for row in active),
        'ready_frames_during_fault': sum(row['reason'] == 'ready' for row in active),
        'faults': [],
    }
    moving = np.sqrt(telemetry['vx']**2+telemetry['vy']**2+telemetry['wz']**2) > 1e-6
    for item in summary['sensor_faults']:
        start, end = item['start_s'], item['end_s']
        start_index, end_index = np.searchsorted(times, [start, end])
        start_index = min(start_index, len(times)-1)
        end_index = min(end_index, len(times)-1)
        stopped = np.flatnonzero((times >= start) & (times < end) & ~moving)
        resumed = np.flatnonzero((times >= end) & moving)
        end_xy = np.array([telemetry['truth_x'][end_index], telemetry['truth_y'][end_index]])
        final_xy = np.array([telemetry['truth_x'][-1], telemetry['truth_y'][-1]])
        result['faults'].append({
            **item,
            'progress_during_fault_m': float(telemetry['eval_progress'][end_index]-telemetry['eval_progress'][start_index]),
            'first_zero_request_delay_s': float(times[stopped[0]]-start) if len(stopped) else None,
            'first_nonzero_request_after_restore_s': float(times[resumed[0]]-end) if len(resumed) else None,
            'displacement_after_restore_m': float(np.linalg.norm(final_xy-end_xy)),
        })
    result['completed_bounded_trial_without_referee_event'] = bool(
        summary['status'] == 'duration_limit' and summary['sim_time_s'] >= CASES[name]['duration']-.04)
    result['advanced_at_least_half_metre'] = result['progress_m'] >= .5
    result['resumed_after_fault'] = all(
        item['first_nonzero_request_after_restore_s'] is not None and
        item['displacement_after_restore_m'] >= .15 for item in result['faults'])
    result['passed'] = bool(result['completed_bounded_trial_without_referee_event'] and
                          result['advanced_at_least_half_metre'] and result['resumed_after_fault'] and
                          (not result['faults'] or result['modified_pixels'] > 0))
    return result


def execute(output, name, video):
    config = CASES[name]
    schedule = output/(name+'_faults.json')
    schedule.write_text(json.dumps(config['faults'], indent=2))
    command = [sys.executable, str(ROOT/'scripts/run_visual.py'), '--duration', str(config['duration']),
               '--sensor-faults', str(schedule), '--output', str(output/name)]
    if 'spawn' in config:
        command.extend(['--spawn', *map(str, config['spawn'])])
    if video:
        command.append('--video')
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1')
    with (output/(name+'.log')).open('w') as log:
        process = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if not (output/name/'telemetry.csv').exists():
        raise RuntimeError(f'{name} failed before saving telemetry, exit {process.returncode}')
    result = analyze(output, name)
    print(name, result['status'], round(result['progress_m'], 3), 'm', flush=True)
    return name, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=CASES, action='append')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = args.case or list(CASES)
    if args.summarize_only:
        results = {name: analyze(output, name) for name in selected}
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = dict(pool.map(lambda name: execute(output, name, args.video), selected))
    report = {'navigation_core_sha256': core_hashes(), 'cases': results,
              'scope': 'Bounded physical simulation scenarios with world-side sensor faults. ' 
                       'Referee events and physical motion are evaluated from recorded truth after control.',
              'acceptance': 'Complete the configured interval without a referee event, advance at least 0.5 m, '
                            'and after each fault restore a nonzero request and move at least 0.15 m.'}
    (output/'corner_results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
