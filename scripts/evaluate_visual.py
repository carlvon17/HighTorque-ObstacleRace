"""Post-run evaluation only: compare recorded motion with the independent world."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
from sim.navigation import Track


def evaluate(run):
    summary = json.loads((run / 'summary.json').read_text())
    data = np.genfromtxt(run / 'telemetry.csv', delimiter=',', names=True)
    states = np.load(run / 'states.npz')
    xy = np.column_stack([data['truth_x'], data['truth_y']])
    track = Track()
    start = next(item for item in track.config['starts'] if item['id'] == summary['start'])
    heading = np.deg2rad(start['yaw_degrees'])
    normal = np.array([np.cos(heading), np.sin(heading)])
    tangent = np.array([-normal[1], normal[0]])
    stripe_offset = xy - np.array(start['position'])
    signed_distance = stripe_offset @ normal
    lateral_distance = stripe_offset @ tangent
    finish = summary['finish_time_s']
    result = {
        'status': summary['status'],
        'finish_time_s': finish,
        'full_lap_progress_ratio': float(data['eval_progress'][-1] / summary['track_length_evaluation_only_m']),
        'minimum_recorded_base_height_m': float(states['qpos'][:, 2].min()),
        'recorded_state_hz': float(1 / np.median(np.diff(data['t']))),
        'event_detail': summary['event_detail'],
        'referee_note': f"Contacts checked during original run at {summary.get('locomotion', {}).get('pd_hz', 500):g} Hz; foot overlap at 50 Hz. Partial foot overhang is allowed.",
    }
    if finish is not None:
        index = np.searchsorted(data['t'], finish)
        result['finish_trigger_normal_offset_m'] = float(signed_distance[index])
        result['finish_trigger_lateral_offset_m'] = float(lateral_distance[index])
        crossings = np.flatnonzero((signed_distance[:-1] < 0) & (signed_distance[1:] >= 0) &
                                    (data['eval_progress'][1:] > track.length*.8) &
                                    (abs(lateral_distance[1:]) < .25)) + 1
        result['physical_finish_crossing_s_evaluation_only'] = float(data['t'][crossings[0]]) if len(crossings) else None
        result['post_finish_observation_s'] = float(data['t'][-1] - finish)
        result['post_finish_displacement_m'] = float(np.linalg.norm(xy[-1] - xy[index]))
        tail = data['t'] > data['t'][-1] - 1
        result['last_second_displacement_m'] = float(np.linalg.norm(xy[tail][-1] - xy[tail][0]))
        result['last_second_max_command'] = [float(np.max(abs(data[k][tail]))) for k in ('vx', 'vy', 'wz')]
    result['full_lap_verified'] = bool(summary['status'] == 'completed' and
                                       .98 < result['full_lap_progress_ratio'] < 1.05 and
                                       summary['event_detail'] is None and
                                       result.get('physical_finish_crossing_s_evaluation_only') is not None and
                                       -.03 <= result['finish_trigger_normal_offset_m'] <= .15 and
                                       abs(result['finish_trigger_lateral_offset_m']) < .25)
    (run / 'evaluation.json').write_text(json.dumps(result, indent=2))

    # This plot is deliberately downstream of control. Its known map and truth
    # trajectory are never read by the autonomous navigation process.
    c = track.config
    canvas = Image.open(run / 'camera_carpet.png').resize((1040, 640))
    draw = ImageDraw.Draw(canvas)
    low = np.array(c['carpet_center']) - np.array(c['carpet_dimensions'][:2]) / 2
    scale = np.array([1040, 640]) / np.array(c['carpet_dimensions'][:2])

    def pixel(p):
        q = (np.asarray(p) - low) * scale
        return float(q[0]), float(640 - q[1])

    for obstacle in c['obstacles']:
        x, y = obstacle['position']
        x += summary['obstacle_shift_world_only_m']
        draw.rectangle([pixel([x-.1, y+.1]), pixel([x+.1, y-.1])], fill=tuple(obstacle['rgb']))
    draw.line([pixel(p) for p in xy], fill=(255, 220, 40), width=3)
    draw.ellipse([pixel(xy[0]+[-.05, .05]), pixel(xy[0]+[.05, -.05])], fill='white')
    draw.text((16, 16), 'RGB-D NAV | YELLOW: recorded physical trajectory (evaluation only)', fill='white')
    finish_label = f'{finish:.2f}s' if finish is not None else 'not reached'
    draw.text((16, 34), f"{summary['status']} | visual finish: {finish_label} | original robot collision geometry", fill='white')
    canvas.save(run / 'trajectory.png')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    for run in args.runs:
        print(run, json.dumps(evaluate(run), indent=2))
