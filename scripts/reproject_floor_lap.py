"""Evaluate mapping by rendering a held Pro camera on a recorded body trajectory."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from deploy.geometry import quaternion_matrix_xyzw
from navigation.runtime import NavigationRuntime, core_hashes
from scripts.evaluate_floor_map import evaluate
from sim.robot import JOINTS
from sim.sensors import SensorRig


def reproject(trajectory, scene, output, seed):
    model = mujoco.MjModel.from_xml_path(str((scene/'scene.xml').resolve()))
    old_model = mujoco.MjModel.from_xml_path(str((trajectory/'scene.xml').resolve()))
    with np.load(trajectory/'states.npz', allow_pickle=False) as record:
        time, qpos = record['time'], record['qpos']
    camera_times = np.array([row['t'] for row in json.loads((trajectory/'observations.json').read_text())])
    # The physics recorder starts at the end of its first interval.
    camera_times = camera_times[(camera_times >= time[0]) & (camera_times <= time[-1])]
    indices = np.searchsorted(time, camera_times)
    if not len(indices) or not np.allclose(time[indices], camera_times, rtol=0, atol=1e-6):
        raise ValueError('Missing physics samples at camera times')
    profile = json.loads((scene/'source/sim/d435i_camera.json').read_text())
    head = json.loads((scene/'source/deploy/task_head_pose.json').read_text())
    settings = json.loads((scene/'camera_settings.json').read_text())
    data = mujoco.MjData(model)
    for name, angle in head['measured_joint_reference_rad'].items():
        data.qpos[model.joint(name).qposadr[0]] = angle
    mujoco.mj_forward(model, data)
    base, camera = model.body('base_link').id, model.camera('nav_camera').id
    to_body = data.xmat[base].reshape(3, 3).T
    position = to_body @ (data.cam_xpos[camera] - data.xpos[base])
    rotation = to_body @ data.cam_xmat[camera].reshape(3, 3) @ np.diag([1., -1., -1.])
    if (not np.allclose(position, settings['position_body_m'], atol=1e-8) or
            not np.allclose(rotation, settings['rotation_optical_to_body'], atol=1e-8)):
        raise ValueError('Archived projector differs from the model at the measured head reference')

    output.mkdir(parents=True, exist_ok=False)
    for name in ('scene.xml', 'camera_settings.json'):
        shutil.copy2(scene/name, output/name)
    shutil.copytree(scene/'source', output/'source')
    orig = [old_model.joint(name).qposadr[0] for name in JOINTS]
    dest = [model.joint(name).qposadr[0] for name in JOINTS]
    start = qpos[indices[0], :7]
    attitude = quaternion_matrix_xyzw(np.roll(start[3:7], -1))
    manifest = dict(validation_mode='prescribed body trajectory; no dynamics or referee',
                    scene_run=str(scene.resolve()), trajectory_run=str(trajectory.resolve()), seed=seed,
                    trajectory_states_sha256=hashlib.sha256((trajectory/'states.npz').read_bytes()).hexdigest(),
                    arguments=dict(spawn=[*start[:2], float(np.arctan2(attitude[1, 0], attitude[0, 0]))]),
                    navigation_core_sha256=core_hashes())
    for name in list(manifest['navigation_core_sha256']):
        shutil.copy2(ROOT/'navigation'/name, output/'source/navigation'/name)
    for name in ('sim/sensors.py', 'scripts/reproject_floor_lap.py', 'scripts/evaluate_floor_map.py'):
        shutil.copy2(ROOT/name, output/'source'/name)
    (output/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    rows, states = [], []
    rig = SensorRig(model, seed=seed, noise=True, camera_profile=profile)
    try:
        if not np.allclose(rig.intrinsics, settings['intrinsics']):
            raise ValueError('Recorded scene must use its archived D435i intrinsics')
        runtime = NavigationRuntime(rig.width, rig.height, rig.intrinsics, position, rotation)
        for i in indices:
            data.time = time[i]
            data.qpos[:7] = qpos[i, :7]
            data.qpos[dest] = qpos[i, orig]
            mujoco.mj_forward(model, data)
            quat = rig.proprioception(data).orientation_wxyz
            runtime.set_attitude(time[i], quaternion_matrix_xyzw(np.roll(quat, -1)))
            observation = runtime.observe(rig.camera(data))
            runtime.command(time[i])
            rows.append(dict(t=time[i], boxes=observation.obstacles.tolist(), **runtime.diagnostic()))
            states.append(data.qpos.copy())
    finally:
        rig.close()
    runtime.floor_map.save(output/'floor_map.npz')
    (output/'observations.json').write_text(json.dumps(rows))
    np.savez_compressed(output/'states.npz', qpos=states, time=time[indices])
    with (output/'telemetry.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['t', 'truth_x', 'truth_y'])
        writer.writerows(zip(time[indices], qpos[indices, 0], qpos[indices, 1]))
    return evaluate(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', required=True, type=Path)
    parser.add_argument('--scene', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(reproject(args.trajectory, args.scene, args.output, args.seed), indent=2))
