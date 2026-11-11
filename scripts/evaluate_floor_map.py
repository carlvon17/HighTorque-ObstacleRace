"""Score estimated poses after the run and render the measured floor atlas."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def camera_alignment(run):
    """Audit projection inputs against saved model kinematics after control."""
    with np.load(run/'floor_map.npz', allow_pickle=False) as atlas:
        if 'camera_position_body_m' not in atlas:
            return {'status': 'legacy_map_without_projection_metadata'}
        origin = atlas['forward_left_origin_m'].tolist()
        times = atlas['time']
        positions = atlas['camera_position_body_m']
        rotations = atlas['camera_rotation_optical_to_body']
        intrinsics = atlas['intrinsics']
    import mujoco

    settings = json.loads((run/'camera_settings.json').read_text())
    model = mujoco.MjModel.from_xml_path(str((run/'scene.xml').resolve()))
    data = mujoco.MjData(model)
    base = model.body('base_link').id
    camera = model.camera('nav_camera').id
    optical_axes = np.diag([1., -1., -1.])
    expected_intrinsics = settings['intrinsics']
    if expected_intrinsics is None:
        width, height = settings['processing_resolution']
        focal_length = height / (2 * np.tan(np.deg2rad(model.cam_fovy[camera]) / 2))
        expected_intrinsics = [focal_length, focal_length, (width-1)/2, (height-1)/2]
    result = {'camera_parent': model.body(model.cam_bodyid[camera]).name,
              'keyframes': len(times), 'patch_forward_left_origin_m': origin,
              'intrinsics_match_run': bool(np.allclose(intrinsics, expected_intrinsics, rtol=0, atol=1e-9))}
    profile = run/'source/deploy/task_head_pose.json'
    mount = run/'source/sim/pro_camera_mount.json'
    if profile.exists() and mount.exists():
        head = json.loads(profile.read_text())
        reference = head['measured_joint_reference_rad']
        for name, angle in reference.items():
            data.qpos[model.joint(name).qposadr[0]] = angle
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        optical_forward = -data.cam_xmat[camera].reshape(3, 3)[:, 2]
        ray = data.xmat[base].reshape(3, 3).T @ optical_forward
        roll, pitch = np.deg2rad(json.loads(mount.read_text())['standby_base_roll_pitch_deg'])
        up_body = np.array([-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)])
        result.update(reference_joint_rad=reference, reference_optical_forward_body=ray.tolist(), measured_standby_down_deg=head['camera_downward_angle_deg'],
                      modeled_at_measured_standby_down_deg=float(np.rad2deg(np.arcsin(np.clip(-ray@up_body, -1., 1.)))))
    if not len(times):
        return dict(result, status='no_keyframes')
    with np.load(run/'states.npz', allow_pickle=False) as states:
        state_time, qpos = states['time'], states['qpos']
    indices = np.searchsorted(state_time, times)
    if (indices >= len(state_time)).any() or not np.allclose(state_time[indices], times, atol=1e-6, rtol=0):
        raise ValueError('Saved model states do not cover the map keyframe timestamps')
    position_error, angle_error = [], []
    for index, position, rotation in zip(indices, positions, rotations):
        data.qpos[:] = qpos[index]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        to_body = data.xmat[base].reshape(3, 3).T
        body_position = to_body @ (data.cam_xpos[camera]-data.xpos[base])
        body_rotation = to_body @ data.cam_xmat[camera].reshape(3, 3) @ optical_axes
        position_error.append(np.linalg.norm(position-body_position))
        cosine = (np.trace(rotation.T@body_rotation)-1.)/2.
        angle_error.append(np.rad2deg(np.arccos(np.clip(cosine, -1., 1.))))
    return dict(result, status='compared_at_keyframe_times',
                max_projection_position_error_m=float(max(position_error)),
                max_projection_rotation_error_deg=float(max(angle_error)))


def render_map(run, observations):
    # Only measured keyframes are used here, including their estimated poses.
    with np.load(run/'floor_map.npz', allow_pickle=False) as data:
        if not len(data['time']):
            return
        h, w = data['patch'].shape[1:]
        origin = data['forward_left_origin_m']
        resolution = float(data['resolution_m'])
        forward = origin[0] + np.arange(h) * resolution
        left = origin[1] + np.arange(w) * resolution
        xy = data['map_xy']
        low, high = xy.min(axis=0)-1., xy.max(axis=0)+1.
        scale = 140.
        width, height = np.ceil((high-low)*scale).astype(int)+1
        canvas = np.full((height, width, 3), 32, np.uint8)
        for center, angle, patch in zip(xy, data['heading'], data['patch']):
            c, s = np.cos(angle), np.sin(angle)
            x = center[0]+c*forward[:, None]-s*left
            y = center[1]+s*forward[:, None]+c*left
            col = np.rint((x-low[0])*scale).astype(int)
            row = np.rint((high[1]-y)*scale).astype(int)
            gray = np.clip(140+60*patch, 50, 235).astype(np.uint8)
            canvas[row, col] = gray[..., None]
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        for obs in observations:
            if obs['map_xy'] is None:
                continue
            yaw = obs['pose'][2]
            c, s = np.cos(yaw), np.sin(yaw)
            for forward, left, _, _ in obs['boxes']:
                x = obs['map_xy'][0]+c*forward-s*left
                y = obs['map_xy'][1]+s*forward+c*left
                col, row = (x-low[0])*scale, (high[1]-y)*scale
                draw.ellipse((col-1, row-1, col+1, row+1), fill='#ffa64d')
        for x, y in xy:
            col, row = (x-low[0])*scale, (high[1]-y)*scale
            draw.ellipse((col-2, row-2, col+2, row+2), fill='#48bde6')
        draw.line((18, height-22, 18+scale, height-22), fill='white', width=2)
        draw.text((18, height-40), '1 m', fill='white')
        draw.text((18, 14), 'RGB-D floor texture | blue: keyframes | orange: tracked box centers', fill='white')
        image.save(run/'floor_map.png')


def evaluate(run):
    rows = json.loads((run/'observations.json').read_text())
    samples = np.genfromtxt(run/'telemetry.csv', delimiter=',', names=True)
    # The simulator records truth at the END of each physics interval. Align
    # it to the camera timestamp and express it in the initial body frame.
    manifest = json.loads((run/'run_manifest.json').read_text())
    spawn = manifest['arguments']['spawn']
    if spawn is None:
        field = json.loads((run/'source/scene_config.json').read_text())
        start = next(s for s in field['starts'] if s['id'] == manifest['arguments']['start'])
        spawn = [*start['position'], np.deg2rad(start['yaw_degrees'])]
    t = np.array([r['t'] for r in rows])
    world = np.column_stack([np.interp(t, samples['t'], samples[key], left=value)
                             for key, value in zip(('truth_x', 'truth_y'), spawn)])
    c, s = np.cos(spawn[2]), np.sin(spawn[2])
    relative = (world-np.array(spawn[:2])) @ np.array([[c, -s], [s, c]])
    report = dict(camera_alignment=camera_alignment(run), camera_frames=len(t),
                  accepted_odometry_frames=sum(r['vo_accepted'] for r in rows), keyframes=rows[-1]['floor_keyframes'],
                  confirmed_revisit_updates=rows[-1]['map_revisits'],
                  local_pose_recoveries=rows[-1]['map_recoveries'],
                  local_map_updates=rows[-1].get('map_tracking_updates', 0),
                  segment_reseeds=rows[-1]['localization_reseeds'])
    near_box = np.array([any(-.4 < x < .85 and abs(y) < .8 for x, y, _, _ in r['boxes'])
                         for r in rows])
    report['near_tracked_box_frames'] = int(near_box.sum())
    confirmed = [r['t'] for r in rows if r['map_revisits']]
    report['first_confirmed_revisit_s'] = confirmed[0] if confirmed else None
    for key in ('pose', 'map_xy'):
        valid = np.array([r[key] is not None for r in rows])
        if not valid.any():
            report[key] = dict(valid_frames=0)
            continue
        estimate = np.array([r[key][:2] for r in rows if r[key] is not None])
        error = np.linalg.norm(estimate-relative[valid], axis=1)
        box_error = error[near_box[valid]]
        report[key] = dict(valid_frames=int(valid.sum()),
                           near_tracked_box_rmse_m=float(np.sqrt(np.mean(box_error**2))) if len(box_error) else None,
                           rmse_m=float(np.sqrt(np.mean(error**2))),
                           p95_m=float(np.percentile(error, 95)), max_m=float(error.max()),
                           final_error_m=float(error[-1]) if valid[-1] else None)
    (run/'mapping_evaluation.json').write_text(json.dumps(report, indent=2))
    render_map(run, rows)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    print(json.dumps(evaluate(parser.parse_args().run), indent=2))
