"""Compare the Pro camera in the calibration attitude and in free-standing AMP."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.ground_calibration import fit_ground
from scripts.run_visual import visual_world
from sim.navigation import Track
from sim.robot import HEAD_JOINTS, Robot
from sim.sensors import SensorRig


def measure(robot, rig, directory, name):
    frame = rig.camera(robot.data)
    rgb = frame.rgb.astype(float)
    v, u = np.indices(frame.depth.shape)
    fx, fy, cx, cy = rig.intrinsics
    floor = (np.isfinite(frame.depth) & (rgb[:,:,1] > 1.25*rgb[:,:,0])
             & (rgb[:,:,1] > 1.10*rgb[:,:,2]) & (v > rig.height*.25))
    rays = np.stack([(u-cx)/fx, (v-cy)/fy, np.ones_like(u)], axis=-1)
    plane = fit_ground(rays[floor]*frame.depth[floor, None])
    Image.fromarray(frame.rgb).save(directory/(name+'.png'))
    camera = robot.data.camera('nav_camera')
    optical_angle = np.degrees(np.arcsin(camera.xmat.reshape(3, 3)[2, 2]))
    return dict(kinematic_angle_deg=float(optical_angle), depth_angle_deg=plane['downward_angle_deg'],
                depth_height_m=plane['camera_height_m'], depth_fit_rms_m=plane['residual_rms_m'],
                head_rad={j:float(robot.data.joint(j).qpos[0]) for j in HEAD_JOINTS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    profile = json.loads((ROOT/'sim/d435i_camera.json').read_text())
    mount = json.loads((ROOT/'sim/pro_camera_mount.json').read_text())
    task = json.loads((ROOT/'deploy/task_head_pose.json').read_text())
    start = next(s for s in Track().config['starts'] if s['id'] == '01')
    model, _ = visual_world(args.output, start, camera_profile=profile)
    rig = SensorRig(model, noise=False, camera_profile=profile)
    pose = [*start['position'], np.deg2rad(start['yaw_degrees'])]
    try:
        robot = Robot(model, pose, from_standby=True)
        roll, pitch = mount['standby_base_roll_pitch_deg']
        matched = measure(robot, rig, args.output, 'standby_reference')
        for _ in range(round(2/robot.dt)):
            robot.step([0,0,0])
        standing = measure(robot, rig, args.output, 'amp_zero_speed')
        result = dict(reference_deg=task['camera_downward_angle_deg'],
                      reference_base_roll_pitch_deg=[roll, pitch],
                      matched_static_attitude=matched, amp_zero_after_2s=standing,
                      mount_assumptions=mount['assumptions'])
        (args.output/'standing.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
        if abs(matched['depth_angle_deg']-task['camera_downward_angle_deg']) > .5:
            raise ValueError('Rendered standby depth did not reproduce the ground-plane calibration')
    finally:
        rig.close()


if __name__ == '__main__':
    main()
