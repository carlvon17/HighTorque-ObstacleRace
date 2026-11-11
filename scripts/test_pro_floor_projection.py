"""Render the held Pro head at its measured standby attitude and test mapping."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from navigation.runtime import NavigationRuntime
from scripts.run_visual import visual_world
from sim.robot import Robot
from sim.sensors import SensorRig


class ProFloorProjectionTests(unittest.TestCase):
    def test_measured_head_view_maps_forward_ground_motion(self):
        profile = json.loads((ROOT/'sim/d435i_camera.json').read_text())
        mount = json.loads((ROOT/'sim/pro_camera_mount.json').read_text())
        head = json.loads((ROOT/'deploy/task_head_pose.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            model, _ = visual_world(Path(directory), {'position': [.55, -1.25]}, camera_profile=profile)
            rig = SensorRig(model, noise=False, camera_profile=profile)
            try:
                shadow = json.loads((ROOT/'deploy/pro_config.json').read_text())
                self.assertEqual(shadow['head_reference_rad'], head['measured_joint_reference_rad'])
                np.testing.assert_allclose(rig.camera_position, shadow['camera_position_body_m'], atol=1e-8)
                np.testing.assert_allclose(rig.camera_rotation, shadow['camera_rotation_optical_to_body'], atol=1e-8)
                robot = Robot(model, [.55, -1.25, 0.])
                data = robot.data
                body = Rotation.from_euler('xyz', [*mount['standby_base_roll_pitch_deg'], 0.], degrees=True)
                data.qpos[2] = .33
                data.qpos[3:7] = np.roll(body.as_quat(), 1)
                runtime = NavigationRuntime(rig.width, rig.height, rig.intrinsics, rig.camera_position, rig.camera_rotation)
                for i in range(18):
                    data.time = i / 15.
                    data.qpos[0] = .55 + .012*i
                    mujoco.mj_forward(model, data)
                    runtime.set_attitude(data.time, body.as_matrix())
                    runtime.observe(rig.camera(data))
                    runtime.command(data.time)
                camera = model.camera('nav_camera').id
                self.assertEqual(model.body(model.cam_bodyid[camera]).name, 'camera_link')
                for name, target in head['measured_joint_reference_rad'].items():
                    self.assertAlmostEqual(float(data.qpos[model.joint(name).qposadr[0]]), target)
                rendered = data.cam_xmat[camera].reshape(3, 3) @ np.diag([1., -1., -1.])
                self.assertGreater(rendered[0, 2], .5)  # forward ground, optical right points to body right
                self.assertLess(rendered[1, 0], -.9)
                self.assertAlmostEqual(np.degrees(np.arcsin(-rendered[2, 2])), head['camera_downward_angle_deg'], delta=.25)
                np.testing.assert_allclose(body.as_matrix() @ rig.camera_rotation, rendered, atol=1e-8)
                self.assertGreaterEqual(len(runtime.floor_map.keys), 2)
                self.assertGreater(runtime.floor_map.local_updates, 0)
                np.testing.assert_allclose(runtime.diagnostic()['map_xy'], [.012*17, 0], atol=.01)
            finally:
                rig.close()
