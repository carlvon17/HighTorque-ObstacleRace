"""Cross-check the supplied URDF head chain against its independent MJCF."""
import json
from pathlib import Path
import sys
import unittest
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.pro_head_model import ProHeadModel
from sim.robot import build_model

ASSET = ROOT/'reference/robot/PiPlusPro_SF_12L8A0G2H0W_Z_261010'


class ProHeadModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.head = ProHeadModel(next((ASSET/'urdf').glob('*261010.urdf')))

    def test_matches_supplied_mjcf_at_multiple_head_angles(self):
        model = mujoco.MjModel.from_xml_path(str(next((ASSET/'xml').glob('*.xml'))))
        data = mujoco.MjData(model)
        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'camera_link')
        rng = np.random.default_rng(11)
        for _ in range(100):
            angles = {n: float(rng.uniform(lo+.05, hi-.05))
                      for n, (lo, hi) in self.head.limits.items()}
            for name, q in angles.items():
                j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                data.qpos[model.jnt_qposadr[j]] = q
            mujoco.mj_forward(model, data)
            rb = data.xmat[base].reshape(3, 3)
            actual = np.eye(4)
            actual[:3, :3] = rb.T@data.xmat[camera].reshape(3, 3)
            actual[:3, 3] = rb.T@(data.xpos[camera]-data.xpos[base])
            np.testing.assert_allclose(self.head.camera_link_transform(angles), actual, atol=1e-10)

    def test_race_scene_starts_at_feedback_and_preserves_absolute_joint_angles(self):
        model, _ = build_model(no_obstacles=True)
        data = mujoco.MjData(model)
        reference = json.loads((ROOT/'deploy/task_head_pose.json').read_text())['measured_joint_reference_rad']
        camera = model.camera('nav_camera').id
        link = model.body('camera_link').id
        self.assertEqual(model.cam_bodyid[camera], link)
        for name, angle in reference.items():
            self.assertAlmostEqual(float(data.joint(name).qpos[0]), angle)
        for yaw, pitch in ((0., 0.), (.3, -.2), (0., reference['head_pitch_joint'])):
            angles = {'head_pitch_joint': pitch, 'head_yaw_joint': yaw}
            for name, angle in angles.items():
                data.joint(name).qpos[0] = angle
            mujoco.mj_forward(model, data)
            expected = self.head.camera_link_transform(angles)
            base = data.body('base_link')
            rb = base.xmat.reshape(3, 3)
            np.testing.assert_allclose(rb.T @ (data.xpos[link]-base.xpos), expected[:3, 3], atol=1e-10)
            np.testing.assert_allclose(rb.T @ data.xmat[link].reshape(3, 3), expected[:3, :3], atol=1e-10)

    def test_rejects_observed_out_of_model_range(self):
        with self.assertRaises(ValueError):
            self.head.camera_link_transform({'head_yaw_joint': 0., 'head_pitch_joint': 1.095787525})
        with self.assertRaises(ValueError):
            self.head.check_angles({'head_yaw_joint': 0., 'head_pitch_joint': float('nan')})

    def test_zero_angle_mechanical_mount_position(self):
        transform = self.head.camera_link_transform({'head_yaw_joint': 0., 'head_pitch_joint': 0.})
        np.testing.assert_allclose(transform[:3, 3], [.047642, .031234, .297392], atol=1e-10)


if __name__ == '__main__':
    unittest.main()
