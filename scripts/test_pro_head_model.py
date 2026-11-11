"""Cross-check the supplied URDF head chain against its independent MJCF."""
from pathlib import Path
import sys
import unittest
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.pro_head_model import ProHeadModel

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
