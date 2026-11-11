"""Floor-calibration geometry checks independent of ROS and physical sensors."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deploy.ground_calibration import fit_ground


class GroundCalibrationTests(unittest.TestCase):
    def cloud(self):
        rng=np.random.default_rng(4)
        n=np.array([.02,-.72,-.69]);n/=np.linalg.norm(n)
        xy=rng.uniform([-.6,.2],[.6,1.8],(2000,2));height=.82
        points=np.c_[xy,(-height-xy@n[:2])/n[2]]
        points+=rng.normal(0,.001,points.shape)
        points=np.r_[points,rng.uniform(-2,2,(300,3))]
        return points,n,height

    def test_metric_height_normal_and_outlier_rejection(self):
        points,n,height=self.cloud();result=fit_ground(points)
        self.assertAlmostEqual(result['camera_height_m'],height,delta=.005)
        self.assertGreater(np.dot(result['normal_camera'],n),.9999)
        self.assertLess(result['residual_rms_m'],.003)

    def test_projection_places_ground_on_zero_height(self):
        points,n,height=self.cloud();result=fit_ground(points)
        r=np.array(result['camera_to_ground_rotation'])
        np.testing.assert_allclose(r@r.T,np.eye(3),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(r),1.)
        good=points[np.abs(points@n+height)<.004]
        mapped=good@r.T+np.array(result['camera_position_ground_m'])
        self.assertLess(np.sqrt(np.mean(mapped[:,2]**2)),.003)

    def test_floor_fit_cannot_claim_body_extrinsics(self):
        result=fit_ground(self.cloud()[0])
        self.assertFalse(result['body_extrinsics_verified'])
        self.assertIn('camera/body yaw',result['unobservable'])

    def test_missing_nonplanar_and_degenerate_data_fail(self):
        rng=np.random.default_rng(10)
        for data in [np.zeros((100,3)),np.full((500,3),np.nan),np.zeros((500,3)),rng.uniform(-2,2,(1000,3))]:
            with self.subTest(shape=data.shape):
                with self.assertRaises(ValueError):fit_ground(data)


if __name__=='__main__':unittest.main()
