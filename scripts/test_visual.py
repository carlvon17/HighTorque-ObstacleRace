"""Sensor-boundary and image-driven navigation regression tests."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dataclasses import fields
import unittest
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from sim.robot import Robot,build_model
from sim.sensors import SensorRig,CameraFrame,Proprioception
from sim.visual_navigation import CameraPerception,VisualNavigator,Perception,RGBDOdometry


class VisualTests(unittest.TestCase):
    def test_real_intrinsics_use_separate_focal_lengths_and_principal_point(self):
        k=(400.,300.,123.,101.)
        per=CameraPerception(320,240,80,np.zeros(3),np.eye(3),intrinsics=k)
        np.testing.assert_allclose(per.rays[131,163],[.1,.1,1.])
        vo=RGBDOdometry(320,240,80,np.zeros(3),np.eye(3),intrinsics=k)
        points=vo.points(np.array([[163.,131.]]),np.full((240,320),2.),np.eye(3))
        np.testing.assert_allclose(points,[[.2,.2,2.]])

    @classmethod
    def setUpClass(cls):
        cls.model,_=build_model()
        cls.rig=SensorRig(cls.model,noise=False)

    @classmethod
    def tearDownClass(cls):cls.rig.close()

    def camera_observation(self,y):
        robot=Robot(self.model,[.55,y,0])
        q=robot.data.sensor('orientation').data
        attitude=Rotation.from_quat(np.r_[q[1:],q[0]]).as_matrix()
        r=self.rig
        perception=CameraPerception(r.width,r.height,r.fovy,r.camera_position,r.camera_rotation,intrinsics=r.intrinsics)
        return perception,perception.update(r.camera(robot.data),np.zeros(3),attitude)

    def test_measurement_packets_cannot_carry_world_pose_or_map(self):
        self.assertEqual({f.name for f in fields(CameraFrame)},{'time','rgb','depth'})
        self.assertEqual({f.name for f in fields(Proprioception)},{'time','joints','orientation_wxyz','gyro'})
        source='\n'.join(p.read_text() for p in (ROOT/'navigation').glob('*.py'))
        for forbidden in ('import mujoco','scene_config','geometry.json','robot.pose','qpos','geom_xpos'):
            self.assertNotIn(forbidden,source)

    def test_lane_offset_is_measured_from_pixels(self):
        _,center=self.camera_observation(-1.25)
        _,shifted=self.camera_observation(-1.15)
        self.assertIsNotNone(center.lane);self.assertIsNotNone(shifted.lane)
        self.assertAlmostEqual(np.polyval(center.lane,0),0,delta=.025)
        self.assertAlmostEqual(np.polyval(shifted.lane,0),-.10,delta=.025)

    def test_black_camera_has_no_ground_truth_fallback(self):
        per,o=self.camera_observation(-1.25)
        nav=VisualNavigator();nav.observe(o,np.zeros(3),0)
        for _ in range(30):nav.update(0,np.zeros(3))
        self.assertGreater(nav.command[0],.1)
        frame=CameraFrame(.06,np.zeros((self.rig.height,self.rig.width,3),np.uint8),np.full((self.rig.height,self.rig.width),np.nan))
        blank=per.update(frame,np.zeros(3),np.eye(3))
        self.assertIsNone(blank.lane)
        nav.observe(blank,np.zeros(3),0)
        for i in range(100):nav.update(.06+i*.02,np.zeros(3))
        np.testing.assert_allclose(nav.command,0,atol=1e-12)

    def test_current_lane_overrides_drifted_old_map(self):
        per,_=self.camera_observation(-1.25)
        x=np.linspace(0,1.4,300)
        wrong=np.vstack([np.column_stack([x,.3*x*x+y]) for y in (-.15,.35)])
        per.white_memory=[(10.,wrong,np.zeros(2))]
        robot=Robot(self.model,[.55,-1.25,0]);raw=self.rig.camera(robot.data)
        observation=per.update(CameraFrame(20.,raw.rgb,raw.depth),np.zeros(3),np.eye(3))
        self.assertIsNotNone(observation.lane)
        self.assertAlmostEqual(np.polyval(observation.lane,.5),0,delta=.025)

    def test_occluded_box_not_forgotten_while_alongside(self):
        per,_=self.camera_observation(-1.25)
        per.obstacle_memory=[{'center':np.array([.2,.1]),'size':np.array([.2,.2]),'axes':np.eye(2),'time':0.}]
        robot=Robot(self.model,[.55,-1.25,0]);frame=self.rig.camera(robot.data)
        frame=CameraFrame(20.,frame.rgb,frame.depth)
        o=per.update(frame,np.zeros(3),np.eye(3))
        self.assertTrue(any(np.linalg.norm(box[:2]-[.2,.1])<.05 for box in o.obstacles))

    def test_finish_requires_visual_stripe_after_lap_turn(self):
        nav=VisualNavigator()
        p=Perception(0,np.zeros(3),np.empty((0,2)),np.empty((0,4)),np.array([[.5,0.]]),.36,np.zeros((1,1,3)))
        nav.observe(p,np.zeros(3),0)
        self.assertIsNone(nav.finish_target)
        nav.observe(p,np.zeros(3),6.)
        self.assertIsNotNone(nav.finish_target)
        nav.update(0,np.array([.51,0,0]))
        self.assertTrue(nav.finished)

    def test_observed_straight_corridor_rejects_occlusion_heading_jump(self):
        nav=VisualNavigator()
        def observation(t,lane,boxes):
            return Perception(t,np.array(lane),np.empty((0,2)),np.asarray(boxes).reshape(-1,4),
                              np.empty((0,2)),.36,np.zeros((1,1,3)))
        for i in range(5):
            p=observation(i*.06,[0,0,0],[[.6,.15,.2,.2]])
            nav.observe(p,np.zeros(3),0)
        nav.observe(observation(.3,[.2,-.15,0],[[.6,.15,.2,.2]]),np.zeros(3),0)
        # The blocking box hands control to side-step avoidance, which holds
        # the observed corridor heading instead of steering on the jumped fit.
        self.assertAlmostEqual(nav.update(.3,np.zeros(3))[2],0)
        self.assertEqual(nav.mode,'cross')
        nav.observe(observation(.36,[.2,.15,0],[]),np.zeros(3),0)
        self.assertAlmostEqual(nav.update(.36,np.zeros(3))[2],0)
        self.assertEqual(nav.mode,'cross')

    def test_observation_lease_tracks_the_onboard_frame_period(self):
        # The Jetson produces a frame every 0.4--0.7 s. With the simulator's fixed
        # 0.35 s window every real image was already stale, so the navigator stayed
        # in lost_vision and the robot never left standing.
        def observation(t):
            return Perception(t,np.array([0.,0.,0.]),np.empty((0,2)),np.empty((0,4)),
                              np.empty((0,2)),.36,np.zeros((1,1,3)))
        slow=VisualNavigator(.22,1.0)
        slow.observe(observation(0.),np.zeros(3),0)
        slow.update(.6,np.zeros(3))
        self.assertNotEqual(slow.mode,'lost_vision')
        fast=VisualNavigator(.22,.35)
        fast.observe(observation(0.),np.zeros(3),0)
        fast.update(.6,np.zeros(3))
        self.assertEqual(fast.mode,'lost_vision')


if __name__=='__main__':unittest.main()
