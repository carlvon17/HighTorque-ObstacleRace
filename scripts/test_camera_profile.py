"""Validate real-camera projection against rendered geometry, not just config."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from dataclasses import replace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from scripts.run_visual import visual_world
from sim.robot import Robot
from sim.sensors import SensorRig
from sim.visual_navigation import CameraPerception,VisualNavigator


class CameraProfileTests(unittest.TestCase):
    def test_inner_slats_do_not_create_duplicate_boxes_or_shift_centers(self):
        profile=json.loads((ROOT/'sim/d435i_camera.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            model,_=visual_world(Path(directory),{'position':[.55,-1.25]},camera_profile=profile)
            rig=SensorRig(model,noise=False,camera_profile=profile)
            try:
                per=CameraPerception(rig.width,rig.height,rig.fovy,rig.camera_position,
                                     rig.camera_rotation,intrinsics=rig.intrinsics)
                sequence=((2.5,1.25),(2.2,1.43),(1.98,1.43),(1.70,1.43),
                          (1.4,1.15),(1.3,1.07),(1.2,1.07),(1.1,1.07),(.95,1.07))
                for i,(x,y) in enumerate(sequence):
                    robot=Robot(model,[x,y,np.pi]);robot.data.time=i*.1
                    robot.data.qpos[2]=.33;mujoco.mj_forward(model,robot.data)
                    observation=per.update(rig.camera(robot.data),np.array([2.5-x,1.25-y,0]),np.eye(3))
                    if i>=4:
                        boxes=observation.obstacles[np.argsort(observation.obstacles[:,0])]
                        self.assertEqual(len(boxes),3)
                        np.testing.assert_allclose(boxes[:,0],[x-1.65,x-1.15,x-.65],atol=.012)
            finally:
                rig.close()

    def test_visible_face_corrects_hidden_box_for_odometry_drift(self):
        profile=json.loads((ROOT/'sim/d435i_camera.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            model,_=visual_world(Path(directory),{'position':[.55,-1.25]},camera_profile=profile)
            rig=SensorRig(model,noise=False,camera_profile=profile)
            try:
                per=CameraPerception(rig.width,rig.height,rig.fovy,rig.camera_position,
                                     rig.camera_rotation,intrinsics=rig.intrinsics)
                sequence=((2.5,1.25,0),(2.2,1.43,0),(1.98,1.43,0),
                          (1.85,1.43,-.01),(1.70,1.43,-.025),(1.53,1.43,-.04))
                for i,(x,y,drift) in enumerate(sequence):
                    robot=Robot(model,[x,y,np.pi]);robot.data.time=i*.1
                    frame=rig.camera(robot.data)
                    observation=per.update(frame,np.array([2.5-x+drift,1.25-y,0]),np.eye(3))
                    boxes=observation.obstacles[np.argsort(observation.obstacles[:,0])]
                    if i>=2:
                        self.assertAlmostEqual(boxes[1,0]-boxes[0,0],.50,delta=.012)
                        self.assertAlmostEqual(boxes[0,0],x-1.65,delta=.012)
            finally:
                rig.close()

    def test_clipped_faces_recover_centers_without_assuming_obstacle_spacing(self):
        profile=json.loads((ROOT/'sim/d435i_camera.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            model,_=visual_world(Path(directory),{'position':[.55,-1.25]},camera_profile=profile)
            # Isolate the two-box gap. The Pro optics can also see the third
            # box from the first sample in this sequence.
            model.body('hb_obstacle_01').pos[1] = 4.
            rig=SensorRig(model,noise=False,camera_profile=profile)
            try:
                for separation in (.50,.57,.44):
                    with self.subTest(separation=separation):
                        model.body('hb_obstacle_02').pos[0]=1.65-separation
                        per=CameraPerception(rig.width,rig.height,rig.fovy,rig.camera_position,
                                             rig.camera_rotation,intrinsics=rig.intrinsics)
                        for i,(x,y) in enumerate(((2.5,1.25),(2.2,1.43),(1.98,1.43))):
                            robot=Robot(model,[x,y,np.pi]);robot.data.time=i*.1
                            frame=rig.camera(robot.data)
                            # Controlled odometry input: moving forward and sideways
                            # while the near box becomes clipped by the image edge.
                            observation=per.update(frame,np.array([2.5-x,1.25-y,0]),np.eye(3))
                            nearest=min(observation.obstacles,key=lambda b:b[0])
                            self.assertAlmostEqual(nearest[0],x-1.65,delta=.012)
                        boxes=observation.obstacles[np.argsort(observation.obstacles[:,0])]
                        self.assertEqual(len(boxes),2)
                        self.assertAlmostEqual(boxes[1,0]-boxes[0,0],separation,delta=.012)
                        plan=VisualNavigator().plan(replace(observation,lane=np.array([0.,0.,.18])))
                        self.assertEqual(plan is not None,separation>.47)
            finally:
                rig.close()

    def test_single_rotated_face_measures_center_normal(self):
        per=CameraPerception(320,240,43.35,np.zeros(3),np.eye(3))
        rng=np.random.default_rng(1)
        for angle in (-.4,.35):
            normal=np.array([np.cos(angle),np.sin(angle)])
            tangent=np.array([-normal[1],normal[0]])
            center=np.array([.8,.1])
            horizontal=np.tile(np.linspace(-.06,.07,40),30)
            cloud=center-.1*normal+horizontal[:,None]*tangent+rng.normal(0,.0007,(1200,2))
            heights=np.repeat(np.linspace(.06,.37,30),40)
            fitted=per.box_surfaces(cloud,heights,np.zeros(2))
            self.assertIsNotNone(fitted)
            _,_,normals,offsets,_,lower,upper=fitted
            self.assertEqual(len(normals),1)
            self.assertAlmostEqual(normals[0]@center,offsets[0],delta=.003)
            self.assertGreater(upper-lower,.05)

    def test_real_camera_ground_projection_and_lane_offset(self):
        profile=json.loads((ROOT/'sim/d435i_camera.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            model,_=visual_world(Path(directory),{'position':[.55,-1.25]},True,camera_profile=profile)
            rig=SensorRig(model,noise=False,camera_profile=profile)
            try:
                for y in (-1.25,-1.15):
                    robot=Robot(model,[.55,y,0])
                    frame=rig.camera(robot.data)
                    self.assertEqual(frame.rgb.shape,(240,320,3))
                    self.assertEqual(frame.depth.shape,(240,320))
                    per=CameraPerception(rig.width,rig.height,rig.fovy,rig.camera_position,
                                         rig.camera_rotation,intrinsics=rig.intrinsics)
                    camera_points=per.rays*frame.depth[...,None]
                    body=camera_points@rig.camera_rotation.T+rig.camera_position
                    rgb=frame.rgb.astype(float)
                    ground=(rgb[:,:,1]>1.4*rgb[:,:,0])&(rgb[:,:,1]>1.12*rgb[:,:,2])&np.isfinite(frame.depth)
                    # World floor is z=0; body is initially unrotated at qpos[2].
                    self.assertGreater(ground.sum(),1000)
                    self.assertLess(np.percentile(abs(body[:,:,2][ground]+robot.data.qpos[2]),95),.004)
                    observation=per.update(frame,np.zeros(3),np.eye(3))
                    self.assertIsNotNone(observation.lane)
                    self.assertAlmostEqual(np.polyval(observation.lane,0),-1.25-y,delta=.03)
            finally:
                rig.close()

    def test_rendered_optical_axis_and_off_axis_point_match_factory_intrinsics(self):
        profile=json.loads((ROOT/'sim/d435i_camera.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            _,xml=visual_world(Path(directory),{'position':[.55,-1.25]},True,camera_profile=profile)
            import xml.etree.ElementTree as ET
            camera=ET.fromstring(xml).find(".//camera[@name='nav_camera']")
            camera.attrib.pop('pos');camera.attrib.pop('xyaxes')
            scene=ET.Element('mujoco');visual=ET.SubElement(scene,'visual')
            ET.SubElement(visual,'global',offwidth='640',offheight='480')
            world=ET.SubElement(scene,'worldbody');world.append(camera)
            for x,y in ((0,0),(.2,.1)):
                ET.SubElement(world,'geom',type='sphere',size='.008',pos=f'{x} {-y} -1')
            model=mujoco.MjModel.from_xml_string(ET.tostring(scene,encoding='unicode'))
            data=mujoco.MjData(model);mujoco.mj_forward(model,data)
            renderer=mujoco.Renderer(model,480,640)
            try:
                renderer.enable_segmentation_rendering();renderer.update_scene(data,camera='nav_camera')
                segmentation=renderer.render()
                for i,(x,y) in enumerate(((0,0),(.2,.1))):
                    v,u=np.where(segmentation[:,:,0]==i)
                    np.testing.assert_allclose([u.mean(),v.mean()],
                        [profile['fx']*x+profile['cx'],profile['fy']*y+profile['cy']],atol=.2)
            finally:
                renderer.close()


if __name__=='__main__':unittest.main()
