from pathlib import Path
import json,sys,unittest
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from deploy.line_follow_core import observe_lane,choose_command,sample_lane_boundaries
VIEW=ROOT/'output/deployment/head_restore_20261110/verified_after_battery/runs/head_restore_after_battery/view'


class LineTrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rgb=cv2.cvtColor(cv2.imread(str(VIEW/'latest_rgb.jpg')),cv2.COLOR_BGR2RGB)
        cls.depth=np.load(VIEW/'latest_depth_m.npy');cls.camera=json.loads((VIEW/'camera_intrinsics.json').read_text())

    def test_real_depth_and_white_boundaries_produce_bounded_command(self):
        r=observe_lane(self.rgb,self.depth,self.camera)
        self.assertTrue(.42<r['lane_width_m']<.6)
        self.assertLessEqual(abs(r['command'][2]),.04)
        self.assertEqual(r['command'][0],.04)

    def test_missing_boundary_fails_closed(self):
        rgb=self.rgb.copy();rgb[:,:rgb.shape[1]//2]=[40,110,65]
        with self.assertRaises(ValueError):observe_lane(rgb,self.depth,self.camera)

    def test_recorded_bend_uses_observed_sections_instead_of_far_extrapolation(self):
        view=ROOT/'output/deployment/line_trial_20261110/actual14_bundle/vision'
        camera=json.loads((view/'camera.json').read_text())
        camera.update(fx=camera['fx']/2,fy=camera['fy']/2,cx=(camera['cx']+.5)/2-.5,cy=(camera['cy']+.5)/2-.5)
        rgb=cv2.cvtColor(cv2.imread(str(view/'latest_rgb.jpg')),cv2.COLOR_BGR2RGB)
        result=observe_lane(rgb,np.load(view/'latest_depth_m.npy'),camera,.4)
        self.assertEqual(result['fit_method'],'sampled_boundaries')
        self.assertLess(abs(result['heading_rad']),np.deg2rad(10))
        self.assertGreaterEqual(len(result['path_samples_ground_m']),6)
        self.assertTrue(.45<result['lane_width_m']<.55)

    def test_sampled_lane_preserves_a_tape_corner_and_rejects_a_narrow_pair(self):
        x=np.linspace(.31,.89,300)
        y=.6*np.maximum(x-.52,0.)
        clouds=[np.column_stack((x,y+offset)) for offset in (-.25,.25)]
        model=sample_lane_boundaries(clouds)
        self.assertLess(abs(model['offset']),.01)
        self.assertGreater(model['path'][-1,1],.1)
        with self.assertRaises(ValueError):sample_lane_boundaries([clouds[0],clouds[0]+[0,.2]])

    def test_stale_navigation_and_manual_priority(self):
        def decide(**extra):
            args=dict(now=10.,manual=[-.1,0.,.1],nav=[.5,.5,1.],manual_active=False,joy_age=.01,nav_age=.01,trial_elapsed=1.,allowed=True)
            args.update(extra);return choose_command(**args)
        self.assertEqual(decide()[0],[.04,0.,.04])
        self.assertEqual(decide(nav_age=.3)[0],[0.,0.,0.])
        self.assertEqual(decide(joy_age=.3)[0],[0.,0.,0.])
        self.assertEqual(decide(trial_elapsed=6.)[0],[0.,0.,0.])
        self.assertEqual(decide(manual_active=True)[1],'manual')
        self.assertEqual(decide(allowed=False,manual_active=True)[0],[0.,0.,0.])
        self.assertEqual(decide(manual_active=True,manual=[float('nan'),0.,0.])[0],[0.,0.,0.])
        self.assertEqual(decide(speed_limit=.4)[0],[.4,0.,.04])
        self.assertEqual(decide(speed_limit=.41)[0],[0.,0.,0.])
        self.assertEqual(decide(speed_limit=.4,yaw_limit=.5)[0],[.4,0.,.5])
        self.assertEqual(decide(speed_limit=.4,yaw_limit=.5,duration=30.,trial_elapsed=30.)[0],[0.,0.,0.])

    def test_left_and_right_curves_preserve_lane_width_and_steering_sign(self):
        camera=dict(fx=301.9,fy=301.9,cx=162.5,cy=122.5)
        v,u=np.indices((240,320));theta=np.deg2rad(38)
        rotation=np.array([[0,-np.sin(theta),np.cos(theta)],[-1,0,0],[0,-np.cos(theta),-np.sin(theta)]])
        directions=np.stack([(u-camera['cx'])/camera['fx'],(v-camera['cy'])/camera['fy'],np.ones_like(v)],-1)@rotation.T
        depth=-.6/directions[:,:,2];points=directions*depth[:,:,None]
        for radius in (1.25,-1.25,2.,-2.):
            rgb=np.full((240,320,3),[65,140,85],dtype=np.uint8)
            distance=np.sqrt(points[:,:,0]**2+(points[:,:,1]-radius)**2)
            white=(abs(distance-(abs(radius)-.25))<.006)|(abs(distance-(abs(radius)+.25))<.006)
            rgb[white]=240
            result=observe_lane(rgb,depth,camera,.4)
            self.assertLess(abs(result['lane_width_m']-.5),.05)
            self.assertGreater(result['command'][2]*radius,0)
            self.assertLessEqual(abs(result['command'][2]),.5)


if __name__=='__main__':unittest.main()
