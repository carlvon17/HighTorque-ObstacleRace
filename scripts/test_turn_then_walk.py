"""Regressions from observed torso sway and delayed visual turn completion."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deploy.line_follow_core import TurnThenWalk


def observation(bearing_degrees):
    return {'center_line':[0.,0.,.55*math.tan(math.radians(bearing_degrees))]}


class TurnFeedbackTests(unittest.TestCase):
    def test_camera_alignment_alone_cannot_finish_physical_turn(self):
        f=TurnThenWalk()
        self.assertEqual(f.command(observation(10),.4,0.)[0],[0.,0.,1.])
        for t in (.2,.4,.6):
            self.assertEqual(f.command(observation(0),.4,t)[0],[0.,0.,1.])

    def test_imu_completion_stops_turn_then_resumes_forward_after_settling(self):
        f=TurnThenWalk();f.command(observation(10),.4,0.)
        self.assertEqual(f.command(observation(1),.4,1.,turn_completed=True)[0],[0.,0.,0.])
        self.assertEqual(f.command(observation(1),.4,1.4)[0],[0.,0.,0.])
        self.assertEqual(f.command(observation(1),.4,1.6)[0],[.4,0.,0.])

    def test_remaining_error_requests_another_bounded_pulse(self):
        f=TurnThenWalk();f.command(observation(-15),.4,0.)
        f.command(observation(-10),.4,1.,turn_completed=True)
        self.assertEqual(f.command(observation(-10),.4,1.4)[0],[0.,0.,0.])
        self.assertEqual(f.command(observation(-10),.4,1.6)[0],[0.,0.,-1.])


if __name__=='__main__':unittest.main()
