"""Regression checks for physical integration and race termination logic."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import unittest
import mujoco
import numpy as np
from sim.robot import Robot, build_model, JOINTS, MJ_TO_ISAAC, ISAAC_TO_MJ
from sim.navigation import Track
from sim.scoring import Referee


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _ = build_model()
        cls.track = Track()

    def robot_at(self, x, y):
        return Robot(self.model, [x, y, 0])

    def test_policy_joint_contract_and_physical_motion(self):
        robot = self.robot_at(.55, -1.25)
        np.testing.assert_array_equal(MJ_TO_ISAAC[ISAAC_TO_MJ], np.arange(20))
        for name, actuator in zip(JOINTS, robot.aids):
            self.assertEqual(self.model.actuator_trnid[actuator, 0], self.model.joint(name).id)
        for _ in range(100):
            robot.step([0, 0, 0])
        start = robot.pose.copy()
        for _ in range(150):
            robot.step([.4, 0, 0])
        self.assertGreater(robot.pose[0]-start[0], .35)
        self.assertGreater(robot.data.qpos[2], .3)
        self.assertTrue(np.isfinite(robot.data.qpos).all())

    def test_single_foot_overhang_does_not_equal_both_feet_out(self):
        for y, expected in [(-1.25, None), (-1.50, None), (-1.85, 'both_feet_outside')]:
            robot = self.robot_at(.55, y)
            referee = Referee(robot, self.track)
            referee.boundary()
            self.assertEqual(referee.reason, expected)

    def test_obstacle_contact_is_terminal(self):
        robot = self.robot_at(1.65, 1.10)
        referee = Referee(robot, self.track)
        referee.contacts()
        self.assertEqual(referee.reason, 'obstacle_contact')

    def test_fall_is_terminal(self):
        robot = self.robot_at(.55, -1.25)
        robot.data.qpos[2] = .15
        mujoco.mj_forward(self.model, robot.data)
        referee = Referee(robot, self.track)
        referee.contacts()
        self.assertIn(referee.reason, ('fall_posture', 'fall_ground_contact'))

    def test_footprint_route_rejects_impossible_width(self):
        with self.assertRaises(ValueError):
            self.track.plan(radius=.3, full_body=False)


if __name__ == '__main__':
    unittest.main()
