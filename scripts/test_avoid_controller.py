"""Side-step avoidance regressions; no simulator pose or AMP modification."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from navigation.avoid_controller import SideStepController, blocking, rotation, signed_distance
from navigation.visual import CameraPerception, Perception, VisualNavigator


def perception(lane=np.zeros(3), obstacles=(), boundaries=(), lane_pair=False, t=0.):
    return Perception(float(t), np.asarray(lane), np.empty((0, 2)),
                      np.asarray(obstacles, dtype=float).reshape(-1, 4),
                      np.empty((0, 2)), .36, np.zeros((1, 1, 3), np.uint8),
                      None, None, True, tuple(boundaries), bool(lane_pair))


def straight_boundaries(offsets=(-.25, .25), slope=0.):
    return tuple((np.array([0., slope, offset]), (.3, 1.2)) for offset in offsets)


class BlockingTests(unittest.TestCase):
    def test_ahead_box_overlapping_lane_is_nearest_first(self):
        boxes = np.array([[1.3, .5, .2, .2], [.8, .15, .2, .2], [.2, .3, .2, .2]])
        np.testing.assert_allclose(blocking(boxes, np.zeros(3), .40), boxes[2])

    def test_box_beyond_trigger_range_or_lane_band_is_ignored(self):
        for box in ([1.05, .15, .2, .2], [-.1, .15, .2, .2], [.8, .45, .2, .2]):
            self.assertIsNone(blocking(np.array([box]), np.zeros(3), .40))


class SideStepTests(unittest.TestCase):
    def navigate(self, p, pose, t=.0):
        navigator = VisualNavigator()
        navigator.observe(p, pose, 0.)
        command = navigator.update(t, pose, .1)
        return navigator, command

    def test_left_box_stops_forward_and_crosses_right(self):
        p = perception(obstacles=[[.8, .15, .2, .2]], boundaries=straight_boundaries(), lane_pair=True)
        navigator, command = self.navigate(p, np.zeros(3))
        self.assertEqual(navigator.mode, 'cross')
        self.assertLessEqual(command[0], .02)
        self.assertLess(command[1], -.09)

    def test_right_box_crosses_left(self):
        p = perception(obstacles=[[.8, -.15, .2, .2]], boundaries=straight_boundaries(), lane_pair=True)
        navigator, command = self.navigate(p, np.zeros(3))
        self.assertEqual(navigator.mode, 'cross')
        self.assertGreater(command[1], .09)

    def test_oblique_boundary_still_centers_and_follows_its_heading(self):
        # A yawed view makes the far boundary a slanted fit; the perpendicular
        # distance at the body stays well-defined, so the crossing still ends.
        slope = .3
        # The box hugs the +.25 boundary: at x=.8 that line sits at y=.49 and
        # the box center is .12 further inside the lane.
        box = [.8, .37, .2, .2]
        controller = SideStepController()
        controller.begin(0., np.zeros(3), 0., box)
        command, _ = controller.update(0., perception(obstacles=[box],
                                                      boundaries=straight_boundaries(slope=slope), lane_pair=True),
                                       np.zeros(3), np.zeros(3), np.zeros(2), 0.)
        self.assertLess(command[1], -.09)
        # Place the body 2 cm left of the oblique line (y at x=.3 is -.16);
        # the new fit is relative to that pose, so both offsets shift.
        pose = np.array([.3, -.16 + .021, 0.])
        boundaries = straight_boundaries(offsets=(-.021, .479), slope=slope)
        outcome = controller.update(.5, perception(t=.4, obstacles=[[.5, .13, .2, .2]],
                                                   boundaries=boundaries, lane_pair=True),
                                    pose, pose, np.zeros(2), 0.)
        self.assertEqual(controller.phase, 'cruise')
        command = outcome[0]
        self.assertGreater(command[0], .15)
        self.assertGreater(command[2], 0.)  # steer onto the line's own heading

    def test_sign_flip_terminates_even_outside_center_tolerance(self):
        controller = SideStepController()
        controller.begin(0., np.zeros(3), 0., [.8, .15, .2, .2])
        controller.update(0., perception(obstacles=[[.8, .15, .2, .2]],
                                         boundaries=straight_boundaries(), lane_pair=True),
                          np.zeros(3), np.zeros(3), np.zeros(2), 0.)
        self.assertAlmostEqual(controller.entry, -.25, delta=1e-9)
        # Overshoot the far boundary by 7 cm: outside CENTER_TOLERANCE, but
        # the entry sign flipped, which is sufficient evidence of centered.
        pose = np.array([0., -.32, 0.])
        # Boundaries are recorded relative to the current pose, so the crossed
        # line now reads +.07 (7 cm to the left of the body).
        controller.update(.5, perception(t=.4, boundaries=straight_boundaries(offsets=(.07, .57)),
                                         lane_pair=True),
                           pose, pose, np.zeros(2), 0.)
        self.assertEqual(controller.phase, 'cruise')

    def test_cruise_rides_ridden_line_with_lateral_servo(self):
        controller = SideStepController()
        controller.begin(0., np.zeros(3), 0., [.8, .15, .2, .2])
        controller.phase = 'cruise'
        controller.guide = (np.array([0., -.25]), np.array([1., 0.]))
        # 6 cm right of the ridden line: servo back toward it while advancing.
        command, _ = controller.update(0., perception(boundaries=straight_boundaries(), lane_pair=True),
                                       np.zeros(3), np.array([0., -.31, 0.]), np.zeros(2), 0.)
        self.assertAlmostEqual(command[0], .20)
        self.assertGreater(command[1], .09)

    def test_cruise_rides_outside_the_line_away_from_the_lane(self):
        controller = SideStepController()
        controller.begin(0., np.zeros(3), 0., [.8, .15, .2, .2])
        controller.phase = 'cruise'
        controller.guide = (np.array([0., -.25]), np.array([1., 0.]))
        # Directly on the ridden line with the lane visible to the left: the
        # hold point sits a little further out, so the servo nudges right.
        command, _ = controller.update(0., perception(boundaries=straight_boundaries(), lane_pair=True),
                                       np.zeros(3), np.array([0., -.25, 0.]), np.zeros(2), 0.)
        self.assertAlmostEqual(controller.ride_offset, .04)
        self.assertLess(command[1], -.09)

    def test_blocking_box_on_ridden_line_retriggers_the_crossing(self):
        controller = SideStepController()
        controller.begin(0., np.zeros(3), 0., [.8, .15, .2, .2])
        controller.phase = 'cruise'
        controller.guide = (np.array([0., -.25]), np.array([1., 0.]))
        pose = np.array([0., -.25, 0.])
        # Riding the right line at y=-.25, the next alternating box sits 0.12 m
        # to its left, within the .45 m re-trigger range. Perception is
        # body-relative to the riding pose, so the boundary offsets and box y
        # are all measured from y=-.25.
        boundaries = straight_boundaries(offsets=(0., .5))
        command, _ = controller.update(0., perception(obstacles=[[.4, .12, .2, .2]],
                                                      boundaries=boundaries, lane_pair=True),
                                       pose, pose, np.zeros(2), 0.)
        self.assertEqual(controller.phase, 'cross')
        # The far boundary is selected on the next fresh frame, which then
        # commands the crossing toward the left line.
        command, _ = controller.update(.1, perception(t=.1, obstacles=[[.4, .12, .2, .2]],
                                                      boundaries=boundaries, lane_pair=True),
                                       pose, pose, np.zeros(2), 0.)
        self.assertGreater(command[1], .09)
        # A re-triggered crossing creeps forward: the box just passed trails
        # by less than a body length, so a pure lateral shift would land on it.
        self.assertGreater(command[0], .02)
        # A box near the opposite boundary does not block the ridden line.
        controller = SideStepController()
        controller.begin(0., np.zeros(3), 0., [.8, .15, .2, .2])
        controller.phase = 'cruise'
        controller.guide = (np.array([0., -.25]), np.array([1., 0.]))
        command, _ = controller.update(0., perception(obstacles=[[.4, .40, .2, .2]],
                                                      boundaries=boundaries, lane_pair=True),
                                       pose, pose, np.zeros(2), 0.)
        self.assertEqual(controller.phase, 'cruise')
        self.assertGreater(command[0], .15)

    def test_clear_two_line_frames_return_to_lane_following(self):
        navigator = VisualNavigator()
        navigator.observe(perception(obstacles=[[.8, .15, .2, .2]],
                                     boundaries=straight_boundaries(), lane_pair=True),
                          np.zeros(3), 0.)
        navigator.update(0., np.zeros(3))
        self.assertEqual(navigator.avoid.phase, 'cross')
        navigator.avoid.phase = 'cruise'
        navigator.avoid.guide = (np.array([0., -.25]), np.array([1., 0.]))
        for index in range(3):
            stamp = .1+index*.1
            clear = perception(obstacles=([[-.9, -.35, .2, .2]] if index == 0 else []),
                               boundaries=straight_boundaries(), lane_pair=True, t=stamp)
            navigator.observe(clear, np.array([0., -.25, 0.]), 0.)
            navigator.update(stamp, np.array([0., -.25, 0.]), .1)
        self.assertEqual(navigator.avoid.phase, 'idle')
        self.assertEqual(navigator.mode, 'lane')

    def test_exit_waits_until_passed_boxes_are_behind(self):
        navigator = VisualNavigator()
        navigator.observe(perception(obstacles=[[.8, .15, .2, .2]],
                                     boundaries=straight_boundaries(), lane_pair=True),
                          np.zeros(3), 0.)
        navigator.update(0., np.zeros(3))
        navigator.avoid.phase = 'cruise'
        navigator.avoid.guide = (np.array([0., -.25]), np.array([1., 0.]))
        # A box beside the near line is remembered as it is overtaken.
        pose = np.array([.2, -.25, 0.])
        navigator.observe(perception(obstacles=[[.2, .40, .2, .2]], t=.1,
                                     boundaries=straight_boundaries(), lane_pair=True),
                          pose, 0.)
        navigator.update(.1, pose)
        self.assertEqual(len(navigator.avoid.passed), 1)
        # Clear two-line frames do not exit while the box is still alongside:
        # the pull back to the lane center would brush it with a trailing arm.
        for index in range(3):
            stamp = .2+index*.1
            navigator.observe(perception(boundaries=straight_boundaries(), lane_pair=True, t=stamp),
                              pose, 0.)
            navigator.update(stamp, pose)
        self.assertEqual(navigator.avoid.phase, 'cruise')
        # Once the box is half a metre behind, the exit completes.
        pose = np.array([.8, -.25, 0.])
        for index in range(3):
            stamp = .5+index*.1
            navigator.observe(perception(boundaries=straight_boundaries(), lane_pair=True, t=stamp),
                              pose, 0.)
            navigator.update(stamp, pose)
        self.assertEqual(navigator.avoid.phase, 'idle')
        self.assertEqual(navigator.mode, 'lane')

    def test_stalled_cross_settles_then_resumes_visual_servo(self):
        controller = SideStepController()
        controller.begin(0., np.zeros(3), 0., [.8, .15, .2, .2])
        p = perception(obstacles=[[.8, .15, .2, .2]], boundaries=straight_boundaries(), lane_pair=True)
        controller.update(0., p, np.zeros(3), np.zeros(3), np.zeros(2), 0.)
        np.testing.assert_array_equal(controller.update(.7, p, np.zeros(3), np.zeros(3), np.zeros(2), 0.)[0],
                                      np.zeros(3))
        self.assertEqual(controller.phase, 'settle')
        np.testing.assert_array_equal(controller.update(1.1, p, np.zeros(3), np.zeros(3), np.zeros(2), 0.)[0],
                                      np.zeros(3))
        self.assertLess(controller.update(1.21, perception(t=1., obstacles=[[.8, .15, .2, .2]],
                                                          boundaries=straight_boundaries(), lane_pair=True),
                                          np.zeros(3), np.zeros(3), np.zeros(2), 0.)[0][1], -.09)
        self.assertEqual(controller.phase, 'cross')
        self.assertEqual(controller.recoveries, 1)

    def test_unresolved_stall_has_bounded_retries(self):
        controller = SideStepController()
        controller.begin(0., np.zeros(3), 0., [.8, .15, .2, .2])
        p = perception(obstacles=[[.8, .15, .2, .2]], boundaries=straight_boundaries(), lane_pair=True)
        controller.update(0., p, np.zeros(3), np.zeros(3), np.zeros(2), 0.)
        for time in np.arange(.7, 8., .1):
            controller.update(time, p, np.zeros(3), np.zeros(3), np.zeros(2), 0.)
        self.assertEqual(controller.phase, 'failed')
        np.testing.assert_array_equal(controller.update(9., p, np.zeros(3), np.zeros(3), np.zeros(2), 0.)[0],
                                      np.zeros(3))
        self.assertEqual(controller.recoveries, 3)

    def test_map_correction_keeps_targets_with_boxes(self):
        controller = SideStepController()
        controller.begin(0., np.zeros(3), 0., [.8, .15, .2, .2])
        controller.update(0., perception(obstacles=[[.8, .15, .2, .2]],
                                         boundaries=straight_boundaries(), lane_pair=True),
                          np.zeros(3), np.zeros(3), np.zeros(2), 0.)
        offset = controller.target-controller.origin
        old = controller.target.copy()
        far_point = controller.far[0].copy()
        translation = np.array([-.05, .02])
        controller.translate_map(translation)
        np.testing.assert_allclose(controller.target-controller.origin, offset)
        np.testing.assert_allclose(controller.target, old+translation)
        np.testing.assert_allclose(controller.far[0], far_point+translation)

    def test_crossing_is_perpendicular_to_the_corridor_despite_body_yaw(self):
        # The body is yawed 0.4 rad off the locked corridor. The crossing
        # command, expressed in the world frame, must stay perpendicular to
        # the corridor rather than to the yawed body axis.
        yaw = .4
        origin = np.array([2., 3.])
        pose = np.r_[origin, yaw]
        controller = SideStepController()
        controller.begin(0., pose, 0., [.8, .15, .2, .2])
        # Boundaries and box arrive in the yawed body frame of the pose.
        frame = rotation(yaw)
        boundaries = tuple((np.array([0., -np.tan(yaw), offset]), (.3, 1.2)) for offset in (-.25, .25))
        p = perception(obstacles=[[.8, .15, .2, .2]], boundaries=boundaries, lane_pair=True)
        command, _ = controller.update(0., p, pose, pose, np.zeros(2), 0.)
        world = command[:2]@rotation(pose[2]).T
        self.assertLess(world[1], -.09)   # cross toward the far (right) boundary
        # Motion is dominated by the lateral component; the small along-corridor
        # part is the deliberate feed-forward that cancels the AMP side-step creep.
        self.assertLess(abs(world[0]), .5*abs(world[1]))


class BoundaryRecordingTests(unittest.TestCase):
    def setUp(self):
        self.perception = CameraPerception(320, 240, 80, np.zeros(3), np.eye(3))
        self.x = np.linspace(.3, 1.2, 200)

    def test_two_lane_width_lines_record_both_boundaries(self):
        points = np.vstack([np.c_[self.x, .25+np.zeros_like(self.x)],
                            np.c_[self.x, -.25+np.zeros_like(self.x)]])
        lane = self.perception.fit_lane(points)
        self.assertIsNotNone(lane)
        self.assertTrue(self.perception.lane_pair)
        self.assertEqual(len(self.perception.boundaries), 2)
        offsets = sorted(np.polyval(poly, .5) for poly, _ in self.perception.boundaries)
        np.testing.assert_allclose(offsets, [-.25, .25], atol=.02)
        np.testing.assert_allclose(np.polyval(lane, .5), 0., atol=.02)

    def test_single_visible_line_records_one_boundary(self):
        points = np.c_[self.x, .25+np.zeros_like(self.x)]
        lane = self.perception.fit_lane(points)
        self.assertIsNotNone(lane)
        self.assertFalse(self.perception.lane_pair)
        self.assertEqual(len(self.perception.boundaries), 1)
        self.assertAlmostEqual(np.polyval(self.perception.boundaries[0][0], .5), .25, delta=.02)
        # The lane strategy still reconstructs the center by the known offset.
        self.assertAlmostEqual(np.polyval(lane, .5), 0., delta=.03)

    def test_two_non_lane_width_lines_do_not_flag_a_pair(self):
        points = np.vstack([np.c_[self.x, .05+np.zeros_like(self.x)],
                            np.c_[self.x, -.05+np.zeros_like(self.x)]])
        self.perception.fit_lane(points)
        self.assertFalse(self.perception.lane_pair)


if __name__ == '__main__':
    unittest.main()
