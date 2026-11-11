"""Regression for an expiring rt1 velocity request while the vision loop is busy."""
import unittest
from unittest.mock import Mock

import numpy as np

from sim.rt1_control import RT1Control, camera_profile, navigation_sources


class RT1SimulationTests(unittest.TestCase):
    def test_pinned_device_inputs(self):
        self.assertTrue(navigation_sources())
        profile = camera_profile()
        self.assertEqual((profile['width'], profile['height'], profile['fps']), (640, 480, 6))
        self.assertEqual(profile['processing_scale'], .4)

    def test_busy_frame_does_not_refresh_velocity(self):
        control = RT1Control(256, 192, (241., 241., 130., 98.), .22)
        control.trace = [dict(capture_ms='0', start_ms='40', plan_ms='1270', background_collected='0')]
        control.period = 8.
        origin = control.origin
        vision = control.runtime = Mock()
        vision.desired = np.array([.22, 0., 0.])
        vision.deadline = origin + 1.
        control.request = vision.desired.copy()
        control.deadline = origin + .16
        rgb = np.zeros((192, 256, 3), np.uint8)
        control.capture(rgb, np.zeros(rgb.shape[:2]), np.eye(3), True)
        for index in range(2, 65):
            t = origin + index*.02
            command = control.tick(t, np.eye(3), np.zeros(3), True)
            if t < origin+.16:
                self.assertGreater(command[0], 0.)
            if t > origin+.16:
                np.testing.assert_array_equal(command, np.zeros(3))
        frame = vision.observe.call_args.args[0]
        self.assertEqual(frame.time, origin)
        vision.command.assert_called_once_with(origin+1.27, enabled=True)
        self.assertGreater(control.expired_ticks, 0)
        self.assertEqual(control.vision_index, 1)
