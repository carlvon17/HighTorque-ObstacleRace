"""Gait-assist adapter regressions: measured initiation floors of the frozen policy."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from navigation.command import (AMPCommandAdapter, HOLD_LATERAL, KICK_FORWARD,
                                KICK_LATERAL, KICK_TICKS, PULSE_TICKS, load_contract)


def track(adapter, desired, ticks):
    """Run the adapter closed-loop against the factory filter it models."""
    c = load_contract()
    filtered = adapter.filtered.copy()
    raws = []
    for _ in range(ticks):
        raws.append(adapter.update(desired, filtered))
        filtered = (c['alpha']*raws[-1]+(1-c['alpha'])*filtered)*np.asarray(c['scale'])
        filtered[abs(filtered) < np.asarray(c['deadband'])] = 0.
    return np.asarray(raws)


class GaitAssistTests(unittest.TestCase):
    def test_default_adapter_never_exceeds_the_navigation_bound(self):
        adapter = AMPCommandAdapter()
        raws = track(adapter, [.22, -.20, .30], 200)
        self.assertTrue(np.all(np.abs(raws) <= [.4, .4, 1.]))
        self.assertEqual(adapter.kick_ticks, 0)

    def test_forward_intent_from_standstill_primes_then_kicks(self):
        adapter = AMPCommandAdapter(gait_assist=True)
        raws = track(adapter, [.22, 0., 0.], PULSE_TICKS+KICK_TICKS+40)
        self.assertEqual(adapter.kick_axis, 0)
        # A settled stand ignores any pure forward request, so the assist
        # first emits a lateral priming pulse at the lateral floor (the
        # arming tick plus PULSE_TICKS decrements stay in the pulse phase).
        pulse = slice(PULSE_TICKS+1)
        self.assertTrue(np.all(raws[pulse, 0] == 0.))
        self.assertTrue(np.all(raws[pulse, 1] >= KICK_LATERAL-.01))
        # The forward kick then rides above the initiation floor; after it
        # the request ramps back to the navigator target without reverse.
        kick = slice(PULSE_TICKS+1, PULSE_TICKS+KICK_TICKS)
        self.assertTrue(np.all(raws[kick, 0] >= .70))
        self.assertTrue(np.all(raws[PULSE_TICKS+1:, 1] == 0.))
        self.assertAlmostEqual(adapter.target[0], .22, delta=1e-9)
        np.testing.assert_allclose(adapter.filtered, [.22, 0., 0.], atol=1e-9)

    def test_lateral_intent_from_standstill_kicks_lateral_not_forward(self):
        adapter = AMPCommandAdapter(gait_assist=True)
        raws = track(adapter, [.05, -.15, 0.], KICK_TICKS+10)
        self.assertEqual(adapter.kick_axis, 1)
        # The small longitudinal hold passes through at its own floor only.
        self.assertLessEqual(np.max(adapter.target[0:1]), .10)
        self.assertLessEqual(np.max(raws[:, 0]), .34)
        self.assertLessEqual(np.max(raws[:KICK_TICKS, 1]), -KICK_LATERAL+.01)
        # The forward axis never rises past its own small floor.
        self.assertLessEqual(adapter.filtered[0], .101)
        self.assertAlmostEqual(adapter.filtered[2], 0., delta=1e-9)

    def test_midgait_small_lateral_intents_gain_authority(self):
        adapter = AMPCommandAdapter(gait_assist=True)
        # Establish a gait: filtered forward above the deadband.
        adapter.filtered = np.array([.22, 0., 0.])
        raws = track(adapter, [.22, .12, 0.], 10)
        self.assertEqual(adapter.kick_ticks, 0)  # not standing, no new kick
        self.assertAlmostEqual(np.max(np.abs(raws[:, 1])), HOLD_LATERAL, delta=.05)

    def test_withdrawn_intent_cancels_the_kick(self):
        adapter = AMPCommandAdapter(gait_assist=True)
        track(adapter, [.22, 0., 0.], 5)
        self.assertGreater(adapter.kick_ticks, 0)
        raws = track(adapter, [0., 0., 0.], 5)
        np.testing.assert_array_equal(raws, 0.)
        self.assertEqual(adapter.kick_ticks, 0)

    def test_kick_stays_inside_the_factory_contract(self):
        c = load_contract()
        adapter = AMPCommandAdapter(gait_assist=True)
        raws = track(adapter, [.3, .2, .3], KICK_TICKS)
        self.assertTrue(np.all(raws >= np.asarray(c['lower'])-1e-9))
        self.assertTrue(np.all(raws <= np.asarray(c['upper'])+1e-9))


class RecoveryInteractionTests(unittest.TestCase):
    def test_kick_window_is_exempt_from_motion_stall_detection(self):
        from navigation.runtime import NavigationRuntime
        runtime = NavigationRuntime(32, 24, [30, 30, 15, 11], [0, 0, 0], np.eye(3), gait_assist=True)
        runtime.set_attitude(1., np.eye(3))
        runtime.last_frame_time = 1.
        runtime.good_frames = 3
        runtime.rgb_usable = True
        runtime.navigator.mode = 'lane'
        # Hold the navigator output steady; this test exercises the runtime
        # gating around the adapter, not the lane strategy.
        runtime.navigator.update = lambda timestamp, pose, dt: np.array([.22, 0., .2])
        runtime.command(1.0)
        self.assertGreater(runtime.adapter.kick_ticks, 0)
        # Without the kick exemption the same stall would settle at t=1.8,
        # mid-kick. Inside the window the command keeps flowing.
        for stamp in np.arange(1.02, 2.6, .02):
            runtime.set_attitude(float(stamp), np.eye(3))
            runtime.last_frame_time = float(stamp)
            runtime.command(float(stamp))
        self.assertEqual(runtime.reason, 'ready')
        # Once the kick has ended, the same zero-velocity stall settles again.
        seen = set()
        for stamp in np.arange(2.6, 4.2, .02):
            runtime.set_attitude(float(stamp), np.eye(3))
            runtime.last_frame_time = float(stamp)
            runtime.command(float(stamp))
            seen.add(runtime.reason)
        self.assertTrue(seen & {'standing_recovery', 'motion_stalled'}, seen)


if __name__ == '__main__':
    unittest.main()
