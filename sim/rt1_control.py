"""Offline rt1 camera, vision-loop and independent 50 Hz command scheduling."""
import csv
from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np

from navigation.command import AMPCommandAdapter
from vendor.pi_plus_pro.rt1.runtime import CameraFrame, OnboardNavigation

ROOT = Path(__file__).resolve().parents[1]
RT1 = ROOT / 'vendor/pi_plus_pro/rt1'
CONFIG = ROOT / 'deploy/pro_model_config.json'
PROFILE = json.loads((RT1 / 'profile.json').read_text())
STANDING_SETTLE_S = 2.


def navigation_sources():
    """Check the hardware hashes, discounting only two package-qualified imports."""
    if hashlib.sha256(CONFIG.read_bytes()).hexdigest() != PROFILE['source_config_sha256']:
        raise ValueError('rt1 camera/IMU configuration differs from device')
    paths = []
    for name, expected in PROFILE['navigation_core_sha256'].items():
        path = RT1 / name if name in ('runtime.py', 'visual.py', 'gap_controller.py') else ROOT / 'navigation' / name
        content = path.read_bytes()
        if path.parent == RT1:
            for dep in (b'visual', b'gap_controller'):
                content = content.replace(b'from .'+dep+b' import ', b'from navigation.'+dep+b' import ')
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError('rt1 navigation source differs from device: '+name)
        paths.append(path.relative_to(ROOT))
    return paths


def camera_profile():
    camera = dict(PROFILE['camera'])
    config = json.loads(CONFIG.read_text())
    camera.update(processing_scale=config['perception_scale'], rectify=True)
    return camera


class RecordedWorker:
    """Expose a tracking result when the Jetson trace had a free worker slot.

    The actual rt1 tracker still computes the result from the simulated image.
    A synchronous rt1 handoff can also collect an outstanding job immediately.
    """
    def __init__(self):
        self.available = False
        self.job = None

    def submit(self, function, *args):
        if self.job is not None:
            raise RuntimeError('rt1 queued more than one background job')
        self.job = function, args
        return self

    def done(self):
        return self.available

    def result(self):
        function, args = self.job
        self.job = None
        return function(*args)

    def shutdown(self, wait=True):
        pass


class RT1Control:
    def __init__(self, width, height, intrinsics, speed):
        config = json.loads(CONFIG.read_text())
        self.runtime = OnboardNavigation(width, height, intrinsics, config['camera_position_body_m'],
            config['camera_rotation_optical_to_body'], speed, config['observation_lease_s'])
        self.runtime.worker.shutdown()
        self.worker = self.runtime.worker = RecordedWorker()
        self.guard = AMPCommandAdapter()
        self.request = np.zeros(3)
        self.deadline = 0.
        self.forwarded = np.zeros(3)
        self.last_publish = None
        self.next_publish = 0.
        self.frames = deque()
        self.in_flight = None
        self.frame_result = None
        self.head_reference = config['head_reference_rad']
        with (RT1 / 'frame_timing.csv').open() as f:
            self.trace = list(csv.DictReader(f))
        self.origin = STANDING_SETTLE_S
        # Repeat the sampled camera/compute cadence without restarting the device.
        self.period = (float(self.trace[-1]['plan_ms'])-float(self.trace[0]['capture_ms']))/1000 + 1/PROFILE['camera']['fps']
        self.capture_index = self.vision_index = 0
        self.planning = False
        self.previous_plan = -1.
        self.published_plans = self.expired_ticks = 0

    def event(self, index, column):
        cycle, row = divmod(index, len(self.trace))
        return self.origin + cycle*self.period + float(self.trace[row][column])/1000

    def camera_due(self, t):
        return self.event(self.capture_index, 'capture_ms') <= t+1e-8

    def capture(self, rgb, depth, attitude, head_ok):
        stamp = self.event(self.capture_index, 'capture_ms')
        self.frames.append((CameraFrame(stamp, rgb, depth), attitude.copy(), head_ok))
        self.capture_index += 1

    def inhibit(self):
        self.request[:] = 0.
        self.deadline = 0.

    def plan(self, t, attitude, enabled):
        self.runtime.set_attitude(t, attitude)
        self.runtime.command(t, enabled=enabled)
        self.previous_plan = t
        if enabled and t <= self.runtime.deadline:
            self.request = self.runtime.desired.copy()
            self.deadline = self.runtime.deadline
        else:
            self.inhibit()
        self.published_plans += 1

    def tick(self, t, attitude, feedback, head_ok):
        self.frame_result = None
        while True:
            column = 'plan_ms' if self.planning else 'start_ms'
            event = self.event(self.vision_index, column)
            if event > t+1e-8:
                break
            if self.planning:
                frame, observation, capture_ok = self.in_flight
                self.plan(event, attitude, head_ok and capture_ok)
                self.frame_result = frame, observation, self.runtime.diagnostic(), event
                self.in_flight = None
                self.vision_index += 1
            else:
                frame, capture_attitude, capture_ok = self.frames.popleft()
                self.runtime.set_attitude(event, attitude)
                self.worker.available = bool(int(self.trace[self.vision_index % len(self.trace)]['background_collected']))
                if capture_ok and event-frame.time <= .2:
                    observation = self.runtime.observe(frame, capture_attitude, frame.time, self.inhibit)
                else:
                    self.inhibit()
                    observation = None
                    capture_ok = False
                self.in_flight = frame, observation, capture_ok
            self.planning = not self.planning
        # Vision can plan between frames; it cannot refresh a request while busy.
        if not self.planning and t >= self.origin and t-self.previous_plan >= .02-1e-8:
            self.plan(t, attitude, head_ok)
        if t+1e-8 >= self.next_publish:
            self.forwarded = self.request.copy() if t <= self.deadline else np.zeros(3)
            self.last_publish = t
            self.next_publish = t+.02
        self.expired_ticks += int(t > self.deadline and self.deadline > 0.)
        # Healthy, hands-off ROS link: publisher and guard tick together. The
        # guard alone adapts the forwarded desired velocity to policy feedback.
        fresh = self.last_publish is not None and t-self.last_publish <= .2
        return self.guard.update(self.forwarded, feedback, enabled=head_ok and fresh and t>=self.origin)

    def close(self, path):
        self.runtime.close(path)
