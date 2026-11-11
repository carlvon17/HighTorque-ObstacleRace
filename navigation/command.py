"""Navigation-side adaptation to the unchanged factory AMP input filter.

The factory still applies its own filter/deadband. This module produces the
raw velocity request; it never changes policy weights or controller settings.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def load_contract():
    return json.loads(Path(__file__).with_name('amp_contract.json').read_text())


def verify_device_files(amp_directory, controller_binary=None, pd_path=None):
    """Read-only identity check. A different factory revision is not accepted."""
    contract = load_contract()
    paths = {name: Path(amp_directory)/name for name in ('amp.yaml', 'amp.trt')}
    expected = dict(contract['sha256'])
    if pd_path is not None:
        paths['pd.yaml'] = Path(pd_path)
    if controller_binary is not None:
        paths['controller_binary'] = Path(controller_binary)
        expected['controller_binary'] = contract['controller_binary_sha256']
    actual = {}
    for name, path in paths.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected[name]:
            raise ValueError('Factory AMP contract changed: ' + name)
        actual[name] = digest
    return actual


KICK_FORWARD=.75     # measured forward gait-initiation floor of this policy
KICK_LATERAL=.45     # measured lateral gait-initiation floor
PULSE_TICKS=10       # 0.2 s priming pulse that unlocks forward kicks
KICK_TICKS=60        # 1.2 s at 50 Hz
HOLD_LATERAL=.40     # measured lateral authority while the gait is active


class AMPCommandAdapter:
    """Quantized low-speed targets and bounded filter feed-forward at 50 Hz.

    Feedback, if provided, is the actual policy's observed vx/vy/wz, not body
    velocity. Stop never requests reverse motion to cancel a filter.

    With ``gait_assist`` the adapter shapes the request around the measured
    gait behaviour of this frozen policy in simulation: from a settled stand
    (a fifth of a second of zero commands suffices) it ignores ANY pure
    forward request -- even 3 s at 0.8 m/s does not initiate -- while a
    lateral or reverse request does, lateral initiation needs ~0.45 m/s, a
    brief 0.45 lateral pulse unlocks subsequent forward kicks, an established
    gait sustains at 0.2, and mid-gait lateral corrections below ~0.4 have no
    authority. The assist emits a 0.2 s lateral priming pulse followed by a
    1.2 s forward kick when forward intent arrives from a standstill, kicks
    laterally for lateral intent, and raises mid-gait lateral intents to the
    effective magnitude. It is off by default; the deployed factory path is
    unchanged.
    """
    def __init__(self, gait_assist=False):
        self.contract = load_contract()
        self.filtered = np.zeros(3)
        self.active = np.zeros(3, dtype=bool)
        self.start_threshold = np.array([.04, .04, .08])
        self.stop_threshold = np.array([.02, .02, .04])
        self.minimum_target = np.array([.10, .10, .20])
        self.raw_limit = np.array([.4, .4, 1.])
        self.deadband = np.asarray(self.contract['deadband'], dtype=float)
        self.gait_assist = bool(gait_assist)
        self.kick_axis = None
        self.kick_sign = 1
        self.kick_ticks = 0
        self.target = np.zeros(3)

    def _gait_assist(self):
        standing = bool(np.all(np.abs(self.filtered) < self.deadband))
        if self.kick_ticks > 0:
            self.kick_ticks -= 1
            if not self.active[self.kick_axis]:
                # The navigator withdrew the intent; do not finish the kick.
                self.kick_ticks = 0
            elif self.kick_axis == 1:
                self.target[1] = np.sign(self.target[1])*KICK_LATERAL
            elif self.kick_ticks >= KICK_TICKS:
                # Priming pulse: pure lateral, under 6 cm of excursion, whose
                # only purpose is to leave the standing attractor.
                self.target[0] = 0.
                self.target[1] = self.kick_sign*KICK_LATERAL
            else:
                self.target[0] = max(self.target[0], KICK_FORWARD)
        elif standing:
            # A pure lateral intent wins the kick: the crossing phase arrives
            # standing with a small longitudinal hold and must not lunge.
            if self.active[1]:
                self.kick_axis, self.kick_ticks = 1, KICK_TICKS
                self.target[1] = np.sign(self.target[1])*KICK_LATERAL
            elif self.active[0] and self.target[0] > 0:
                self.kick_sign = 1 if self.target[1] >= 0 else -1
                self.kick_axis, self.kick_ticks = 0, KICK_TICKS+PULSE_TICKS
                self.target[0] = 0.
                self.target[1] = self.kick_sign*KICK_LATERAL
        elif self.active[1] and abs(self.target[1]) < HOLD_LATERAL:
            self.target[1] = np.sign(self.target[1])*HOLD_LATERAL

    def update(self, desired, feedback=None, enabled=True):
        desired = np.asarray(desired, dtype=float)
        if desired.shape != (3,) or not np.isfinite(desired).all():
            raise ValueError('Invalid navigation velocity')
        if feedback is not None:
            feedback = np.asarray(feedback, dtype=float)
            if feedback.shape != (3,) or not np.isfinite(feedback).all():
                raise ValueError('Invalid policy command feedback')
            self.filtered = feedback.copy()
        self.active = ((abs(desired) >= self.start_threshold) |
                       (self.active & (abs(desired) > self.stop_threshold))) if enabled else np.zeros(3, bool)
        self.target = np.where(self.active, np.sign(desired)*np.maximum(abs(desired), self.minimum_target), 0.)
        # With this exact Pro network, 0.10--0.16 forward-only inputs reach
        # the policy but settle back into standing in the simulation probe.
        # 0.20 sustains walking. Lateral motion has a different response and
        # must retain small longitudinal corrections for narrow box gaps.
        if self.target[0] > 0 and self.target[1] == 0:
            self.target[0] = max(.20, self.target[0])
        if self.gait_assist:
            self._gait_assist()
        c = self.contract
        previous = self.filtered.copy()
        raw = (self.target/np.asarray(c['scale'])-(1-c['alpha'])*previous)/c['alpha']
        raw = np.where(self.target > 0, np.maximum(raw, 0), np.minimum(raw, 0))
        driven = self.active
        if self.kick_ticks > 0:
            # A kick drives axes the navigator is not commanding (the priming
            # pulse is lateral under a purely forward intent).
            driven = driven.copy()
            driven[self.kick_axis] = True
            if self.kick_axis == 0 and self.kick_ticks >= KICK_TICKS:
                driven[1] = True
        raw = np.where(driven, raw, 0.)
        limit = self.raw_limit.copy()
        if self.kick_ticks > 0:
            # Only the kicked axis (and its priming pulse) may exceed the
            # navigation-side bound; the factory contract bound still applies.
            limit[self.kick_axis] = max(limit[self.kick_axis],
                                        KICK_FORWARD if self.kick_axis == 0 else KICK_LATERAL)
            if self.kick_axis == 0 and self.kick_ticks >= KICK_TICKS:
                limit[1] = max(limit[1], KICK_LATERAL)
        raw = np.clip(raw, np.maximum(c['lower'], -limit), np.minimum(c['upper'], limit))
        self.filtered = (c['alpha']*raw+(1-c['alpha'])*previous)*c['scale']
        self.filtered[abs(self.filtered) < np.asarray(c['deadband'])] = 0.
        return raw
