"""A short motion, an actual zero publication, then a newer camera capture."""
import numpy as np


class BoxSteps:
    def __init__(self):
        self.active=None
        self.stop_requested=None
        self.zero_stamp=None
        self.previous=None
        self.count=0
        self.reason=None

    def stop(self,stamp):
        if self.active is not None and self.active['sent']:
            self.previous=self.active
        self.active=None
        if self.stop_requested is None:
            self.stop_requested=stamp
            self.zero_stamp=None

    def published(self,stamp,vector):
        if np.max(np.abs(vector))<1.e-8:
            if self.stop_requested is not None and self.zero_stamp is None and stamp>=self.stop_requested:
                self.zero_stamp=stamp
        elif self.active is not None:
            self.active['sent']=True

    def check(self,t,frame,position):
        move=self.active
        if move is not None and (t>=move['until'] or np.linalg.norm(position-move['origin'])>=move['distance']):
            self.stop(t)
        if self.stop_requested is not None:
            if self.zero_stamp is None:
                return 'await_zero_publication'
            if frame.stamp<=self.zero_stamp:
                return 'wait_post_step_capture'
            self.stop_requested=None
        return None

    def apply(self,t,frame,position,vector,axis,sign,distance=None):
        if not np.any(np.abs(vector)>1.e-5):
            if self.active is not None:self.stop(t)
            return vector
        if self.active is not None:
            if self.active['axis']!=axis:
                self.stop(t)
                self.reason='direction_change'
                return np.zeros(3)
            return vector
        measured=frame.visual_xy[axis] if axis<2 else None
        if axis<2 and measured is None:
            self.reason='missing_step_measurement'
            return np.zeros(3)
        previous=self.previous
        if previous is not None and previous['axis']<2:
            k=previous['axis']
            measured_after=frame.visual_xy[k]
            if measured_after is None or previous['sign']*(measured_after-previous['visual'])<.003:
                self.reason='wait_visual_displacement'
                return np.zeros(3)
        self.previous=None
        # A geometric stop is evaluated on every policy publication as well
        # as at each plan. The independent time cap also bounds every grant
        # when the encoders are missing or the robot remains stationary.
        duration=.36 if axis==1 else .40
        if distance is None:
            distance=.046 if axis==1 else .063
        self.active=dict(axis=axis,sign=sign,visual=measured,origin=position.copy(),
                         until=t+duration,distance=distance,sent=False)
        self.count+=1
        self.reason='short_side_step' if axis==1 else 'short_forward_step' if axis==0 else 'short_turn'
        return vector
