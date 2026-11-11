"""Brake and cross narrow gaps between visually observed alternating boxes."""
import numpy as np


def rotation(yaw):
    return np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])


class GapController:
    def __init__(self):
        self.phase='idle'
        self.origin=None
        self.target=None
        self.heading=0.
        self.started=0.
        self.completed=[]
        self.error=np.zeros(2)
        self.stalled_since=None
        self.recoveries=0

    def translate_map(self, displacement):
        """Keep targets attached to the same visually corrected box map."""
        for name in ('origin','target','identity'):
            value=getattr(self,name,None)
            if value is not None:setattr(self,name,value+displacement)
        self.completed=[value+displacement for value in self.completed]

    def update(self, time, perception, observed_pose, pose, world_velocity, corridor_heading=None):
        if self.phase=='idle':
            heading=observed_pose[2] if corridor_heading is None else corridor_heading
            frame=rotation(observed_pose[2]).T@rotation(heading)
            boxes=np.asarray(perception.obstacles)
            offsets=boxes[:,1]-np.polyval(perception.lane,boxes[:,0])
            aligned=np.column_stack([boxes[:,:2]@frame,boxes[:,2:4]@abs(frame),offsets])
            behind=[o for o in aligned if -.5<o[0]<0]
            ahead=[o for o in aligned if 0<o[0]<.8]
            if not behind or not ahead:return None
            back=max(behind,key=lambda o:o[0]);front=min(ahead,key=lambda o:o[0])
            offsets=[back[4],front[4]]
            if offsets[0]*offsets[1]>=0:return None
            lower=back[0]+back[2]/2+.135
            upper=front[0]-front[2]/2-.135
            if lower>=upper:return None
            gap=(lower+upper)/2
            if not -.06<gap<.13:return None
            origin=observed_pose[:2]+np.array([gap,0.])@rotation(heading).T
            # The gap identity belongs to the observed box pair, independent
            # of which side of the corridor the body currently occupies.
            identity=observed_pose[:2]+(back[:2]+front[:2])/2@rotation(heading).T
            if any(np.linalg.norm(identity-old)<.20 for old in self.completed):return None
            self.origin=origin
            self.identity=identity
            # Cross perpendicular to the observed corridor, even when the body
            # starts a few degrees off-axis. A body-fixed crossing would turn
            # that yaw error into centimetres of longitudinal drift.
            self.heading=heading
            lateral=front[1]-np.sign(offsets[1])*(front[3]/2+.29)
            self.target=observed_pose[:2]+np.array([gap,lateral])@rotation(self.heading).T
            self.phase='brake';self.started=time
            self.stalled_since=None;self.recoveries=0
        basis=rotation(self.heading)
        velocity=world_velocity@basis
        position_error=(self.target-pose[:2])@basis
        self.error=position_error
        if self.phase=='failed':return np.zeros(2)
        if self.phase=='settle':
            if time-self.started<.5:return np.zeros(2)
            self.phase='shift';self.started=time;self.stalled_since=None
        if self.phase=='brake':
            if time-self.started>.5 and abs(velocity[0])<.08:
                self.phase='shift';self.started=time
            else:return np.zeros(2)
        # This frozen AMP can remain in a crouched stationary state after a
        # direction change even though it receives nonzero velocity targets.
        # A measured stall is cleared by its ordinary zero-command standing
        # behavior, then the same visual error servo resumes. Never drive a
        # timed lateral distance or keep retrying an unresolved stall forever.
        if abs(position_error[1])>.025 and abs(velocity[1])<.015:
            if self.stalled_since is None:self.stalled_since=time
            elif time-self.stalled_since>.6:
                self.recoveries+=1
                self.phase='settle' if self.recoveries<=2 else 'failed'
                self.started=time
                return np.zeros(2)
        else:self.stalled_since=None
        lateral_error=position_error[1]-.2*velocity[1]
        vy=np.clip(3*lateral_error,-.20,.20)
        if abs(lateral_error)>.008:vy=np.sign(vy)*max(.10,abs(vy))
        else:vy=0.
        vx=np.clip(6*position_error[0]-.04*abs(vy)/.12-2.*velocity[0],-.12,.12)
        if abs(position_error[0])<.015 and abs(position_error[1])<.025 and time-self.started>.5:
            self.completed.append(self.identity.copy());self.phase='idle'
            return np.zeros(2)
        return np.array([vx,vy])@basis.T@rotation(pose[2])
