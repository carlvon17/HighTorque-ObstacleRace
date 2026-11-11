"""Side-step box avoidance: cross to the far lane boundary and ride that line.

A detected box always stands on one side of the lane. The robot brakes, shifts
laterally until the far boundary passes beneath the body, then follows that
single line forward until the next box or a clear two-line view. Every lateral
decision is a perpendicular distance from the body axis to a boundary, measured
in the attitude-levelled ground frame: an oblique (yawed) image of the line
cannot hide the crossing, because that distance passes through zero and flips
sign as the body moves across the line.
"""
from __future__ import annotations
import numpy as np

TRIGGER_RANGE=1.0        # a box closer than this ahead starts a crossing from lane mode
GUIDE_TRIGGER=.45        # re-trigger distance while riding a line; below the 0.5 m box
                         # spacing so cruising advances, with room to stop and cross
LANE_BAND=0.40           # box-center offset from the lane center that blocks the lane
GUIDE_BAND=0.32          # box-center offset from the ridden line that blocks it
CENTER_TOLERANCE=.05     # boundary is under the body
FLIP_TOLERANCE=.09       # a sign change this small still counts as centered
MAX_CROSS_TRAVEL=.55     # absolute lateral cap so a crossing always terminates
RIDE_OFFSET=.04          # ride just outside the line: white box surfaces bias
                         # the line fit inward, and the lane side is where boxes are
EXIT_FRAMES=3            # consecutive clear two-line frames that end avoidance
EXIT_CLEARANCE=.50       # a remembered box must be at least this far away
                         # before returning to the lane, or the pull toward
                         # the lane center brushes it with a trailing arm
HOLD_LIMIT=4.            # seconds without any far boundary before giving up


def rotation(yaw):
    return np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])


def wrap(angle):
    return (angle+np.pi)%(2*np.pi)-np.pi


def blocking(obstacles,poly,band):
    """Nearest box ahead whose center lies within ``band`` of the lane center."""
    best=None
    for box in np.asarray(obstacles,dtype=float).reshape(-1,4):
        ahead=box[0]
        if 0<ahead<TRIGGER_RANGE and abs(box[1]-np.polyval(poly,ahead))<band:
            if best is None or ahead<best[0]:best=box
    return best


def boundary_lines(boundaries,observed_pose):
    """Boundaries as world (point, unit tangent, curvature) at their support start."""
    result=[]
    for poly,span in boundaries or ():
        x=float(np.clip(span[0],0.,.6))
        slope=float(np.polyval(np.polyder(poly),x))
        tangent=np.array([1.,slope]);tangent/=np.linalg.norm(tangent)
        frame=rotation(observed_pose[2])
        point=np.array([x,float(np.polyval(poly,x))])@frame.T+observed_pose[:2]
        result.append((point,tangent@frame.T,abs(float(poly[0])) if len(poly)>2 else 0.))
    return result


def signed_distance(pose,point,tangent):
    """Positive when the line passes to the left of the body."""
    normal=np.array([-tangent[1],tangent[0]])
    return float((point-pose[:2])@normal)


class SideStepController:
    """Cross away from each box and ride the far boundary until the lane is clear."""

    def __init__(self):
        self.phase='idle'
        self.heading=None
        self.reference_heading=None
        self.box_point=None
        self.origin=None
        self.target=None
        self.entry=None
        self.far=None            # (point, tangent) of the boundary being crossed toward
        self.guide=None          # (point, tangent) of the ridden boundary
        self.passed=[]           # world centers of boxes met during this episode
        self.frame=None
        self.exit_streak=0
        self.started=0.
        self.stalled_since=None
        self.settled_until=0.
        self.recoveries=0
        self.creep=False
        self.ride_offset=0.
        self.last_time=None
        self.dt=0.
        self.error=np.zeros(2)
        self.path=np.empty((0,2))

    def translate_map(self,displacement):
        """Keep targets attached to the same visually corrected box map."""
        for name in ('origin','target','box_point'):
            value=getattr(self,name)
            if value is not None:setattr(self,name,value+displacement)
        for name in ('far','guide'):
            value=getattr(self,name)
            if value is not None:setattr(self,name,(value[0]+displacement,value[1]))
        self.passed=[point+displacement for point in self.passed]

    def begin(self,time,observed_pose,corridor_heading,box,creep=False):
        """Start crossing toward the boundary farthest from ``box``.

        ``creep`` advances slowly forward during the crossing instead of
        holding position: between alternating boxes a pure lateral shift
        would land on the box just passed, which trails by less than a body
        length.
        """
        self.box_point=np.array(box[:2])@rotation(observed_pose[2]).T+observed_pose[:2]
        self.reference_heading=None if corridor_heading is None else float(corridor_heading)
        self.heading=None if corridor_heading is None else float(corridor_heading)
        self.origin=observed_pose[:2].copy()
        self.creep=creep
        self.target=None;self.entry=None;self.far=None;self.guide=None
        self.exit_streak=0;self.frame=None;self.ride_offset=0.
        self.stalled_since=None;self.recoveries=0
        self.phase='cross';self.started=float(time)

    def update(self,time,perception,observed_pose,pose,world_velocity,corridor_heading):
        """Body-frame [vx, vy, yaw_rate] and a direct-servo flag, or None when idle."""
        if self.phase=='idle':return None
        if self.phase=='failed':return np.zeros(3),True
        if self.last_time is not None:self.dt=min(.1,max(0.,time-self.last_time))
        self.last_time=float(time)
        fresh=perception.time!=self.frame
        if fresh:self.frame=perception.time
        lines=boundary_lines(perception.boundaries,observed_pose)
        if self.phase=='settle':
            if time<self.settled_until:return np.zeros(3),True
            self.phase='cross';self.started=time;self.stalled_since=None
        if self.phase=='cross':
            outcome=self._cross(time,observed_pose,pose,world_velocity,lines,fresh)
            if outcome is not None:return outcome
            # The crossing just completed mid-tick; this frame is consumed, so
            # cruise servos on it without re-running its fresh-frame checks.
            return self._cruise(time,perception,observed_pose,pose,world_velocity,
                                corridor_heading,lines,False)
        return self._cruise(time,perception,observed_pose,pose,world_velocity,
                            corridor_heading,lines,fresh)

    def _select_far(self,lines):
        """The straight boundary farthest from the blocking box.

        Boxes hug one side of the lane, so the farthest boundary is the far
        side of the lane regardless of where the body currently stands.
        """
        best=None;best_score=.15
        for point,tangent,curvature in lines:
            if curvature>.2:continue
            if self.reference_heading is not None and \
               abs(wrap(np.arctan2(tangent[1],tangent[0])-self.reference_heading))>.35:continue
            score=abs(signed_distance(np.r_[self.box_point,0.],point,tangent))
            if score>best_score:best,best_score=(point,tangent),score
        return best

    def _cross(self,time,observed_pose,pose,world_velocity,lines,fresh):
        if fresh:
            far=self._select_far(lines)
            if far is not None:
                self.far=far
                if self.heading is None:
                    # No corridor lock: orient the crossing along the selected
                    # boundary itself instead of the possibly yawed body.
                    self.heading=float(np.arctan2(far[1][1],far[1][0]))
                if self.entry is None:self.entry=signed_distance(observed_pose,*far)
                normal=np.array([-far[1][1],far[1][0]])
                self.target=observed_pose[:2]+signed_distance(observed_pose,*far)*normal
        if self.far is None or self.heading is None:
            if time-self.started>HOLD_LIMIT:self.phase='failed'
            return np.zeros(3),True
        basis=rotation(self.heading)
        velocity=world_velocity@basis
        if self.creep and self.dt>0:
            # Walk the hold reference forward: the body advances diagonally
            # past the box it just cleared instead of shifting beside it.
            self.origin=self.origin+basis[0]*.10*self.dt
        error=(pose[:2]-self.origin)@basis
        self.error=np.array([error[0],error[1]])
        self.path=np.empty((0,2))
        distance=signed_distance(pose,*self.far)
        if abs(distance)<CENTER_TOLERANCE or \
           (self.entry is not None and self.entry*distance<0 and abs(distance)<FLIP_TOLERANCE):
            # The boundary is under the body, or was just crossed: the sign
            # change is yaw-invariant evidence of "line in the middle of view".
            self._start_cruise(time,far_line=self.far)
            return None
        if abs(error[1])>MAX_CROSS_TRAVEL:
            # The lateral cap guarantees termination even without a usable
            # boundary; cruising re-centers on the line once it reappears.
            self._start_cruise(time,far_line=self.far)
            return None
        if abs(distance)>.025 and abs(velocity[1])<.015:
            if self.stalled_since is None:self.stalled_since=time
            elif time-self.stalled_since>.6:
                self.recoveries+=1
                if self.recoveries<=2:
                    self.phase='settle';self.settled_until=time+.5
                else:self.phase='failed'
                return np.zeros(3),True
        else:self.stalled_since=None
        lateral_error=distance-.2*velocity[1]
        vy=np.clip(3*lateral_error,-.20,.20)
        vy=np.sign(vy)*max(.10,abs(vy)) if abs(lateral_error)>.008 else 0.
        # Pure lateral walking creeps forward in this AMP policy; hold the
        # longitudinal position closed-loop instead of integrating it.
        hold=-6*error[0]-2*velocity[0]
        if self.creep:vx=np.clip(hold,-.12,.12)
        else:vx=np.clip(hold-.04*abs(vy)/.12,-.12,.12)
        wz=np.clip(2.5*wrap(self.heading-pose[2]),-.5,.5)
        command=np.zeros(3)
        command[:2]=np.array([vx,vy])@basis.T@rotation(pose[2])
        command[2]=wz
        return command,True

    def _start_cruise(self,time,far_line):
        self.guide=far_line
        self.far=None;self.entry=None;self.ride_offset=0.
        self.phase='cruise';self.started=time;self.exit_streak=0

    def _cruise(self,time,perception,observed_pose,pose,world_velocity,corridor_heading,lines,fresh):
        if fresh:
            self._record_boxes(perception,observed_pose)
            clear=not any(0.<box[0]<1.5 and abs(box[1])<1.2
                          for box in np.asarray(perception.obstacles,dtype=float).reshape(-1,4))
            behind=all(np.linalg.norm(point-pose[:2])>EXIT_CLEARANCE
                       for point in self.passed)
            self.exit_streak=self.exit_streak+1 if (clear and behind and perception.lane_pair) else 0
            if self.exit_streak>=EXIT_FRAMES:
                # Several frames with no box ahead, the boxes of this episode
                # behind, and both boundaries visible: the lane is clear and
                # line following resumes.
                self.phase='idle';self.guide=None;self.passed=[]
                return None
            if self.guide is not None:
                box=self._guide_blocking(perception,observed_pose)
                if box is not None:
                    self.begin(time,observed_pose,corridor_heading,box,creep=True)
                    return self.update(time,perception,observed_pose,pose,world_velocity,corridor_heading)
            best=None
            for candidate in lines:
                distance=signed_distance(observed_pose,candidate[0],candidate[1])
                if abs(distance)<.30 and (best is None or abs(distance)<abs(best[1])):
                    best=((candidate[0],candidate[1]),distance)
            if best is not None:self.guide=best[0]
            for candidate in lines:
                interior=signed_distance(np.r_[candidate[0],0.],*self.guide)
                if .3<abs(interior)<.8:
                    # The other boundary lies on the interior side; ride the
                    # same small distance outward, away from the boxes.
                    self.ride_offset=-np.sign(interior)*RIDE_OFFSET
                    break
        if self.guide is None:
            self.error=np.zeros(2);self.path=np.empty((0,2))
            return np.zeros(3),False
        point,tangent=self.guide
        distance=signed_distance(pose,point,tangent)
        velocity=world_velocity@rotation(pose[2])
        self.error=np.array([0.,distance])
        self.target=point
        lateral_error=distance-self.ride_offset-.2*velocity[1]
        vy=np.clip(3*lateral_error,-.20,.20)
        vy=np.sign(vy)*max(.10,abs(vy)) if abs(lateral_error)>.008 else 0.
        wz=np.clip(2.5*wrap(np.arctan2(tangent[1],tangent[0])-pose[2]),-.8,.8)
        along=np.linspace(.2,1.2,6)
        self.path=((point[None,:]+along[:,None]*tangent[None,:])-pose[:2])@rotation(pose[2])
        return np.array([.20,vy,wz]),False

    def _record_boxes(self,perception,observed_pose):
        """Remember world centers of boxes met along the ridden line.

        Boxes just overtaken leave the forward window while still beside the
        body; the exit test needs their positions to stay gated until the
        return to the lane center cannot brush them.
        """
        if self.guide is None:return
        point,tangent=self.guide
        normal=np.array([-tangent[1],tangent[0]])
        frame=rotation(observed_pose[2])
        for box in np.asarray(perception.obstacles,dtype=float).reshape(-1,4):
            world=np.array(box[:2])@frame.T+observed_pose[:2]
            along=(world-point)@tangent;lateral=(world-point)@normal
            if -.3<along<1.5 and abs(lateral)<.5 \
               and all(np.linalg.norm(world-kept)>.3 for kept in self.passed):
                self.passed.append(world)

    def _guide_blocking(self,perception,observed_pose):
        """Nearest box ahead whose center lies within GUIDE_BAND of the ridden line."""
        point,tangent=self.guide
        normal=np.array([-tangent[1],tangent[0]])
        frame=rotation(observed_pose[2])
        best=None;best_along=None
        for box in np.asarray(perception.obstacles,dtype=float).reshape(-1,4):
            world=np.array(box[:2])@frame.T+observed_pose[:2]
            along=(world-point)@tangent;lateral=(world-point)@normal
            if 0<along<GUIDE_TRIGGER and abs(lateral)<GUIDE_BAND:
                if best is None or along<best_along:best,best_along=box,along
        return best
