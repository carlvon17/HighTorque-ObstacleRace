"""Body-relative box-face measurements and a metric servo for staggered crates."""
from dataclasses import asdict, dataclass, replace
from threading import Lock

from navigation.leg_odometry import ProLegOdometry
from navigation.box_tape import BoxTape
from navigation.box_steps import BoxSteps
from navigation.box_tuning import BoxTuning
from navigation.box_body import limb_points, station_window
import cv2
import numpy as np


def rot(a):
    return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])


def wrap(a):
    return (a+np.pi)%(2*np.pi)-np.pi


def wall_headings(level, height, keep, components):
    """Measure square-box axes from vertical surfaces in each depth component."""
    if not len(components):
        return {}
    component_image=np.zeros(keep.shape,np.int32)
    component_image[keep]=components
    stride=2
    du=level[stride:-stride,2*stride:]-level[stride:-stride,:-2*stride]
    dv=level[2*stride:,stride:-stride]-level[:-2*stride,stride:-stride]
    normals=np.cross(du,dv)
    magnitude=np.linalg.norm(normals,axis=2)
    normal_z=np.abs(normals[...,2])/np.maximum(magnitude,1e-12)
    inside=(height[stride:-stride,stride:-stride]<.365)&(normal_z<.25)
    labels=component_image[stride:-stride,stride:-stride][inside]
    normals=normals[inside]
    result={}
    for label in np.unique(labels):
        if not label:
            continue
        walls=normals[labels==label]
        if len(walls)<25:
            continue
        angles=np.arctan2(walls[:,1],walls[:,0])
        votes=np.exp(4j*angles)
        # Both visible walls of a cube give the same fourth harmonic.
        # Reject components with no dominant square wall orientation.
        mean=np.mean(votes)
        if abs(mean)<.68:
            continue
        inliers=np.real(votes*np.conj(mean)/abs(mean))>.87
        if np.sum(inliers)>=20:
            result[label]=np.angle(np.mean(votes[inliers]))/4
    return result


@dataclass(frozen=True)
class BoxFrame:
    position: np.ndarray
    velocity: np.ndarray
    centers: tuple
    heading: float
    side: float
    stamp: float
    leg_anchor: object
    kick_until: float
    identified: tuple = ()
    axis_evidence: tuple = (False,False)
    tape_fix: bool = False
    tape_longitudinal: bool = False
    direct_axes: tuple = ((),())
    linked: tuple = ()
    visual_xy: tuple = (None,None)


class BoxServo:
    def __init__(self):
        self.control_lock=Lock()
        self.invalid_projection_time = -np.inf
        self.leg_lock=Lock()
        self.latest_leg=None
        self.frame=None
        self.pending_frame=None
        self.heading=None
        self.pos=np.zeros(2)
        self.velocity=np.zeros(2)
        self.boxes=[]
        self.index=0
        self.phase='enter'
        self.along_started=None
        self.side=0
        self.legs=ProLegOdometry()
        self.leg_anchor=None
        self.last_time=None
        self.last_odom=None
        self.visible=[]
        self.done=False
        self.last_cmd=np.zeros(3)
        self.target=None
        self.error=None
        self.track_mid=None
        self.corrections=[]
        self.travel=0.
        self.stationary=0.
        self.kick_until=0.
        self.tape=BoxTape()
        self.steps=BoxSteps()
        self.tuning=BoxTuning()
        self.tuning_revision=0
        self.pending_tuning=None
        self.panel_gap=None
        self.release_requested=False
        self.hold_reason='initializing'
        self.step_count=0
        self.entry_station=None
        self.covisible=set()
        self.forward_still_since=None
        self.restart_until=0.
        self.body_sample=None
        self.station_bounds=None
        self.approached=False
        self.body_history=[]

    def observe(self, frame, attitude, camera, tracked, origin_yaw, lane):
        yaw=np.arctan2(attitude[1,0],attitude[0,0])
        plane=camera.ground.result
        if self.done:return
        if self.frame is not None:
            self.pending_frame=replace(self.frame,stamp=frame.time,identified=(),axis_evidence=(False,False),
                                       tape_fix=False,tape_longitudinal=False,direct_axes=((),()),linked=(),visual_xy=(None,None))
        xyz=(camera.rays*frame.depth[...,None])@camera.rotation32.T+camera.position32
        level=xyz@attitude.T
        if not plane.valid:return
        height=plane.height(level)
        keep=np.isfinite(height)&(height>.065)&(height<.48)&(xyz[...,0]>.15)&(frame.depth<2.6)
        points3=level[keep]
        grid=np.zeros((240,240),np.uint8)
        cells=np.clip(np.floor(points3[:,:2]/.025+120).astype('int32'),0,239)
        grid[cells[:,1],cells[:,0]]=1
        n,labels=cv2.connectedComponents(cv2.dilate(grid,np.ones((3,3),np.uint8)))
        groups=labels[cells[:,1],cells[:,0]]
        headings=wall_headings(level,height,keep,groups) if self.heading is None else {}
        clouds=[]
        for j in range(1,n):
            points=points3[groups==j]
            if len(points)<25:continue
            pts=points[:,:2]
            if np.max(np.ptp(pts,axis=0))>.39 or np.ptp(points[:,2])<.055:continue
            rect=cv2.minAreaRect(pts[::max(1,len(pts)//1300)].astype('float32'))
            if max(rect[1])<.078:continue
            axes=cv2.boxPoints(rect);v=axes[1]-axes[0]
            a=np.arctan2(v[1],v[0])
            if self.heading is None:
                if j not in headings:
                    continue
                a=headings[j]
            a=yaw+((a-yaw+np.pi/4)%(np.pi/2)-np.pi/4)
            clouds.append((pts,a))
        if self.heading is None:
            if not clouds:return
            candidates=[(pts,a) for pts,a in clouds if np.linalg.norm(np.median(pts,axis=0))<1.60 and abs(wrap(a-yaw))<.70]
            if not candidates:return
            pts,self.heading=min(candidates,key=lambda c:np.linalg.norm(np.median(c[0],axis=0)))
        basis=rot(self.heading)
        odom=tracked.xy@rot(origin_yaw).T@basis
        dt=0. if self.last_time is None else max(0.,frame.time-self.last_time)
        delta=np.zeros(2) if self.last_odom is None else odom-self.last_odom
        if dt and (not tracked.odometry.accepted or np.linalg.norm(delta)>dt*.7):
            delta=self.velocity*dt
        last_anchor=self.leg_anchor
        with self.leg_lock:
            self.leg_anchor=self.legs.at(frame.time)
        if last_anchor is not None and self.leg_anchor is not None:
            delta=(self.leg_anchor-last_anchor)@basis
        old=self.pos.copy();self.pos+=delta
        self.last_odom=odom;self.last_time=frame.time
        measured=[]
        lens=(camera.position32@attitude.T)[:2]@basis
        for pts,a in clouds:
            p=pts@basis
            lo,hi=np.percentile(p,[1,99],axis=0)
            if np.any(hi-lo>.255):continue
            # A square gives its longitudinal center directly from a front face.
            # Use the toe-facing lateral side when the outline is clipped.
            center=(lo+hi)/2
            center[0]=lo[0]+.10
            if hi[1]-lo[1]<.17:
                center[1]=lo[1]+.10 if center[1]>lens[1] else hi[1]-.10
            measured.append(center)
        hits=[];new=[]
        for rel in measured:
            p=self.pos+rel
            errors=[np.linalg.norm(b-p) for b in self.boxes]
            if errors and min(errors)<.24:
                i=int(np.argmin(errors));hits.append((i,rel,self.boxes[i]-p))
            elif not self.boxes or rel[0]>.40:
                new.append(rel)
        # Use only measured vertical face normals. A clipped top or a visible
        # rear slat is not a measurement of the near face of the cube.
        depth_points=level.copy()
        depth_points[...,:2]=level[...,:2]@basis
        step=2
        ddx=depth_points[step:-step,2*step:]-depth_points[step:-step,:-2*step]
        ddy=depth_points[2*step:,step:-step]-depth_points[:-2*step,step:-step]
        norm=np.cross(ddx,ddy); mag=np.linalg.norm(norm,axis=2)
        norm=norm/np.maximum(mag[...,None],1e-12)
        wall=keep[step:-step,step:-step]&(height[step:-step,step:-step]<.365)
        xy=depth_points[step:-step,step:-step,:2]
        floor_mask=(np.isfinite(height)&(abs(height)<.035)).astype(np.uint8)
        rim_mask=(height>.365)&(height<.44)&cv2.dilate(floor_mask,np.ones((5,5),np.uint8)).astype(bool)
        rim=depth_points[rim_mask,:2]
        self.corrections=[]
        face_ids=[set(),set()]
        visual_xy=[None,None]
        for axis in range(2):
            surf=xy[wall&(abs(norm[...,axis])>.91)]
            err=[]
            for idx,box in enumerate(self.boxes):
                predicted=box-self.pos
                face=predicted[axis]+np.sign(lens[axis]-predicted[axis])*.1
                dist=surf[:,axis]-face
                match=(abs(dist)<.075)&(abs(surf[:,1-axis]-predicted[1-axis])<.13)
                vals=dist[match]
                if len(vals)<12:continue
                bins=np.round(vals/.004).astype(int)
                modal=np.argmax(np.bincount(bins+20))-20
                near=vals[abs(vals-.004*modal)<.008]
                if len(near)>=10:
                    err.append(float(np.median(near)))
                    face_ids[axis].add(idx)
            if not err:
                # At close range the head sees the top outline after losing
                # the front wall. A rim touching observed floor supplies an
                # independent metric edge even during a lateral crossing.
                for idx,box in enumerate(self.boxes):
                    predicted=box-self.pos
                    for sign in (-1.,1.):
                        face=predicted[axis]+sign*.1
                        dist=rim[:,axis]-face
                        ids=(abs(dist)<.052)&(abs(rim[:,1-axis]-predicted[1-axis])<.106)
                        ds=dist[ids]
                        if len(ds)<9:continue
                        bins=np.rint(ds/.003).astype(int)+20
                        peak=(np.argmax(np.bincount(bins))-20)*.003
                        on=abs(ds-peak)<.007
                        if on.sum()>=9 and np.ptp(rim[ids,1-axis][on])>.046:
                            err.append(float(np.median(ds[on])))
                            face_ids[axis].add(idx)
            if err:
                visual_xy[axis]=float(self.pos[axis]-np.median(err))
                maximum,gain=(.038,.85) if axis==0 else (.018,.58)
                self.pos[axis]-=np.clip(np.median(err),-maximum,maximum)*gain
            self.corrections.append(err)
        line_err=[]
        floor=np.empty((0,2))
        if self.track_mid is not None and plane.valid:
            rgb=frame.rgb
            white=(abs(height)<.02)&(rgb.min(axis=2)>115)&(rgb.max(axis=2)-rgb.min(axis=2)<.15*rgb.max(axis=2))
            floor=depth_points[white,:2]
            center=self.track_mid-self.pos[1]
            for v in (center-.25,center+.25):
                pick=floor[(abs(floor[:,1]-v)<.065)&(floor[:,0]<1.1)&(floor[:,0]>.22)]
                if len(pick)>15 and np.ptp(pick[:,0])>.18:
                    # Exclude the start of a curve and transverse tape.
                    fit=np.polyfit(pick[:,0],pick[:,1],1)
                    if abs(fit[0])<.045 and np.percentile(abs(pick[:,1]-v-np.median(pick[:,1]-v)),85)<.018:
                        line_err.append(np.median(pick[:,1]-v))
            if line_err:
                if visual_xy[1] is None:visual_xy[1]=float(self.pos[1]-np.median(line_err))
                self.pos[1]-=.55*np.clip(np.median(line_err),-.018,.018)
        self.visible=[]
        for i,rel,_ in hits:self.visible.append(i)
        for rel in sorted(new,key=lambda p:p[0]):
            p=self.pos+rel
            if self.boxes and min(np.linalg.norm(b-p) for b in self.boxes)<.23:continue
            if self.boxes and p[0]<self.boxes[0][0]-.26 and self.index==0 and self.phase in ('enter','along'):
                self.boxes.insert(0,p);self.side=0
                self.covisible={(a+1,b+1) for a,b in self.covisible}
                self.visible=[v+1 for v in self.visible]+[0]
                face_ids=[{j+1 for j in ids} for ids in face_ids]
                continue
            if self.boxes and p[0]<self.boxes[-1][0]+.26:continue
            self.boxes.append(p);self.visible.append(len(self.boxes)-1)
        if not self.boxes:self.heading=None;return
        identified=set(self.visible)
        for idx,rel,err in hits:
            # An ID is usable as a visual anchor only within its own metric
            # footprint. The looser association above remains for tracking.
            if np.linalg.norm(err)>.105:
                identified.discard(idx)
        x_seen,y_seen=(bool(group) for group in self.corrections)
        current_reference=identified | (face_ids[0]&face_ids[1])
        implied=[self.boxes[idx]-rel for idx,rel,_ in hits if idx in identified]
        if implied:
            observed_position=np.median(implied,axis=0)
            for axis in range(2):
                if visual_xy[axis] is None:visual_xy[axis]=float(observed_position[axis])
        self.covisible.update((a,b) for a in current_reference for b in current_reference
                              if a!=b and abs(self.boxes[a][1]-self.boxes[b][1])<=.55)
        connected={self.index}
        for _ in self.boxes:
            connected.update(b for a,b in self.covisible if a in connected)
        tape_correction,tape_x=self.tape.match(floor,self.pos, bool(identified) and x_seen)
        if tape_correction is not None and not identified:
            if not x_seen and tape_x:
                visual_xy[0]=float(self.pos[0]+tape_correction[0])
                self.pos[0]+=.28*tape_correction[0]
            if not (y_seen or line_err):
                visual_xy[1]=float(self.pos[1]+tape_correction[1])
                self.pos[1]+=.28*tape_correction[1]
        if self.track_mid is None and len(self.boxes)>1:
            self.track_mid=(self.boxes[0][1]+self.boxes[1][1])/2
        if not self.side or (self.index==0 and self.phase in ('enter','along')):
            if len(self.boxes)>1:
                self.side=np.sign(self.boxes[1][1]-self.boxes[0][1])
            elif lane is not None:
                p=(self.boxes[0]-self.pos)@basis.T@rot(yaw)
                self.side=np.sign(np.polyval(lane,p[0])-p[1])
        if dt:
            v=(self.pos-old)/dt
            self.velocity=.56*self.velocity+.44*np.clip(v,-.50,.50)
            self.travel+=np.linalg.norm(self.pos-old)
        if self.index==0 and self.travel<.08 and self.last_cmd[0]>.06 and abs(self.last_cmd[1])<.16:
            self.stationary+=dt
            if self.stationary>.5:
                self.kick_until=frame.time+.45;self.stationary=-.65

        # The real publisher runs concurrently with RGB-D processing. Publish
        # a complete capture result by one reference assignment; its arrays
        # are independent of the mutable next-frame tracker.
        self.pending_frame=BoxFrame(self.pos.copy(),self.velocity.copy(),tuple(self.boxes),self.heading,
                            self.side,self.last_time,self.leg_anchor,self.kick_until,
                            tuple(sorted(identified)),(x_seen,y_seen),bool(line_err) or tape_correction is not None,bool(tape_x),
                            tuple(tuple(sorted(v)) for v in face_ids),tuple(sorted(connected&current_reference)),tuple(visual_xy))

    def commit_frame(self):
        if self.pending_frame is not None:
            with self.control_lock:
                self.frame,self.pending_frame=self.pending_frame,None

    def update_legs(self,t,positions,attitude):
        with self.leg_lock:
            accepted=self.legs.update(t,positions,attitude)
            if accepted:
                self.latest_leg=(t,self.legs.xy.copy(),self.legs.velocity.copy())
                points=limb_points(positions,attitude)
                if points is not None:
                    self.body_history=[prev for prev in self.body_history if t-prev[0]<.22]
                    self.body_history.append((t,points))
                    self.body_sample=(t,np.vstack([prev[1] for prev in self.body_history]))
            return accepted

    def command(self,t,yaw):
        with self.control_lock:
            return self._command(t,yaw,self.frame)

    def publication(self,frame,deadline):
        """Bind the box measurements and their expiry to this published plan."""
        def velocity(stamp,planned,attitude):
            if stamp>=deadline:
                return np.zeros(3)
            if not self.control_lock.acquire(blocking=False):
                return np.zeros(3)
            try:
                # A planning thread may already have advanced to its next
                # observation. Never advance that plan using this older frame.
                if self.frame is not frame:
                    return np.zeros(3)
                yaw=np.arctan2(attitude[1,0],attitude[0,0])
                guided=self._command(stamp,yaw,frame)
                return np.zeros(3) if guided is None else guided
            finally:
                self.control_lock.release()
        def published(stamp,vector):
            with self.control_lock:
                self.steps.published(stamp,vector)
        velocity.published=published
        return velocity

    def queue_tuning(self, revision, values):
        settings=BoxTuning.read(values)
        with self.control_lock:
            if revision<=self.tuning_revision or (self.pending_tuning and revision<=self.pending_tuning[0]):
                return
            self.pending_tuning=(revision,settings)
            if self.frame is None or self.done:
                self._adopt_tuning()

    def _adopt_tuning(self):
        if self.pending_tuning is not None:
            self.tuning_revision,self.tuning=self.pending_tuning
            self.pending_tuning=None

    def panel_status(self, now):
        with self.control_lock:
            frame=self.frame
            fresh=(frame is not None and frame.stamp>self.invalid_projection_time and 0<=now-frame.stamp<=.50)
            gap=self.panel_gap
            gate=self.hold_reason if fresh else 'no_current_geometry'
            if fresh and self.steps.active is not None and gate in ('confirmed_path','lateral_step'):
                gate=self.steps.reason
            return dict(applied=asdict(self.tuning),revision=self.tuning_revision,
                        phase=self.phase,index=self.index,done=self.done,started=bool(frame and frame.centers),
                        gate=gate,
                        gap=gap[1] if fresh and gap and gap[0]==frame.stamp else None,
                        count=self.steps.count,moving=self.steps.active is not None and fresh)

    def invalidate_projection(self, stamp):
        with self.control_lock:
            self.invalid_projection_time = max(self.invalid_projection_time, stamp)

    def _command(self,t,yaw,frame):
        if frame is not None and frame.centers and not self.done and frame.stamp <= self.invalid_projection_time:
            self.last_cmd = np.zeros(3)
            return self.last_cmd.copy()
        if frame is None or not frame.centers or self.done or self.index>=len(frame.centers):return None
        if not frame.side:
            self.hold_reason='no_passage_side'
            return np.zeros(3)
        i=self.index
        center=frame.centers[i]
        leg=self.latest_leg
        vel=frame.velocity
        if frame.leg_anchor is not None and leg is not None and 0<=t-leg[0]<=.10:
            basis=rot(frame.heading)
            vel=leg[2]@basis
            pos=frame.position+(leg[1]-frame.leg_anchor)@basis+vel*(t-leg[0])
        else:
            pos=frame.position+frame.velocity*min(.30,max(0.,t-frame.stamp))
        reason=self.steps.check(t,frame,pos)
        if reason:
            self.hold_reason=reason
            return np.zeros(3)
        if self.steps.active is None:
            self._adopt_tuning()
        phase=(self.phase,self.index,self.approached)
        vector=self._drive(t,yaw,frame,pos,vel)
        if self.release_requested and self.steps.active is None:
            self.done=True
            return None
        if phase!=(self.phase,self.index,self.approached) or self.release_requested:
            self.steps.stop(t)
            self.hold_reason='phase_observation'
            return np.zeros(3)
        if vector is None:return None
        local=vector[:2]@rot(yaw).T@rot(frame.heading)
        axis=1 if self.hold_reason=='lateral_step' else 0
        if np.linalg.norm(local)<.015 and abs(vector[2])>.02:axis=2
        sign=np.sign(local[axis]) if axis<2 else np.sign(vector[2])
        step_cm=self.tuning.side_step if axis==1 else self.tuning.forward_step if axis==0 else 6.3
        vector=self.steps.apply(t,frame,pos,vector,axis,sign,step_cm/100.)
        if self.steps.reason and self.steps.active is None and np.all(vector==0):
            self.hold_reason=self.steps.reason
        self.step_count=self.steps.count
        self.last_cmd=vector.copy()
        return vector

    def _drive(self,t,yaw,frame,pos,vel):
        i=self.index
        center=frame.centers[i]
        side=frame.side*(-1 if i%2 else 1)
        entry_gap=self.tuning.first_gap/100. if i==0 else .39
        pass_offset=round(entry_gap+.03,4)
        y=center[1]+side*pass_offset
        x_seen,y_seen=frame.axis_evidence
        box_anchor=bool(frame.identified) or (x_seen and y_seen)
        tape_anchor=frame.tape_fix and frame.tape_longitudinal
        if self.phase=='enter':
            direct=(i in frame.identified) or (i in frame.direct_axes[0] and (i in frame.direct_axes[1] or frame.tape_fix))
            evidence=direct or (bool(frame.linked) and x_seen and (y_seen or frame.tape_fix))
        else:
            evidence=(box_anchor and (y_seen or frame.tape_fix or bool(frame.identified))) or tape_anchor
        if not evidence or t-frame.stamp>.50:
            self.hold_reason='no_current_geometry'
            self.last_cmd=np.zeros(3)
            return self.last_cmd.copy()
        self.panel_gap=None
        if frame.visual_xy[1] is not None:
            side_gap=min(side*(pos[1]-center[1]),side*(frame.visual_xy[1]-center[1]))
            self.panel_gap=(frame.stamp,float(side_gap*100.))
        if self.phase=='enter' and not self.approached:
            measured_x=frame.visual_xy[0]
            if measured_x is None:
                self.hold_reason='wait_range'
                return np.zeros(3)
            # Stop before the near face while both feet and hands still
            # have room to settle. At this distance the head can see the
            # lateral face while the body aligns with the crate corridor.
            clearance=center[0]-max(pos[0],measured_x)-.10
            angle=wrap(frame.heading-yaw)
            if clearance>.69 or abs(angle)>.17:
                vx=np.clip(1.8*(clearance-.65)-.55*vel[0],0.,.20)
                if clearance<.68:vx=0.
                body=np.array([vx,0.])@rot(frame.heading).T@rot(yaw)
                w=np.clip(2.4*angle,-.57,.57)
                if vx>.12 and abs(vel[0])<.045 and abs(w)<.16:w=.23
                self.hold_reason='observe_first_from_standoff'
                self.last_cmd=np.r_[body,w]
                return self.last_cmd.copy()
            self.approached=True
            self.last_cmd=np.zeros(3)
            return self.last_cmd.copy()
        side_gap=side*(pos[1]-center[1])
        if frame.visual_xy[1] is not None:
            side_gap=min(side_gap,side*(frame.visual_xy[1]-center[1]))
        lateral_step=self.phase in ('shift','lane_entry') or (self.phase=='enter' and side_gap<entry_gap)
        if lateral_step:
            if self.steps.active is None:self.entry_station=float(pos[0])
        elif self.phase in ('along','enter') and (frame.visual_xy[1] is None or side_gap<round(entry_gap-.03,4)):
            self.hold_reason='lateral_clearance'
            self.last_cmd=np.zeros(3)
            return self.last_cmd.copy()
        self.hold_reason='lateral_step' if lateral_step else 'confirmed_path'
        if i+1<len(frame.centers):
            gate=(center[0]+frame.centers[i+1][0])/2-.037
        else:
            gate=center[0]+.38
        if self.phase=='enter':
            target_x=self.entry_station if lateral_step else center[0]-.36
        elif self.phase=='along':
            target_x=gate
        elif self.phase=='lane_entry':
            target_x=gate+.04
            y=.5*(frame.centers[-2][1]+frame.centers[-1][1])
        else:
            y=frame.centers[i+1][1]-side*.42
            # As the torso moves across the lane, the trailing outside foot
            # approaches the previous crate. Advance through the midpoint
            # gradually after the leading arm has crossed the next crate.
            total=abs(y-(center[1]+side*pass_offset))
            fraction=np.clip(1.-abs(y-pos[1])/total,0.,1.)
            target_x=gate+.079*np.clip((fraction-.25)/.54,0.,1.)
            body=self.body_sample
            if body is not None and 0<=t-body[0]<=.12:
                lo,hi=station_window(body[1],rot(frame.heading),pos,y,center,frame.centers[i+1])
                self.station_bounds=(float(np.clip(lo,-10.,10.)),float(np.clip(hi,-10.,10.)))
                if np.isfinite(lo) and np.isfinite(hi):
                    target_x=.5*(lo+hi)
                else:
                    target_x=np.clip(target_x,lo+.023,hi-.028)
        dx=target_x-pos[0];lateral=y-pos[1]
        vx0,vy0=vel
        ahead=dx-.55*vx0
        across=lateral-.33*vy0
        vx=np.clip(1.7*ahead,-.18,self.tuning.forward_speed/100.)
        side_limit=self.tuning.side_speed/100.
        vy=np.clip(2.3*across,-side_limit,side_limit)
        if lateral_step:
            if abs(lateral)>.028 and vy*lateral>=0 and abs(vy)<.19:
                vy=np.copysign(min(.19,side_limit),lateral)
            vx=np.clip(3*(dx-.39*vx0)-.088*min(1.,abs(vy)/.15),-.20,.13)
        elif abs(ahead)<.020:vx=0.
        if self.phase=='along' and abs(vy)<.035 and 0.<vx<.15 and dx<.12:
            # The stock velocity adapter otherwise rounds every positive
            # straight request up to .20. Govern the final pulse explicitly.
            elapsed_us=round((t-self.along_started)*1_000_000)
            pulse_on=elapsed_us%300_000<105_000
            vx=.20 if dx>.055 and ahead>.045 and pulse_on else 0.
        self.target=[float(target_x),float(y)];self.error=[float(dx),float(lateral)]
        prior_phase=self.phase
        if prior_phase=='enter' and not lateral_step and abs(lateral)<.035 and dx<.065:
            self.phase='along'
            self.along_started=t
        elif (prior_phase=='along' and abs(dx)<.052 and abs(vx0)<.09 and
              frame.visual_xy[0] is not None and abs(target_x-frame.visual_xy[0])<.064):
            if i+1<len(frame.centers):self.phase='shift'
            elif len(frame.centers)>=3:
                self.phase='lane_entry'
        elif (prior_phase=='shift' and abs(lateral)<.035 and abs(vy0)<.11 and
              frame.visual_xy[1] is not None and abs(y-frame.visual_xy[1])<.060):
            self.phase='along'
            self.along_started=t
            self.index+=1
        elif prior_phase=='lane_entry' and abs(lateral)<.04 and abs(vy0)<.10 and pos[0]>center[0]+.30:
            self.release_requested=True
            return np.zeros(3)
        if self.phase!=prior_phase:
            self.hold_reason='phase_observation'
            self.last_cmd=np.zeros(3)
            return self.last_cmd.copy()
        desired_heading=frame.heading
        if self.phase=='shift':
            # Yaw toward the side just used to pass the previous crate.
            # This draws the upcoming-side wrist back and moves the trailing
            # foot away from the previous crate while traversing the neck.
            desired_heading+=side*.24
        elif self.phase=='along' and i+1<len(frame.centers):
            desired_heading+=side*.24*np.clip(1.-max(0.,dx)/.24,0.,1.)
        heading_error=wrap(desired_heading-yaw)
        body=np.array([vx,vy])@rot(frame.heading).T@rot(yaw)
        w=np.clip(2.4*heading_error,-.5,.5)
        if t<frame.kick_until and abs(w)<.20:w=.23
        if self.phase in ('along','enter') and not lateral_step and vx>.17 and abs(vx0)<.055:
            if self.forward_still_since is None:self.forward_still_since=t
            if t-self.forward_still_since>.60:
                self.restart_until=t+.56
                self.forward_still_since=t+.75
        else:self.forward_still_since=None
        if not lateral_step and t<self.restart_until and abs(w)<.21:
            w=.24 if heading_error>=0 else -.24
        self.last_cmd=np.r_[body,w]
        return self.last_cmd.copy()

    def diagnostic(self):
        return dict(crate_phase=self.phase,crate_index=self.index,crate_side=float(self.side),
                    crate_gate=self.hold_reason,crate_steps=self.step_count,
                    crate_evidence=None if self.frame is None else [self.frame.identified,self.frame.axis_evidence,self.frame.tape_fix,self.frame.tape_longitudinal],
                    crate_visual_xy=None if self.frame is None else self.frame.visual_xy,
                    crate_station_bounds=self.station_bounds,crate_zero_stamp=self.steps.zero_stamp,
                    crate_step_until=None if self.steps.active is None else self.steps.active['until'],
                    crate_reference=None if self.frame is None else [self.frame.direct_axes,self.frame.linked],
                    crate_pose=self.pos.tolist(),crate_velocity=self.velocity.tolist(),crate_heading=self.heading,
                    crate_centers=[v.tolist() for v in self.boxes],crate_visible=self.visible,
                    crate_residuals=self.corrections,crate_target=self.target,crate_error=self.error,crate_done=self.done)
