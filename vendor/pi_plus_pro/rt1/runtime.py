"""Identical sensor-to-command pipeline for simulation, replay and ROS.

All positions are estimated from RGB-D. Attitude is world_from_body from an
AHRS measurement. No world position or simulator object data is accepted.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from navigation.command import AMPCommandAdapter
from navigation.environment import verify_dependencies
from navigation.floor_map import FloorMap
from navigation.motion_recovery import MotionRecovery
from .visual import CameraPerception, Perception, RGBDOdometry, VisualNavigator, rotation2


@dataclass(frozen=True)
class CameraFrame:
    time: float
    rgb: np.ndarray
    depth: np.ndarray


def core_hashes():
    root = Path(__file__).parent
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.iterdir()) if p.suffix in ('.py', '.json')}


class NavigationRuntime:
    def __init__(self, width, height, intrinsics, camera_position, camera_rotation, speed=.22, observation_lease=.35):
        verify_dependencies()
        if not np.isfinite(speed) or not 0 < speed <= .4:
            raise ValueError('Navigation target speed must be within (0, 0.4]')
        if not np.isfinite(observation_lease) or not 0 < observation_lease <= 2.:
            raise ValueError('Observation lease must be within (0, 1] s')
        self.observation_lease = float(observation_lease)
        self.position = np.asarray(camera_position, dtype=float)
        self.camera_rotation = np.asarray(camera_rotation, dtype=float)
        if self.position.shape != (3,) or not np.isfinite(self.position).all():
            raise ValueError('Invalid camera position')
        self._rotation(self.camera_rotation)
        self.perception = CameraPerception(width, height, 80., self.position, self.camera_rotation, intrinsics)
        self.odometry = RGBDOdometry(width, height, 80., self.position, self.camera_rotation, intrinsics)
        self.floor_map = FloorMap(self.odometry.floor)
        self.navigator = VisualNavigator(speed, self.observation_lease)
        self.adapter = AMPCommandAdapter()
        self.initial_yaw = None
        self.yaw = 0.
        self.turn = 0.
        self.distance = 0.
        self.xy = np.zeros(2)
        self.attitude = np.eye(3)
        self.last_imu_time = None
        self.last_frame_time = None
        self.last_lane_observation_time = None
        self.rgb_usable = False
        self.last_control_time = None
        self.observation = None
        self.good_frames = 0
        self.reason = 'initializing'
        self.desired = np.zeros(3)
        self.lost_since = None
        self.localization_reseeds = 0
        self.motion_recovery = MotionRecovery()

    @staticmethod
    def _rotation(value):
        if (value.shape != (3, 3) or not np.isfinite(value).all() or
                not np.allclose(value.T@value, np.eye(3), atol=1e-4) or
                not np.isclose(np.linalg.det(value), 1, atol=1e-4)):
            raise ValueError('Expected a proper rotation matrix')

    @property
    def pose(self):
        return np.r_[self.xy, self.yaw]

    def set_attitude(self, timestamp, world_from_body):
        matrix = np.asarray(world_from_body, dtype=float)
        self._rotation(matrix)
        if not np.isfinite(timestamp) or (self.last_imu_time is not None and timestamp < self.last_imu_time):
            raise ValueError('Nonmonotonic IMU timestamp')
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
        if self.initial_yaw is None:
            self.initial_yaw = yaw
        initial = np.eye(3)
        initial[:2, :2] = rotation2(-self.initial_yaw)
        self.attitude = initial@matrix
        yaw = np.arctan2(self.attitude[1, 0], self.attitude[0, 0])
        self.turn += (yaw-self.yaw+np.pi) % (2*np.pi)-np.pi
        self.yaw = float(yaw)
        self.last_imu_time = float(timestamp)

    def _observed_map_white(self, timestamp):
        """Keep a stationary lane reference available while visual motion is lost.

        A corridor is created only from repeated observed straight lane fits near
        boxes. Its direction and center are measurements; the 50 cm width is the
        same task dimension used by perception and planning, not a field pose.
        """
        heading = self.navigator.corridor_heading
        point = self.navigator.corridor_point
        alongside = any(-.4 < local[0] < .85 and abs(local[1]) < .8
                        for local in ((box['center']-self.xy)@rotation2(self.yaw)
                                      for box in self.perception.obstacle_memory))
        if (alongside and heading is not None and point is not None
                and np.isfinite(heading) and np.isfinite(point).all()):
            tangent = np.array([np.cos(heading), np.sin(heading)])
            normal = np.array([-tangent[1], tangent[0]])
            # Recenter the finite reference along the corridor as the robot
            # advances. Keep its measured perpendicular offset unchanged.
            center = point+tangent*((self.xy-point)@tangent)-self.xy
            along = np.linspace(-.4, 2.7, 311)
            return np.vstack([center+along[:, None]*tangent+side*normal
                              for side in (-.25, .25)])
        # Once tracking is lost no observations are fused. Expiring the last
        # accepted lines by wall-clock time alone would prevent later recovery
        # even when the robot has stopped at precisely the same place.
        reference_time = timestamp if self.lost_since is None else self.lost_since
        whites = [points for stamp, points, _ in self.perception.white_memory
                  if reference_time-stamp < 8.]
        return np.vstack(whites)-self.xy if whites else np.empty((0, 2))

    def observe(self, frame, world_from_body_at_frame=None, imu_time=None):
        stamp = self.last_imu_time if imu_time is None else imu_time
        if self.last_imu_time is None or stamp is None or abs(frame.time-stamp) > .15:
            raise ValueError('No time-aligned attitude for camera frame')
        frame_attitude = self.attitude
        if world_from_body_at_frame is not None:
            matrix = np.asarray(world_from_body_at_frame, dtype=float)
            self._rotation(matrix)
            initial = np.eye(3)
            initial[:2, :2] = rotation2(-self.initial_yaw)
            frame_attitude = initial@matrix
        if self.last_frame_time is not None and frame.time <= self.last_frame_time:
            raise ValueError('Nonmonotonic camera timestamp')
        if frame.depth.shape != self.perception.rays.shape[:2] or frame.rgb.shape != (*frame.depth.shape, 3):
            raise ValueError('Camera resolution differs from calibrated intrinsics')
        usable_pixels = (frame.rgb.max(axis=2)>5) & (frame.rgb.min(axis=2)<250)
        self.rgb_usable = bool(np.count_nonzero(usable_pixels) >= frame.depth.size*.01)
        if self.last_frame_time is not None and frame.time-self.last_frame_time > self.observation_lease:
            self.good_frames = 0
        tracking_frame = frame if self.rgb_usable else CameraFrame(frame.time, np.zeros_like(frame.rgb), frame.depth)
        xy = self.odometry.update(tracking_frame, frame_attitude, self.xy)
        if not self.odometry.accepted and self.perception.obstacle_memory:
            # Relocalize against geometry previously measured by this robot.
            # A partial/blank keyframe must not make a known box disappear.
            white=self._observed_map_white(frame.time)
            near_edges=[];far_edges=[]
            along=np.linspace(-.10,.10,31)
            for box in self.perception.obstacle_memory:
                axes=box.get('axes',np.eye(2))
                for axis in (0,1):
                    for side in (-.10,.10):
                        local=np.zeros((len(along),2));local[:,axis]=side;local[:,1-axis]=along
                        points=local@axes+box['center']-self.xy
                        normal=np.sign(side)*axes[axis]
                        (near_edges if np.mean(points,axis=0)@normal<0 else far_edges).append(points)
            geometry=self.odometry.geometry
            reference=[geometry.sample(white),*[geometry.sample(np.vstack(edges)) if edges else np.empty((0,2),np.float32) for edges in (near_edges,far_edges)]]
            measured=geometry.project(tracking_frame,frame_attitude)
            correction=geometry.register(reference,measured)
            if correction is not None and np.linalg.norm(correction)<.09:
                xy=self.xy+correction
                self.odometry.accepted=True
                self.odometry.inliers=geometry.support
                self.odometry.method='observed_map'
                self.odometry.previous=None
                # Seed the next image tracker at the recovered metric pose.
                self.odometry.update(tracking_frame,frame_attitude,xy)
                self.odometry.accepted=True
                self.odometry.inliers=geometry.support
                self.odometry.method='observed_map'
                self.odometry.relocalized=True
        # Search recent textured keyframes after an interruption, before the
        # local tracker gives up its segment. Hidden boxes stay in odom.
        previous_time = self.odometry.previous_time
        if (not self.odometry.accepted and self.rgb_usable and previous_time is not None
                and frame.time-previous_time > .25):
            recovered = self.floor_map.update(frame, frame_attitude,
                self.odometry.geometry.ground.result, self.xy, self.distance, recovering=True)
            if recovered is not None:
                xy = recovered
                self.odometry.previous = None
                self.odometry.update(frame, frame_attitude, xy)
                self.odometry.accepted = True
                self.odometry.relocalized = True
                self.odometry.method = 'floor_keyframe'
        if self.odometry.accepted:
            self.lost_since = None
        else:
            if self.lost_since is None:self.lost_since=frame.time
            # In an empty corridor, restart a purely local coordinate segment
            # after stopping. Never move/reseed a map containing hidden boxes
            # or a tracked finish target; their relative positions matter.
            if (frame.time-self.lost_since>.6 and not self.perception.obstacle_memory
                    and self.navigator.finish_target is None and not np.any(self.adapter.target)):
                self.odometry.previous=None
                self.odometry.previous_time=None
                self.perception.white_memory.clear()
                self.perception.previous_lane=None
                self.lost_since=frame.time
                self.localization_reseeds+=1
                self.floor_map.start_segment()
        self.distance += float(np.linalg.norm(xy-self.xy))
        self.xy = xy
        frame_pose = np.r_[self.xy, np.arctan2(frame_attitude[1, 0], frame_attitude[0, 0])]
        if self.odometry.accepted and self.rgb_usable:
            self.floor_map.update(frame, frame_attitude, self.odometry.geometry.ground.result,
                                  self.xy, self.distance)
            self.observation = self.perception.update(frame, frame_pose, frame_attitude,
                ground_plane=self.odometry.geometry.ground.result)
            if self.observation.lane is not None and self.observation.lane_observed:
                self.last_lane_observation_time = float(frame.time)
            self.navigator.gap.translate_map(getattr(self.perception,'map_translation',np.zeros(2)))
        else:
            # Neither erase hidden boxes nor fuse pixels at an unknown pose.
            # Retain the accepted reference, inhibit motion, and recover first.
            obstacles = []
            for record in self.perception.obstacle_memory:
                local = (record['center']-self.xy)@rotation2(frame_pose[2])
                obstacles.append([*local, *record['size']])
            self.observation = Perception(frame.time, None, np.empty((0,2)),
                np.asarray(obstacles).reshape(-1,4), np.empty((0,2)), 0., frame.rgb.copy())
        if self.odometry.relocalized:
            # A pose correction is not physical body velocity. Reacquire
            # three trusted images before resuming the local motion servo.
            self.good_frames=0
            self.navigator.velocity_pose=frame_pose[:2].copy()
            self.navigator.velocity_time=frame.time
            self.navigator.measured_velocity[:]=0.
        self.navigator.observe(self.observation, frame_pose, self.turn,
                               motion_observed=self.odometry.accepted and not self.odometry.relocalized)
        has_route = self.observation.lane is not None or self.navigator.gap.phase!='idle'
        valid = has_route and self.odometry.accepted and self.rgb_usable
        self.good_frames = self.good_frames+1 if valid else 0
        self.last_frame_time = float(frame.time)
        return self.observation

    def command(self, timestamp, policy_feedback=None, enabled=True):
        if not np.isfinite(timestamp) or (self.last_control_time is not None and timestamp <= self.last_control_time):
            raise ValueError('Nonmonotonic control timestamp')
        dt = .02 if self.last_control_time is None else min(.1, timestamp-self.last_control_time)
        self.last_control_time = float(timestamp)
        if not enabled:
            self.reason = 'disabled'
        elif self.last_imu_time is None or not 0 <= timestamp-self.last_imu_time <= .15:
            self.reason = 'stale_imu'
        elif self.last_frame_time is None or not 0 <= timestamp-self.last_frame_time <= self.observation_lease:
            self.reason = 'stale_camera'
        elif not self.rgb_usable:
            self.reason = 'unusable_rgb'
        elif self.good_frames < 3:
            self.reason = 'waiting_for_trusted_visual_motion_and_lane'
        else:
            self.desired = self.navigator.update(timestamp, self.pose, dt)
            self.reason = (self.navigator.mode if self.navigator.mode in
                ('lost_vision', 'blocked', 'finished', 'initializing') else 'ready')
        reliable_motion=(self.navigator.velocity_valid and self.navigator.velocity_time is not None
                         and 0<=timestamp-self.navigator.velocity_time<=self.observation_lease)
        recovery=self.motion_recovery.update(timestamp,self.desired,self.pose,
            self.navigator.measured_velocity,reliable_motion and self.reason=='ready' and self.navigator.gap.phase=='idle')
        if reliable_motion and self.reason=='ready' and self.navigator.gap.phase=='idle':
            if recovery=='settle':self.reason='standing_recovery'
            elif recovery=='failed':
                self.reason='motion_stalled';self.navigator.mode='blocked'
        if self.reason != 'ready':
            self.desired[:] = 0.
            self.navigator.command[:] = 0.
        return self.adapter.update(self.desired, policy_feedback, self.reason == 'ready')

    def diagnostic(self):
        return dict(reason=self.reason, mode=self.navigator.mode, pose=self.pose.tolist(),
                    map_xy=(self.xy+self.floor_map.offset).tolist() if self.floor_map.attached else None,
                    floor_keyframes=len(self.floor_map.keys), map_revisits=self.floor_map.revisits,
                    map_tracking_updates=self.floor_map.local_updates,
                    map_recoveries=self.floor_map.recoveries,
                    turn=self.turn, distance=self.distance, vo_inliers=self.odometry.inliers,
                    vo_accepted=self.odometry.accepted, good_frames=self.good_frames,
                    velocity_valid=self.navigator.velocity_valid,
                    odometry_method=self.odometry.method,
                    relocalized=self.odometry.relocalized,
                    last_lane_observation_time=self.last_lane_observation_time,
                    rgb_usable=self.rgb_usable,
                    floor_tracking_reason=self.odometry.floor.reason,
                    floor_tracking_quality=dict(self.odometry.floor.quality),
                    gap_phase=self.navigator.gap.phase,gap_error=self.navigator.gap.error.tolist(),
                    gap_recoveries=self.navigator.gap.recoveries, gap_hold=self.navigator.gap.hold_reason,
                    motion_recovery=self.motion_recovery.phase,
                    motion_recoveries=self.motion_recovery.recoveries,
                    gap_target=None if self.navigator.gap.target is None else self.navigator.gap.target.tolist(),
                    measured_velocity=self.navigator.measured_velocity.tolist(),
                    effective_lane=None if getattr(self.navigator,'effective_lane',None) is None else self.navigator.effective_lane.tolist(),
                    corridor_heading=self.navigator.corridor_heading,
                    corridor_point=None if self.navigator.corridor_point is None else self.navigator.corridor_point.tolist(),
                    finish_target=None if self.navigator.finish_target is None else self.navigator.finish_target.tolist(),
                    finish_normal=None if self.navigator.finish_normal is None else self.navigator.finish_normal.tolist(),
                    ground=self.odometry.geometry.ground.result.diagnostic(),
                    localization_reseeds=self.localization_reseeds,
                    desired_velocity=self.desired.tolist(),
                    target_policy_velocity=self.adapter.target.tolist(),
                    predicted_policy_velocity=self.adapter.filtered.tolist())


class OnboardNavigation:
    """The camera loop owns steering. One worker owns optional tracking on clear lanes.

    On box/finish entry, drain the one in-flight observation and hand the tracker
    back to the camera loop. No mutable tracker data are read while the worker runs.
    """
    def __init__(self, width, height, intrinsics, position, rotation, speed, observation_lease):
        self.tracked = NavigationRuntime(width, height, intrinsics, position, rotation, speed, observation_lease)
        self.local = CameraPerception(width, height, 80., position, rotation, intrinsics)
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.snapshot = self.tracked.diagnostic()
        self.missed_previous = False
        self.recovery_stamp = self.recovery_sample = None
        self.snapshot_time = None
        self.lease = observation_lease
        self.speed = speed
        self.adapter = AMPCommandAdapter()
        self.recovery = MotionRecovery()
        self.initial_yaw = self.yaw = self.last_imu_time = None
        self.turn = 0.
        self.last_frame_time = self.lane_time = self.lane_yaw = None
        self.scene = 'unknown_depth'
        self.lane = self.support = None
        self.box_latched = self.saw_box = self.finish_latched = False
        self.tracking_control = self.finished = self.holding_lane = False
        self.last_control_time = self.deadline = None
        self.desired = np.zeros(3)
        self.reason = 'initializing'

    def set_attitude(self, timestamp, rotation):
        NavigationRuntime._rotation(rotation)
        if not np.isfinite(timestamp) or (self.last_imu_time is not None and timestamp<self.last_imu_time):
            raise ValueError('Nonmonotonic IMU timestamp')
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        if self.initial_yaw is None:
            self.initial_yaw = yaw
        if self.yaw is not None:
            self.turn += (yaw-self.yaw+np.pi)%(2*np.pi)-np.pi
        self.yaw, self.last_imu_time = yaw, timestamp
        self.attitude = rotation.copy()

    def _track(self, frame, attitude, turn):
        r = self.tracked
        if r.initial_yaw is None:
            r.initial_yaw = self.initial_yaw
        if r.last_imu_time is None or frame.time>=r.last_imu_time:
            r.set_attitude(frame.time, attitude)
        r.turn = turn
        r.observe(frame, attitude, frame.time)
        state = r.diagnostic()
        state['nearby_box'] = any(-.46<b[0]<1.15 and abs(b[1])<.85 for b in r.observation.obstacles)
        return frame.time, state

    def _collect(self):
        if self.pending is not None:
            self.snapshot_time, self.snapshot = self.pending.result()
            self.pending = None

    def close(self, path):
        try:
            self._collect()
        finally:
            self.worker.shutdown(wait=True)
        self.tracked.floor_map.save(path)

    def observe(self, frame, attitude, imu_time, inhibit):
        if not np.isfinite(frame.time) or abs(frame.time-imu_time)>.15 or self.last_imu_time is None:
            raise ValueError('No time-aligned attitude for camera frame')
        if self.last_frame_time is not None and frame.time<=self.last_frame_time:
            raise ValueError('Nonmonotonic camera timestamp')
        if frame.depth.shape!=self.local.rays.shape[:2] or frame.rgb.shape!=(*frame.depth.shape,3):
            raise ValueError('Camera resolution differs from calibrated intrinsics')
        self.last_frame_time = frame.time
        if self.pending is not None and self.pending.done():
            self._collect()
        fresh_lane = self.lane_time is not None and frame.time-self.lane_time<self.lease
        capture_yaw = np.arctan2(attitude[1,0], attitude[0,0])
        reference = None
        if fresh_lane:
            # Rotate the previous centre into the new heading before identifying a single edge.
            delta = self.lane_yaw-capture_yaw
            if abs(np.arctan2(np.sin(delta),np.cos(delta)))<.30:
                x = np.linspace(*self.support, 12)
                points = np.column_stack([x, np.polyval(self.lane, x)])@rotation2(delta).T
                if np.ptp(points[:,0])>.12:
                    reference = np.polyfit(points[:,0], points[:,1], 2)
        observation, self.scene = self.local.inspect_lane(frame, attitude, reference)
        image_ok = np.count_nonzero((frame.rgb.max(axis=2)>5)&(frame.rgb.min(axis=2)<250))>=frame.depth.size*.01
        if not image_ok:
            self.scene = 'unusable_rgb'
        was_tracking = self.tracking_control
        recent_box = (self.snapshot_time is not None and frame.time-self.snapshot_time<=self.lease
                      and self.snapshot.get('nearby_box', False))
        if self.scene=='raised_object' or recent_box:
            self.box_latched = True
        if (self.turn>5.5 and len(observation.finish)) or self.snapshot.get('finish_target') is not None:
            self.finish_latched = True
        self.tracking_control = self.box_latched or self.finish_latched
        self.holding_lane = False
        if self.tracking_control:
            if not was_tracking or self.scene in ('unusable_rgb','unknown_depth'):
                inhibit()
            self._collect()
            self.snapshot_time, self.snapshot = self._track(frame, attitude, self.turn)
            r = self.tracked
            boxes = r.observation.obstacles
            # Count as passed only a tracked box that is actually behind the body.
            nearby = any(-.46<b[0]<1.25 and abs(b[1])<.85 for b in boxes)
            passed = any(b[0]<-.46 and abs(b[1])<.85 for b in boxes)
            if r.odometry.accepted and nearby:
                self.saw_box = True
            if (self.box_latched and self.saw_box and passed and not nearby and self.scene=='clear'
                    and r.odometry.accepted and r.navigator.gap.phase=='idle' and observation.lane_observed):
                self.box_latched = self.saw_box = False
            self.finish_latched |= r.navigator.finish_target is not None
            self.tracking_control = self.box_latched or self.finish_latched
        elif self.pending is None:
            # Drop optional mapping frames if busy. Never enqueue a backlog in front of steering.
            self.pending = self.worker.submit(self._track, frame, attitude.copy(), self.turn)
        if self.tracking_control:
            self.lane_time = None
            return self.tracked.observation
        if self.scene=='clear' and observation.lane_observed:
            self.lane, self.support = observation.lane.copy(), observation.lane_support
            self.lane_time, self.lane_yaw = frame.time, capture_yaw
        elif self.scene=='clear' and reference is not None and not self.missed_previous:
            self.holding_lane = True
        else:
            self.lane_time = None
            inhibit()
        self.missed_previous = not observation.lane_observed
        return observation

    def command(self, timestamp, enabled=True):
        previous = self.last_control_time
        self.last_control_time = timestamp
        dt = .02 if previous is None else np.clip(timestamp-previous, 0., .10)
        self.deadline = self.last_frame_time+self.lease if self.last_frame_time is not None else timestamp
        if not enabled:
            self.reason = 'disabled'
        elif self.last_imu_time is None or not 0<=timestamp-self.last_imu_time<=.15:
            self.reason = 'stale_imu'
        elif self.last_frame_time is None or timestamp>self.deadline:
            self.reason = 'stale_camera'
        elif self.scene in ('unknown_depth','unusable_rgb'):
            self.reason = self.scene
        elif self.tracking_control:
            r = self.tracked
            # A raised surface the tracker has not identified is still occupied space.
            known = any(-.46<b[0]<1.35 and abs(b[1])<.85 for b in r.observation.obstacles)
            if self.box_latched and not known and r.navigator.gap.phase=='idle':
                self.reason = 'waiting_for_box_geometry'
            else:
                if self.last_imu_time>=r.last_imu_time:
                    r.set_attitude(self.last_imu_time, self.attitude)
                r.command(timestamp)
                self.finished = r.navigator.finished
                self.desired = r.desired.copy()
                self.reason = r.reason
                if r.navigator.gap.drive_until is not None:
                    self.deadline = min(self.deadline, r.navigator.gap.drive_until)
        elif self.lane_time is None or timestamp>self.lane_time+self.lease:
            self.reason = 'lost_vision'
        else:
            self.reason = 'ready'
            self.deadline = min(self.deadline, self.lane_time+self.lease)
            near, far = self.support
            near = max(.15, near)
            look = np.clip(near+.25, near, max(near,far))
            heading = np.arctan(np.polyval(np.polyder(self.lane),look))
            heading += (self.lane_yaw-self.yaw+np.pi)%(2*np.pi)-np.pi
            slope = np.polyval(np.polyder(self.lane),near)
            side = (np.polyval(self.lane,near)-near*slope)/np.hypot(1.,slope)
            vx = min(self.speed, .20) if self.holding_lane else self.speed
            target = np.array([vx, np.clip(1.8*side,-.20,.20), np.clip(2.5*heading,-.8,.8)])
            self.desired += np.clip(target-self.desired, -np.array([.4,.5,1.5])*dt, np.array([.4,.5,1.5])*dt)
            # Only a *new* accepted velocity observation can accumulate stationary time.
            if self.snapshot_time is not None and self.snapshot_time!=self.recovery_sample:
                self.recovery_sample = self.snapshot_time
                valid = self.snapshot['velocity_valid'] and 0<=timestamp-self.snapshot_time<=self.lease
                if self.recovery_stamp is None or self.snapshot_time-self.recovery_stamp>self.lease:
                    self.recovery.stalled_since = None
                self.recovery_stamp = self.snapshot_time
                self.recovery.update(self.snapshot_time,self.desired,np.asarray(self.snapshot['pose']),
                                     self.snapshot['measured_velocity'],valid)
            if self.recovery.phase=='failed':
                self.reason = 'motion_stalled'
            elif self.recovery.phase=='settle' and timestamp<self.recovery.settle_until:
                self.reason = 'standing_recovery'
        if self.reason!='ready' or self.tracking_control:
            self.recovery.stalled_since = None
        if self.reason!='ready' or timestamp>self.deadline:
            self.desired[:] = 0.
        return self.adapter.update(self.desired, enabled=self.reason=='ready' and timestamp<=self.deadline)

    def diagnostic(self):
        state = dict(self.tracked.diagnostic() if self.tracking_control else self.snapshot)
        state.update(reason=self.reason, control_source='tracked' if self.tracking_control else 'current_lane',
                     mode=state['mode'] if self.tracking_control else 'lane', scene=self.scene,
                     lane_held=self.holding_lane, tracking_frame_time=self.snapshot_time,
                     desired_velocity=self.desired.tolist(), local_lane_time=self.lane_time,
                     target_policy_velocity=self.adapter.target.tolist())
        return state
