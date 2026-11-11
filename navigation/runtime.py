"""Identical sensor-to-command pipeline for simulation, replay and ROS.

All positions are estimated from RGB-D. Attitude is world_from_body from an
AHRS measurement. No world position or simulator object data is accepted.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from navigation.command import AMPCommandAdapter
from navigation.environment import verify_dependencies
from navigation.motion_recovery import MotionRecovery
from navigation.visual import CameraPerception, Perception, RGBDOdometry, VisualNavigator, rotation2


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
    def __init__(self, width, height, intrinsics, camera_position, camera_rotation, speed=.22):
        verify_dependencies()
        if not np.isfinite(speed) or not 0 < speed <= .4:
            raise ValueError('Navigation target speed must be within (0, 0.4]')
        self.position = np.asarray(camera_position, dtype=float)
        self.camera_rotation = np.asarray(camera_rotation, dtype=float)
        if self.position.shape != (3,) or not np.isfinite(self.position).all():
            raise ValueError('Invalid camera position')
        self._rotation(self.camera_rotation)
        self.perception = CameraPerception(width, height, 80., self.position, self.camera_rotation, intrinsics)
        self.odometry = RGBDOdometry(width, height, 80., self.position, self.camera_rotation, intrinsics)
        self.navigator = VisualNavigator(speed)
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
        if self.last_frame_time is not None and frame.time-self.last_frame_time > .35:
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
        self.distance += float(np.linalg.norm(xy-self.xy))
        self.xy = xy
        frame_pose = np.r_[self.xy, np.arctan2(frame_attitude[1, 0], frame_attitude[0, 0])]
        if self.odometry.accepted and self.rgb_usable:
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
        self.navigator.observe(self.observation, frame_pose, self.turn)
        valid = self.observation.lane is not None and self.odometry.accepted and self.rgb_usable
        self.good_frames = self.good_frames+1 if valid else 0
        self.last_frame_time = float(frame.time)
        return self.observation

    def command(self, timestamp, policy_feedback=None, enabled=True):
        if not np.isfinite(timestamp) or (self.last_control_time is not None and timestamp <= self.last_control_time):
            raise ValueError('Nonmonotonic control timestamp')
        dt = .02 if self.last_control_time is None else min(.1, timestamp-self.last_control_time)
        self.last_control_time = float(timestamp)
        self.desired = self.navigator.update(timestamp, self.pose, dt)
        if not enabled:
            self.reason = 'disabled'
        elif self.last_imu_time is None or not 0 <= timestamp-self.last_imu_time <= .15:
            self.reason = 'stale_imu'
        elif self.last_frame_time is None or not 0 <= timestamp-self.last_frame_time <= .35:
            self.reason = 'stale_camera'
        elif not self.rgb_usable:
            self.reason = 'unusable_rgb'
        elif self.good_frames < 3:
            self.reason = 'waiting_for_trusted_visual_motion_and_lane'
        elif self.navigator.mode in ('lost_vision', 'blocked', 'finished', 'initializing'):
            self.reason = self.navigator.mode
        else:
            self.reason = 'ready'
        recovery=self.motion_recovery.update(timestamp,self.desired,self.pose,
            self.navigator.measured_velocity,self.reason=='ready' and self.navigator.gap.phase=='idle')
        if self.reason=='ready' and self.navigator.gap.phase=='idle':
            if recovery=='settle':self.reason='standing_recovery'
            elif recovery=='failed':
                self.reason='motion_stalled';self.navigator.mode='blocked'
        if self.reason != 'ready':
            self.desired[:] = 0.
            self.navigator.command[:] = 0.
        return self.adapter.update(self.desired, policy_feedback, self.reason == 'ready')

    def diagnostic(self):
        return dict(reason=self.reason, mode=self.navigator.mode, pose=self.pose.tolist(),
                    turn=self.turn, distance=self.distance, vo_inliers=self.odometry.inliers,
                    vo_accepted=self.odometry.accepted, good_frames=self.good_frames,
                    odometry_method=self.odometry.method,
                    relocalized=self.odometry.relocalized,
                    last_lane_observation_time=self.last_lane_observation_time,
                    rgb_usable=self.rgb_usable,
                    floor_tracking_reason=self.odometry.floor.reason,
                    floor_tracking_quality=dict(self.odometry.floor.quality),
                    gap_phase=self.navigator.gap.phase,gap_error=self.navigator.gap.error.tolist(),
                    gap_recoveries=self.navigator.gap.recoveries,
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
