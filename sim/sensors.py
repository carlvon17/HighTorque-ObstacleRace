"""Simulated onboard measurements; simulator state is confined to this adapter.

The navigation packet deliberately has no global position, ground-truth velocity,
contact IDs, segmentation IDs, obstacle coordinates, or track geometry.
"""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from sim.robot import JOINTS, VENDOR


@dataclass(frozen=True)
class Proprioception:
    time: float
    joints: np.ndarray
    orientation_wxyz: np.ndarray
    gyro: np.ndarray


@dataclass(frozen=True)
class CameraFrame:
    time: float
    rgb: np.ndarray
    depth: np.ndarray


class SensorRig:
    width, height, fovy = 480, 360, 70.
    camera_position = np.array([.09, 0, .22])
    # OpenCV optical frame (right, down, forward) -> robot body frame.
    camera_rotation = np.array([[0, -.5, np.sqrt(.75)], [-1, 0, 0], [0, -np.sqrt(.75), -.5]])

    def __init__(self, model, seed=0, noise=True, camera_profile=None):
        camera=model.camera('nav_camera').id
        self.fovy=float(model.cam_fovy[camera])
        self.camera_position=model.cam_pos[camera].copy()
        q=model.cam_quat[camera]
        self.camera_rotation=Rotation.from_quat(np.r_[q[1:],q[0]]).as_matrix()@np.diag([1,-1,-1])
        self.intrinsics = None
        self.fps = 1/.06
        self.capture_width, self.capture_height = self.width, self.height
        if camera_profile is not None:
            self.capture_width = int(camera_profile['width'])
            self.capture_height = int(camera_profile['height'])
            scale = camera_profile['processing_scale']
            self.width = int(self.capture_width*scale)
            self.height = int(self.capture_height*scale)
            self.fps = float(camera_profile['fps'])
            # Use the same pixel-center convention as deploy/shadow.py.
            self.intrinsics = (camera_profile['fx']*scale, camera_profile['fy']*scale,
                               (camera_profile['cx']+.5)*scale-.5,
                               (camera_profile['cy']+.5)*scale-.5)
        self.renderer = mujoco.Renderer(model, height=self.capture_height, width=self.capture_width)
        self.joint_ids = [model.joint(n).qposadr[0] for n in JOINTS]
        self.rng = np.random.default_rng(seed)
        self.noise = noise

    def proprioception(self, data):
        q = data.sensor('orientation').data.copy()
        joints = data.qpos[self.joint_ids].copy()
        gyro = data.sensor('angular-velocity').data.copy()
        if self.noise:
            rotation = Rotation.from_quat(np.r_[q[1:], q[0]]) * Rotation.from_rotvec(self.rng.normal(0, .0008, 3))
            q = rotation.as_quat()[[3,0,1,2]]
            joints += self.rng.normal(0, .0001, len(joints))
            gyro += self.rng.normal(0, .003, 3)
        return Proprioception(float(data.time), joints, q, gyro)

    def camera(self, data):
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera='nav_camera')
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        if self.noise:
            depth += self.rng.normal(0, .0007, depth.shape)
        if (self.width, self.height) != (self.capture_width, self.capture_height):
            rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth[(depth < .15) | (depth > 4.)] = np.nan
        return CameraFrame(float(data.time), rgb, depth)

    def close(self):
        self.renderer.close()


class KinematicOdometry:
    """Leg encoder + attitude odometry with a lowest-foot contact estimate.

This owns a separate robot-only FK model whose root translation is always zero.
Neither simulator contacts nor simulated root position/velocity enter it.
"""
    def __init__(self):
        source = VENDOR / 'legged_lab/assets/hightorque/pi_plus/pi_plus.xml'
        root = ET.parse(source).getroot()
        root.find('compiler').attrib.pop('meshdir', None)
        for mesh in root.findall('asset/mesh'):
            mesh.set('file', str(source.parent/'meshes'/mesh.get('file')))
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.qadr = [self.model.joint(n).qposadr[0] for n in JOINTS]
        self.feet = [self.model.body(f'{s}_ankle_roll_link').id for s in ('l', 'r')]
        self.soles = []
        for b in self.feet:
            g = next(g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == b and self.model.geom_contype[g])
            mesh = self.model.geom_dataid[g]
            vertices = self.model.mesh_vert[self.model.mesh_vertadr[mesh]:self.model.mesh_vertadr[mesh]+self.model.mesh_vertnum[mesh]].copy()
            self.soles.append((g, vertices))
        self.xy = np.zeros(2)
        self.rotation = np.eye(3)
        self.initial_yaw = None
        self.previous = None
        self.previous_z = None
        self.previous_vertices = None
        self.yaw = 0.
        self.turn = 0.
        self.distance = 0.

    def update(self, measurement):
        q = measurement.orientation_wxyz
        r = Rotation.from_quat(np.r_[q[1:],q[0]]).as_matrix()
        yaw = np.arctan2(r[1,0], r[0,0])
        if self.initial_yaw is None:
            self.initial_yaw = yaw
        self.rotation = Rotation.from_euler('z', -self.initial_yaw).as_matrix() @ r
        heading = np.arctan2(self.rotation[1,0], self.rotation[0,0])
        self.turn += (heading-self.yaw+np.pi)%(2*np.pi)-np.pi
        self.yaw = heading
        self.data.qpos[:3] = 0
        self.data.qpos[3:7] = Rotation.from_matrix(self.rotation).as_quat()[[3,0,1,2]]
        self.data.qpos[self.qadr] = measurement.joints
        mujoco.mj_kinematics(self.model, self.data)
        feet = self.data.xpos[self.feet].copy()
        vertices=[v@self.data.geom_xmat[g].reshape(3,3).T+self.data.geom_xpos[g] for g,v in self.soles]
        z = np.array([v[:,2].min() for v in vertices])
        if self.previous is not None:
            # Both endpoints should belong to support; exclude a swinging foot.
            score = z+self.previous_z
            foot = int(np.argmin(score))
            # A contact point stays still; the ankle origin moves when the foot
            # rocks. Use the same low sole vertex at both endpoints.
            vertex=int(np.argmin(vertices[foot][:,2]+self.previous_vertices[foot][:,2]))
            delta = self.previous_vertices[foot][vertex,:2]-vertices[foot][vertex,:2]
            if np.linalg.norm(delta) < .03:
                self.xy += delta
                self.distance += np.linalg.norm(delta)
        self.previous, self.previous_z = feet, z
        self.previous_vertices=vertices
        return np.r_[self.xy, self.yaw]

    @property
    def pose(self):
        return np.r_[self.xy, self.yaw]
