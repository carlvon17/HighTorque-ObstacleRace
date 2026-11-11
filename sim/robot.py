"""MuJoCo integration with the audited physical Pi Plus Pro AMP by default.

The original upstream checkpoint is available only through an explicit policy
path with the historical Mini model. Default geometry is the supplied Pro MJCF.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
from torch import nn
from sim.device_amp import DEVICE, DeviceAMP
from sim.field import DEFAULT_BEND, FIELD_DIRS

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/mini_pi_plus_amp"
DEFAULT = np.array([-.25, 0, 0, .65, -.4, 0, 0, .2, 0, -1.2,
                    -.25, 0, 0, .65, -.4, 0, 0, -.2, 0, -1.2])
JOINTS = [f"{side}_{name}_joint" for side in ("l", "r") for name in
          ("hip_pitch", "hip_roll", "thigh", "calf", "ankle_pitch", "ankle_roll",
           "shoulder_pitch", "shoulder_roll", "upper_arm", "elbow")]
MJ_TO_ISAAC = np.array([0, 6, 10, 16, 1, 7, 11, 17, 2, 8, 12, 18, 3, 9, 13, 19, 4, 14, 5, 15])
ISAAC_TO_MJ = np.argsort(MJ_TO_ISAAC)
KP = np.array([80, 80, 80, 80, 60, 60, 30, 30, 30, 30] * 2)
KD = np.array([1.1, 1.1, 1.1, 1.1, 1.2, 1.2, .6, .6, .6, .6] * 2)


PRO_NAME = "PiPlusPro_SF_12L8A0G2H0W_Z_261010"
PRO_XML = ROOT / "reference/robot" / PRO_NAME / "xml" / (PRO_NAME + ".xml")
HEAD_JOINTS = ("head_yaw_joint", "head_pitch_joint")


def set_camera_intrinsics(camera, profile):
    """Set pixel-center intrinsics on the MuJoCo image plane."""
    w, h = profile['width'], profile['height']
    camera.attrib.pop('fovy', None)
    # Only the focal length and sensor size ratios matter here.
    camera.set('resolution', f'{w} {h}')
    camera.set('sensorsize', f'{w*1e-5} {h*1e-5}')
    camera.set('focalpixel', f"{profile['fx']} {profile['fy']}")
    camera.set('principalpixel', f"{(w-1)/2-profile['cx']} {(h-1)/2-profile['cy']}")


def configure_pro(root):
    # The vendor export supplies mass, joints, limits and collision shapes;
    # it has no actuators or onboard sensors.
    base = root.find("worldbody/body[@name='base_link']")
    for geom in base.iter('geom'):
        if geom.get('contype') != '0':
            geom.set('group', '3')  # render the CAD skins, hide contact primitives
    actuator = ET.SubElement(root, 'actuator')
    armature = {'ankle': .016838, 'shoulder': .008419, 'upper': .008419, 'elbow': .008419}
    for name in JOINTS:
        joint = base.find(f".//joint[@name='{name}']")
        limits = joint.get('actuatorfrcrange')
        # The CAD export omits reflected motor inertia. Retain the upstream
        # transmission estimates by motor class for the 1 kHz sampled PD.
        motor_type = name.split('_')[1]
        joint.set('armature', str(armature.get(motor_type, .044277)))
        ET.SubElement(actuator, 'motor', name=name, joint=name, ctrlrange=limits)
    ET.SubElement(base, 'site', name='imu', size='.001', rgba='0 0 0 0')
    sensor = ET.SubElement(root, 'sensor')
    ET.SubElement(sensor, 'framequat', name='orientation', objtype='site', objname='imu')
    ET.SubElement(sensor, 'gyro', name='angular-velocity', site='imu')
    reference = json.loads((ROOT/'deploy/task_head_pose.json').read_text())['measured_joint_reference_rad']
    holds = ET.SubElement(root, 'equality')
    for joint in HEAD_JOINTS:
        q = reference[joint]
        link = base.find(f".//body/joint[@name='{joint}']/..")
        hinge = link.find(f"joint[@name='{joint}']")
        lo, hi = map(float, hinge.get('range').split())
        if not lo <= q <= hi:
            raise ValueError(f'Invalid task head angle for {joint}: {q}')
        # In MJCF, ref is subtracted from q during FK. Rotate the neutral
        # link by the same amount to preserve FK at *every* encoder angle.
        # This also makes a standalone scene open at the calibrated head pose.
        hinge.set('ref', str(q))
        link.set('axisangle', f"{hinge.get('axis')} {q}")
        ET.SubElement(holds, 'joint', name='hold_'+joint, joint1=joint,
                      polycoef='0 0 0 0 0', solref='.002 1')
    mount = json.loads((ROOT/'sim/pro_camera_mount.json').read_text())
    camera = ET.SubElement(base.find(".//body[@name='camera_link']"), 'camera', name='nav_camera',
                           pos=' '.join(map(str, mount['position_camera_link_m'])),
                           xyaxes=' '.join(map(str, mount['xyaxes_camera_link'])))
    set_camera_intrinsics(camera, json.loads((ROOT/'sim/d435i_camera.json').read_text()))


def build_model(no_obstacles=False, timestep=.001, legacy=False, *, bend_profile=DEFAULT_BEND):
    """Assemble the robot, its calibrated sensors and the competition field."""
    source = VENDOR / "legged_lab/assets/hightorque/pi_plus/pi_plus.xml" if legacy else PRO_XML
    if legacy and not source.exists():
        raise FileNotFoundError("Run .venv/bin/python scripts/fetch_pi_plus.py first")
    root = ET.parse(source).getroot()
    meshdir = source.parent / root.find("compiler").attrib.pop("meshdir", "meshes")
    if not legacy:
        configure_pro(root)
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", str((meshdir / mesh.get("file")).resolve()))
    for world in root.findall("worldbody"):
        for geom in list(world.findall("geom")):
            if geom.get("name") == "floor":
                world.remove(geom)
    field_path = FIELD_DIRS[bend_profile] / "mujoco/field.xml"
    if not field_path.exists():
        raise FileNotFoundError("Run scripts/generate_field.py first")
    field = ET.parse(field_path).getroot()
    for texture in field.findall("asset/texture"):
        texture.set("file", str(field_path.parent / texture.get("file")))
    if no_obstacles:
        body = field.find("worldbody/body")
        for child in list(body.findall("body")):
            body.remove(child)
    root.extend(list(field))
    ET.SubElement(root, "option", timestep=str(timestep), gravity="0 0 -9.81")
    visual = root.find("visual")
    visual.find("global").set("offwidth", "1280")
    visual.find("global").set("offheight", "720")
    ET.SubElement(visual, "map", znear="0.005")
    if legacy:
        base = root.find("worldbody/body[@name='base_link']")
        ET.SubElement(base, "camera", name="nav_camera", pos="0.09 0 0.22",
                      xyaxes="0 -1 0 0.5 0 0.8660254", fovy="70")
    ET.SubElement(root.find("worldbody"), "camera", name="overview", pos="3 0 11",
                  xyaxes="1 0 0 0 1 0", fovy="36")
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    if model.nu != 20 or model.nq != (27 if legacy else 29):
        raise ValueError(f"Unexpected robot dimensions: nq={model.nq}, nu={model.nu}")
    return model, xml


class _UnusedNormalizerState:
    """Inert placeholder for training-only discriminator statistics."""


def load_policy(path):
    """Load tensor-only training checkpoint; do not execute pickle objects."""
    numpy_core = getattr(np, '_core', None) or np.core
    allowed = [(_UnusedNormalizerState, "rsl_rl.utils.utils.Normalizer"), np.ndarray,
               np.dtype, type(np.dtype(np.float64)), type(np.dtype(np.float32)),
               (numpy_core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct")]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = {k[6:]: v for k, v in checkpoint["model_state_dict"].items() if k.startswith("actor.")}
    indices = sorted(int(k.split('.')[0]) for k in state if k.endswith(".weight"))
    layers = []
    for i, index in enumerate(indices):
        weight = state[f"{index}.weight"]
        layers.append(nn.Linear(weight.shape[1], weight.shape[0]))
        if i < len(indices) - 1:
            layers.append(nn.ELU())
    actor = nn.Sequential(*layers)
    actor.load_state_dict(state)
    actor.eval()
    if tuple(actor(torch.zeros(1, 345)).shape) != (1, 20):
        raise ValueError("Policy must map 5 x 69 observations to 20 actions")
    return actor


class Robot:
    dt = .02

    def __init__(self, model, start, policy=None, *, from_standby=False):
        self.model = model
        self.data = mujoco.MjData(model)
        self.base = model.body("base_link").id
        self.qadr = np.array([model.joint(n).qposadr[0] for n in JOINTS])
        self.vadr = np.array([model.joint(n).dofadr[0] for n in JOINTS])
        self.aids = np.array([model.actuator(n).id for n in JOINTS])
        torch.set_num_threads(1)
        self.device = DeviceAMP() if policy is None or Path(policy).resolve() == (DEVICE/'actor.npz').resolve() else None
        self.policy_path = DEVICE/'actor.npz' if self.device else Path(policy)
        self.policy = self.device.actor if self.device else load_policy(self.policy_path)
        self.kp, self.kd = KP.copy(), KD.copy()
        if self.device:
            c = self.device.config
            if c['joint_names'] != [JOINTS[i] for i in MJ_TO_ISAAC]:
                raise ValueError('Device and MuJoCo joint ordering do not agree')
            self.dt = 1/c['rl_ctrl_f']
            self.kp = np.asarray(c['kp'])[ISAAC_TO_MJ]
            self.kd = np.asarray(c['kd'])[ISAAC_TO_MJ]
            if not np.isclose(model.opt.timestep, 1/c['pd_ctrl_f']):
                raise ValueError('Device AMP requires the audited 1000 Hz PD/physics timestep')
        self.substeps = round(self.dt/model.opt.timestep)
        if not np.isclose(self.substeps*model.opt.timestep, self.dt):
            raise ValueError('Physics timestep must divide the policy period')
        self.history = np.zeros(345, dtype=np.float32)
        self.action = np.zeros(20, dtype=np.float32)
        self.command = np.zeros(3)
        self.data.qpos[self.qadr] = DEFAULT
        self.data.qpos[:2] = start[:2]
        self.data.qpos[3:7] = [np.cos(start[2]/2), 0, 0, np.sin(start[2]/2)]
        self.from_standby = from_standby and model.nq == 29
        if self.from_standby:
            mount = json.loads((ROOT/'sim/pro_camera_mount.json').read_text())
            roll, pitch = np.deg2rad(mount['standby_base_roll_pitch_deg'])
            mujoco.mju_euler2Quat(self.data.qpos[3:7], np.array([roll, pitch, start[2]]), 'XYZ')
        mujoco.mj_forward(model, self.data)

    def locomotion_metadata(self):
        result = self.device.metadata() if self.device else {
            'profile': 'legacy_upstream_checkpoint', 'actor_path': str(self.policy_path),
            'rl_hz': 1/self.dt, 'pd_hz': 1/self.model.opt.timestep}
        result['robot_model'] = PRO_NAME if self.model.nq == 29 else 'upstream Mini Pi Plus'
        result['from_measured_standby_attitude'] = self.from_standby
        return result

    @property
    def pose(self):
        rotation = self.data.xmat[self.base].reshape(3, 3)
        return np.array([*self.data.xpos[self.base, :2], np.arctan2(rotation[1, 0], rotation[0, 0])])

    def step(self, command, callback=None):
        self.command = np.asarray(command, dtype=float)
        if self.command.shape != (3,) or not np.isfinite(self.command).all():
            raise ValueError("Expected finite body-frame vx, vy, yaw_rate")
        gyro = self.data.sensor("angular-velocity").data
        gravity = self.data.xmat[self.base].reshape(3, 3).T @ np.array([0., 0., -1.])
        relative_position = (self.data.qpos[self.qadr] - DEFAULT)[MJ_TO_ISAAC]
        velocity = self.data.qvel[self.vadr][MJ_TO_ISAAC]
        if self.device:
            # The CAD zero for shoulder pitch/roll differs from the encoder
            # convention in the policy. Map through the nominal posture.
            observations = self.device.observe(gyro, gravity, self.command,
                                               relative_position+self.device.offset, velocity)
            self.history = observations[0].copy()
            self.action = self.device.infer(observations)
            target = DEFAULT + (self.device.target()-self.device.offset)[ISAAC_TO_MJ]
        else:
            obs = np.concatenate([gyro, gravity, self.command, relative_position, velocity, self.action])
            self.history[:-69] = self.history[69:]
            self.history[-69:] = obs
            with torch.inference_mode():
                self.action = np.clip(self.policy(torch.from_numpy(np.clip(self.history, -100, 100))[None]).numpy()[0], -100, 100)
            if not np.isfinite(self.action).all():
                raise RuntimeError("Non-finite policy output")
            target = DEFAULT + .25 * self.action[ISAAC_TO_MJ]
        for _ in range(self.substeps):
            torque = self.kp * (target - self.data.qpos[self.qadr]) - self.kd * self.data.qvel[self.vadr]
            self.data.ctrl[self.aids] = np.clip(torque, self.model.actuator_ctrlrange[self.aids, 0],
                                                      self.model.actuator_ctrlrange[self.aids, 1])
            mujoco.mj_step(self.model, self.data)
            if callback:
                callback()
        # Refresh body transforms after integration for navigation and scoring.
        mujoco.mj_forward(self.model, self.data)
