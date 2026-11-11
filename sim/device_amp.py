"""Pi Plus Pro AMP contract audited from the running controller on 2026-11-10.

Inputs/outputs use policy joint order. The MuJoCo adapter owns the conversion
between hardware joint zeros and the original Mini Pi Plus simulation zeros.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

DEVICE = Path(__file__).resolve().parents[1] / 'vendor/pi_plus_pro'


def load_device_bundle():
    source = json.loads((DEVICE/'SOURCE.json').read_text())
    for name, digest in source['sha256'].items():
        if hashlib.sha256((DEVICE/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Device policy bundle checksum mismatch: {name}')
    config = yaml.safe_load((DEVICE/'amp.yaml').read_text())
    if (config['algorithm'], config['dofs'], config['num_single_obs'], config['frame_stack']) != ('amp', 20, 69, 5):
        raise ValueError('Unsupported device AMP contract')
    actor = nn.Sequential(nn.Linear(345,512), nn.ELU(), nn.Linear(512,256), nn.ELU(),
                          nn.Linear(256,128), nn.ELU(), nn.Linear(128,20))
    with np.load(DEVICE/'actor.npz', allow_pickle=False) as data:
        actor.load_state_dict({k:torch.from_numpy(data[k].copy()) for k in data.files})
    return config, actor.eval(), source


class DeviceAMP:
    """Normal Soccer mode, as initialized by the factory AMP constructor.

This models the locomotion command path; gamepad mode buttons, startup FSM,
hardware watchdogs and head control are outside the simulation adapter.
"""
    def __init__(self):
        self.config, self.actor, self.source = load_device_bundle()
        c = self.config
        self.command = np.zeros(3, dtype=np.float64)
        self.history = np.zeros((c['frame_stack'], c['num_single_obs']), dtype=np.float64)
        self.action = np.zeros(c['dofs'], dtype=np.float64)
        self.offset = np.asarray(c['urdf_offset'], dtype=float)
        self.command_scale = np.array([c['cmd_lin_vel_scale']]*2+[c['cmd_ang_vel_scale']])
        self.command_lower = np.array([c['cmd_vel_x_min'], c['cmd_vel_y_min'], c['cmd_vel_yaw_min']])
        self.command_upper = np.array([c['cmd_vel_x_max_soccer_normal'], c['cmd_vel_y_max'], c['cmd_vel_yaw_max']])
        if c['control_mode'] != 'Soccer':
            raise ValueError('This audited bundle expects Soccer mode')

    def filter_command(self, command):
        command = np.asarray(command, dtype=float)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError('Expected finite body-frame vx, vy, yaw_rate')
        c = self.config
        alpha = c['cmd_vel_filter_scale_soccer']
        self.command = (alpha*np.clip(command, self.command_lower, self.command_upper)
                        +(1-alpha)*self.command)*self.command_scale
        # Factory AMPPolicy applies deadbands AFTER filtering and feeds the
        # deadbanded result back into the next tick (0x16481f0..0x1648384).
        self.command[np.abs(self.command) < np.array([.08,.08,.17])] = 0.
        return self.command.copy()

    def observe(self, gyro, gravity, command, position, velocity):
        c = self.config
        obs = np.concatenate([np.asarray(gyro)*c['rbt_ang_vel_scale'], gravity,
                              self.filter_command(command),
                              np.asarray(position)*c['rbt_lin_pos_scale']-self.offset,
                              np.asarray(velocity)*c['rbt_lin_vel_scale'], self.action])
        if obs.shape != (69,) or not np.isfinite(obs).all():
            raise ValueError('Invalid AMP observation')
        self.history[:-1] = self.history[1:]
        self.history[-1] = np.clip(obs, -c['clip_obs'], c['clip_obs'])
        return self.history.astype(np.float32).reshape(1, -1)

    def infer(self, observations):
        with torch.inference_mode():
            action = self.actor(torch.from_numpy(observations)).numpy()[0]
        if not np.isfinite(action).all():
            raise RuntimeError('Non-finite device policy output')
        self.action = np.clip(action, self.config['clip_actions_lower'], self.config['clip_actions_upper'])
        return self.action.copy()

    def target(self):
        c = self.config
        displacement = c['action_scale']*np.asarray(c['action_scales'])*self.action
        return self.offset + np.clip(displacement, c['clip_output_lower'], c['clip_output_upper'])

    def metadata(self):
        return {'profile': 'pi_plus_pro_device_20261110', 'policy_name': self.config['policy_name'],
                'actor_path': str(DEVICE/'actor.npz'), 'source_sha256': self.source['sha256'],
                'rl_hz': self.config['rl_ctrl_f'], 'pd_hz': self.config['pd_ctrl_f'],
                'command_mode': 'Soccer/normal', 'command_filter_alpha': self.config['cmd_vel_filter_scale_soccer'],
                'command_deadband': [.08,.08,.17], 'clip_obs': self.config['clip_obs']}
