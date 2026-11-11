"""World-side RGB-D fault injection for physical closed-loop regression runs.

World geometry identifies which painted boundary to erase from the image.
Only the resulting RGB-D frame crosses the navigation interface.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree


class SensorFaults:
    KINDS = {'hide_inner_line', 'hide_outer_line', 'hide_both_lines',
             'rgb_black', 'rgbd_black', 'depth_missing', 'depth_holes'}

    def __init__(self, path, rig, track):
        self.schedule = [] if path is None else json.loads(Path(path).read_text())
        for item in self.schedule:
            if item['kind'] not in self.KINDS:
                raise ValueError('Unknown sensor fault: ' + item['kind'])
            if not 0 <= item['start_s'] < item['end_s']:
                raise ValueError('Fault interval must have 0 <= start < end')
        self.rig = rig
        self.boundaries = [cKDTree(track.inner), cKDTree(track.outer)]
        self.active = []
        self.rng = np.random.default_rng(71)
        if rig.intrinsics is None:
            focal = rig.height / (2 * np.tan(np.deg2rad(rig.fovy) / 2))
            intrinsics = (focal, focal, (rig.width-1)/2, (rig.height-1)/2)
        else:
            intrinsics = rig.intrinsics
        fx, fy, cx, cy = intrinsics
        columns, rows = np.meshgrid(np.arange(rig.width), np.arange(rig.height))
        self.rays = np.stack([(columns-cx)/fx, (rows-cy)/fy, np.ones_like(columns)], -1)

    def apply(self, frame, data, base):
        self.active = []
        for fault in self.schedule:
            if not fault['start_s'] <= frame.time < fault['end_s']:
                continue
            kind = fault['kind']
            altered = 0
            if kind.startswith('hide_'):
                body = (self.rays * frame.depth[..., None]) @ self.rig.camera_rotation.T + self.rig.camera_position
                world = body @ data.xmat[base].reshape(3, 3).T + data.xpos[base]
                rgb = frame.rgb.astype(float)
                marking = (np.isfinite(world).all(axis=2) & (abs(world[..., 2]) < .015) &
                           (rgb.min(axis=2) > 90) & (np.ptp(rgb, axis=2) < .25*rgb.max(axis=2)))
                rows, columns = np.nonzero(marking)
                if len(rows):
                    points = world[rows, columns, :2]
                    distances = np.column_stack([tree.query(points)[0] for tree in self.boundaries])
                    selected = distances.min(axis=1) < .025
                    if kind == 'hide_inner_line':
                        selected &= distances[:, 0] < distances[:, 1]
                    elif kind == 'hide_outer_line':
                        selected &= distances[:, 1] < distances[:, 0]
                    mask = np.zeros(frame.depth.shape, np.uint8)
                    mask[rows[selected], columns[selected]] = 255
                    altered = int(selected.sum())
                    frame.rgb[:] = cv2.inpaint(frame.rgb, mask, 3., cv2.INPAINT_TELEA)
            elif kind in ('rgb_black', 'rgbd_black'):
                frame.rgb[:] = 0
                altered = frame.depth.size
                if kind == 'rgbd_black':
                    frame.depth[:] = np.nan
            elif kind == 'depth_missing':
                frame.depth[:] = np.nan
                altered = frame.depth.size
            elif kind == 'depth_holes':
                mask = self.rng.random(frame.depth.shape) < .10
                frame.depth[mask] = np.nan
                altered = int(mask.sum())
            self.active.append({'kind': kind, 'altered_pixels': altered})
        return frame
