"""Ground-truth baseline: collision-checked lane lattice and velocity tracking.

The route is an oracle map baseline, NOT camera perception or MPPI. The
navigation/loco boundary is body-frame [vx, vy, yaw_rate] in SI units.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


class Track:
    def __init__(self):
        self.config = json.loads((ROOT / "scene_config.json").read_text())
        geometry = json.loads((ROOT / "output/athletics/geometry.json").read_text())
        self.inner, self.outer = (np.array(geometry[key]) for key in ("inner", "outer"))
        source = np.array(geometry["center"])[::-1]
        source = np.vstack([source, source[:1]])
        cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(source, axis=0), axis=1))]
        self.length = cumulative[-1]
        self.s = np.linspace(0, self.length, int(np.ceil(self.length/.01)), endpoint=False)
        self.center = np.column_stack([np.interp(self.s, cumulative, source[:, i]) for i in range(2)])
        tangent = np.roll(self.center, -1, axis=0) - np.roll(self.center, 1, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        self.tangent = tangent
        self.normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        self.yaw = np.arctan2(tangent[:, 1], tangent[:, 0])
        self.tree = cKDTree(self.center)
        self.obstacles = np.array([o["position"] for o in self.config["obstacles"]])

    def obstacle_distance(self, points):
        points = np.asarray(points)
        distances = []
        half = np.array(self.config["obstacle_dimensions"][:2])/2
        for obstacle in self.config["obstacles"]:
            a = np.deg2rad(obstacle["yaw_degrees"])
            rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            local = (points - obstacle["position"]) @ rotation
            delta = np.abs(local) - half
            distances.append(np.linalg.norm(np.maximum(delta, 0), axis=-1) + np.minimum(np.max(delta, axis=-1), 0))
        return np.min(distances, axis=0)

    def plan(self, radius=.14, margin=.005, no_obstacles=False, full_body=True):
        """DP lateral lattice. Circle is a planning approximation, scored separately."""
        limit = .23 if full_body else (self.config["track_width"]-self.config["line_width"])/2-radius-margin
        if limit <= 0:
            raise ValueError("Planning envelope does not fit the lane")
        offsets = np.linspace(-limit, limit, 81)
        points = self.center[:, None] + self.normal[:, None]*offsets[None, :, None]
        clearance = self.obstacle_distance(points)
        valid = clearance >= radius+margin if not no_obstacles else np.ones(clearance.shape, dtype=bool)
        if full_body:
            if any(abs(o['yaw_degrees'] % 90) > 1e-6 for o in self.config['obstacles']):
                raise ValueError("Full-body baseline currently requires axis-aligned obstacles")
            # This baseline only supports the current straight obstacle section.
            for obstacle in self.obstacles:
                idx = self.tree.query(obstacle)[1]
                if abs(np.sin(self.yaw[idx])) > .01:
                    raise ValueError("Full-body baseline requires obstacles on the horizontal straight")
        body_half_extents = np.array([.13, .21])
        body_margins = np.array([margin, .015])
        def body_obstacles():
            for o in self.config['obstacles']:
                half = np.array(self.config['obstacle_dimensions'][:2])/2
                if round(o['yaw_degrees']/90) % 2:
                    half = half[::-1]
                yield np.array(o['position']), half + body_half_extents + body_margins

        def body_clear(p):
            # In the straight obstacle section the torso stays parallel to track.
            # Conservative arm/foot rectangle: 26 cm long, 42 cm wide.
            # Only used there; the original 3D contacts remain the referee.
            result = np.ones(p.shape[:-1], dtype=bool)
            for center, half in body_obstacles():
                result &= np.any(np.abs(p-center) >= half, axis=-1)
            return result

        def swept_body_clear(a, b):
            # Exact segment/AABB intersection in the straight-section C-space.
            delta = b-a
            result = np.ones(delta.shape[:-1], dtype=bool)
            for center, half in body_obstacles():
                parallel = np.abs(delta) < 1e-12
                divisor = np.where(parallel, 1., delta)
                t1, t2 = (center-half-a)/divisor, (center+half-a)/divisor
                lo, hi = np.minimum(t1, t2), np.maximum(t1, t2)
                outside = np.abs(a-center) >= half
                lo = np.where(parallel, np.where(outside, np.inf, -np.inf), lo)
                hi = np.where(parallel, np.where(outside, -np.inf, np.inf), hi)
                enter = np.maximum(np.max(lo, axis=-1), 0.)
                leave = np.minimum(np.min(hi, axis=-1), 1.)
                result &= enter >= leave
            return result
        if full_body and not no_obstacles:
            valid = body_clear(points)
        n, k = valid.shape
        costs = np.full(k, np.inf)
        costs[k//2] = 0
        parents = np.zeros((n, k), dtype=int)
        delta = offsets[:, None]-offsets[None, :]
        transition = 8*delta**2/(self.length/n)**2
        transition[np.abs(delta)/(self.length/n)>(40 if full_body else 1.8)] = np.inf
        for i in range(1, n):
            candidate = costs[:, None] + transition
            if not no_obstacles and np.min(clearance[i]) < .4:
                if full_body:
                    candidate[~swept_body_clear(points[i-1, :, None], points[i, None, :])] = np.inf
                else:
                    for frac in (.25, .5, .75):
                        between = points[i-1, :, None]*(1-frac) + points[i, None, :]*frac
                        candidate[self.obstacle_distance(between)<radius+margin] = np.inf
            parent = np.argmin(candidate, axis=0)
            costs = candidate[parent, np.arange(k)] + offsets**2*2
            costs[~valid[i]] = np.inf
            parents[i] = parent
        j = k//2
        if not np.isfinite(costs[j]):
            raise ValueError("No route fits this planning envelope; change the gait/layout, not the collision geometry")
        choice = np.empty(n, dtype=int)
        for i in range(n-1, -1, -1):
            choice[i] = j
            j = parents[i, j]
        self.path = points[np.arange(n), choice]
        self.offsets = offsets[choice]
        self.radius = radius
        self.margin = margin
        self.full_body = full_body
        self.no_obstacles = no_obstacles
        return self.path


class Navigator:
    def __init__(self, track, start_pose, speed=.25):
        self.track = track
        self.speed = speed
        self.command = np.zeros(3)
        self.index = track.tree.query(start_pose[:2])[1]
        self.start_s = track.s[self.index]
        self.progress = 0.
        self.previous_s = self.start_s
        self.finished = False

    def update(self, pose, dt=.02):
        track = self.track
        # Restrict nearest-point search to the local segment; never jump lanes.
        ids = (self.index+np.arange(-20, 40)) % len(track.s)
        self.index = int(ids[np.argmin(np.linalg.norm(track.center[ids]-pose[:2], axis=1))])
        s = track.s[self.index]
        ds = (s-self.previous_s+track.length/2)%track.length-track.length/2
        self.progress += ds
        self.previous_s = s
        self.finished = self.progress >= track.length
        if self.finished:
            desired = np.zeros(3)
        else:
            near = not track.no_obstacles and np.min(np.linalg.norm(track.obstacles-pose[:2], axis=1)) < .85
            speed = min(self.speed, .10) if near else self.speed
            lookahead = .10 if near else .18
            target_i = (self.index+max(1, round(lookahead/(track.length/len(track.s))))) % len(track.s)
            target = track.path[target_i]
            yaw = pose[2]
            rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            error = (target-pose[:2]) @ rotation
            desired = np.array([speed, np.clip(1.8*error[1], -.22, .22),
                                np.clip(2.5*wrap(track.yaw[target_i]-yaw), -.8, .8)])
            if near and track.full_body:
                target_i = (self.index+3) % len(track.s)
                error = (track.path[target_i]-pose[:2]) @ rotation
                desired[0] = np.clip(1.4*error[0], -.03, .10)
                desired[1] = np.clip(2.5*error[1], -.16, .16)
                if abs(error[1]) > .015:
                    desired[0] = 0.
                norm = np.linalg.norm(desired[:2])
                if norm > .005 and norm < .12:
                    desired[:2] *= .12/norm
        # Acceleration limits in the controller boundary, not in the joint loop.
        self.command += np.clip(desired-self.command, -np.array([.4, .5, 1.5])*dt, np.array([.4, .5, 1.5])*dt)
        return self.command.copy()
