"""Explicit simulation scoring assumptions; not an official referee model."""
import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon


class Referee:
    def __init__(self, robot, track):
        self.robot = robot
        m = robot.model
        # White-line centre convention, consistent with field generation.
        self.lane = Polygon(track.outer).difference(Polygon(track.inner))
        self.feet = []
        self.foot_bodies = {m.body(f"{side}_ankle_roll_link").id for side in ("l", "r")}
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in self.foot_bodies and m.geom_contype[g]:
                mesh = m.geom_dataid[g]
                vertices = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]].copy()
                vertices = vertices[ConvexHull(vertices).vertices]
                self.feet.append((g, vertices))
        self.obstacles = {g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or '').startswith('hb_obstacle_geom_')}
        self.ground = m.geom("hb_carpet_collision").id
        self.reason = None
        self.event_time = None
        self.event_detail = None
        self.max_foot_width = 0.
        self.min_base_height = float('inf')
        self.partial_foot_overhang_frames = 0

    def fail(self, reason, detail=None):
        if self.reason is None:
            self.reason, self.event_time, self.event_detail = reason, float(self.robot.data.time), detail

    def contacts(self):
        m, d = self.robot.model, self.robot.data
        for contact in d.contact:
            if contact.dist > 0:
                continue
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & self.obstacles:
                names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) for g in pair]
                self.fail("obstacle_contact", names)
            if self.ground in pair:
                other = next(g for g in pair if g != self.ground)
                if m.geom_bodyid[other] not in self.foot_bodies:
                    self.fail("fall_ground_contact", m.body(int(m.geom_bodyid[other])).name)
        self.min_base_height = min(self.min_base_height, float(d.qpos[2]))
        if d.qpos[2] < .20 or d.xmat[self.robot.base].reshape(3, 3)[2, 2] < .5:
            self.fail("fall_posture")

    def boundary(self):
        d = self.robot.data
        polygons = []
        all_vertices = []
        for g, vertices in self.feet:
            world = vertices @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            polygons.append(Polygon(world[ConvexHull(world[:, :2]).vertices, :2]))
            all_vertices.append(world)
        yaw = self.robot.pose[2]
        lateral = np.vstack(all_vertices)[:, :2] @ np.array([-np.sin(yaw), np.cos(yaw)])
        self.max_foot_width = max(self.max_foot_width, float(np.ptp(lateral)))
        # Both projected foot polygons must entirely lose overlap to terminate.
        if all(not p.intersects(self.lane) for p in polygons):
            self.fail("both_feet_outside")
        if any(not self.lane.covers(p) for p in polygons):
            self.partial_foot_overhang_frames += 1
