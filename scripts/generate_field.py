"""Generate metric track geometry, painted carpet texture, and portable MJCF."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
from obstacle_parts import crate_boards

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.field import FIELD_DIRS
C = json.loads((ROOT / "scene_config.json").read_text())


def smooth_geometry():
    r, R, d = C["small_radius"], C["large_radius"], C["center_distance"]
    angle = math.radians(C["large_arc_join_degrees"])
    p0 = np.array([C["transition_start_x"], r])
    p1 = p0 + [C["transition_handle_start"], 0]
    p3 = np.array([d + R * math.cos(angle), R * math.sin(angle)])
    p2 = p3 - C["transition_handle_end"] * np.array([1., 1.])
    t = np.linspace(0, 1, 180)[:, None]
    bezier = (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t*t*p2 + t**3*p3
    straight = np.column_stack([np.linspace(0, p0[0], 240), np.full(240, r)])
    theta = np.linspace(angle, -angle, 900)
    big = np.column_stack([d+R*np.cos(theta), R*np.sin(theta)])
    lower_b = bezier[::-1] * [1, -1]
    lower_s = straight[::-1] * [1, -1]
    theta = np.linspace(-math.pi/2, -3*math.pi/2, 450)
    small = np.column_stack([r*np.cos(theta), r*np.sin(theta)])
    inner = np.concatenate([straight[:-1], bezier[:-1], big[:-1], lower_b[:-1], lower_s[:-1], small[:-1]])
    tangent = np.roll(inner, -1, axis=0) - np.roll(inner, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    outer = inner + normal*C["track_width"]
    center = inner + normal*C["track_width"]/2
    return inner, outer, center, normal


def length(points):
    return float(np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1).sum())


def corner_contour(offset):
    # Each tape intersects its own concentric arc. The outer tape reaches
    # its corner earlier than the inner tape when travelling along +X.
    small = C["small_radius"] + offset
    large = C["large_radius"] + offset
    distance = C["center_distance"]
    join_x = distance - math.sqrt(large**2 - small**2)
    angle = math.atan2(small, join_x - distance)
    upper = np.column_stack([np.linspace(0, join_x, 240), np.full(240, small)])
    theta = np.linspace(angle, -angle, 900)
    big_arc = np.column_stack([distance + large*np.cos(theta), large*np.sin(theta)])
    lower = upper[::-1] * [1, -1]
    theta = np.linspace(-math.pi/2, -3*math.pi/2, 450)
    small_arc = small * np.column_stack([np.cos(theta), np.sin(theta)])
    return np.concatenate([upper[:-1], big_arc[:-1], lower[:-1], small_arc[:-1]])


def tape_miters(points):
    # Offset adjacent segments by the tape half-width and intersect their
    # offset lines at each vertex. This keeps the tape continuous at a kink.
    segments = np.roll(points, -1, axis=0) - points
    segments /= np.linalg.norm(segments, axis=1)[:, None]
    outgoing = np.column_stack([-segments[:, 1], segments[:, 0]])
    incoming = np.roll(outgoing, 1, axis=0)
    return (incoming + outgoing) / (1 + (incoming*outgoing).sum(axis=1))[:, None]


def generate(profile):
    out = FIELD_DIRS[profile]
    mj = out / "mujoco"
    (mj / "assets").mkdir(parents=True, exist_ok=True)
    if profile == "smooth":
        inner, outer, center, inner_normals = smooth_geometry()
        outer_normals = inner_normals
        bend_note = "The transition curve is a tangent-continuous cubic Bezier, not dimensioned in the source."
    else:
        width = C["track_width"]
        inner, outer, center = (corner_contour(offset) for offset in (0., width, width/2))
        inner_normals, outer_normals = tape_miters(inner), tape_miters(outer)
        bend_note = "Straight tape meets the large circle at their intersection, mirrored on the return side. The photo identifies a sharp join; its angle was not surveyed."
    data = {"inner": inner.tolist(), "outer": outer.tolist(), "center": center.tolist(), "normals": inner_normals.tolist()}
    if profile == "corner":
        data["outer_normals"] = outer_normals.tolist()
    (out / "geometry.json").write_text(json.dumps(data))
    metrics = {
        "bend_profile": profile,
        "inner_boundary_centerline_perimeter_m": length(inner),
        "outer_boundary_centerline_perimeter_m": length(outer),
        "lane_centerline_perimeter_m": length(center),
        "boundary_centerline_separation_m": C["track_width"],
        "clear_width_between_tape_edges_m": C["track_width"]-C["line_width"],
        "rule_text_inner_perimeter_m": C["reference_perimeter_m"],
        "perimeter_difference_m": length(inner)-C["reference_perimeter_m"],
        "obstacle_layout_basis": C["obstacle_layout_basis"],
        "obstacle_centers_xy_m": [o["position"] for o in C["obstacles"]],
        "obstacle_inner_outer_order": [o["edge"] for o in C["obstacles"]],
        "obstacle_longitudinal_spacing_m": [C["obstacles"][i+1]["position"][0]-C["obstacles"][i]["position"][0] for i in range(2)],
        "obstacle_edge_alignment": C["obstacle_edge_alignment"],
        "obstacle_appearance": C["obstacle_appearance"],
        "obstacle_colors_rgb": [o["rgb"] for o in C["obstacles"]],
        "assumptions": [
            "The diagram's 1 m / 2 m arrows are interpreted as inner-boundary radii.",
            "The 5 m arrow is interpreted as the two circle centers' separation.",
            bend_note,
            "The 0.5 m lane width is measured between boundary tape centerlines; clear gap is 0.492 m.",
            "The site photograph supports inner/outer/inner staggering. The 0.5 m longitudinal spacing, 0.65 m first station, exact tape alignment, board construction and yaw remain estimates, not surveyed coordinates.",
            "Red, yellow and green crate paint is a requested perception robustness test, not an assumed competition color requirement.",
            "Painted wood boards are visual only. Each obstacle retains a solid 0.2 x 0.2 x 0.4 m collision envelope.",
            "Start marker diameter (6 cm), carpet extent, thickness, and friction are modeling assumptions.",
            "Obstacle boxes are fixed; the reference provides dimensions but no mass or dynamics.",
            "No competition robot or control policy is included."
        ]
    }
    (out / "measurements.json").write_text(json.dumps(metrics, indent=2))
    width, height = C["texture_resolution"]
    cx, cy = C["carpet_center"]
    sx, sy, _ = C["carpet_dimensions"]
    xmin, ymin = cx-sx/2, cy-sy/2

    def pixel(p):
        return ((p[0]-xmin)/sx*(width-1), (1-(p[1]-ymin)/sy)*(height-1))

    # Mild deterministic grain, without adding geometric bumps to the collision surface.
    rng = np.random.default_rng(2026)
    grain = rng.integers(-3, 4, (height, width, 1), dtype=np.int16)
    rgb = np.clip(np.array(C["carpet_rgb"], dtype=np.int16)+grain, 0, 255).astype(np.uint8)
    im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(im)
    for points, normals in ((inner, inner_normals), (outer, outer_normals)):
        # Separate closed polygons ensure no fill across the entire interior.
        outside = [pixel(p) for p in points+normals*C["line_width"]/2]
        inside = [pixel(p) for p in points-normals*C["line_width"]/2]
        mask = Image.new("L", im.size)
        md = ImageDraw.Draw(mask)
        md.polygon(outside, fill=255)
        md.polygon(inside, fill=0)
        im.paste((246, 248, 239), mask=mask)
    draw = ImageDraw.Draw(im)
    for start in C["starts"]:
        x, y = start["position"]
        a, b = pixel((x-.03, y+.03)), pixel((x+.03, y-.03))
        draw.ellipse([a, b], fill=(225, 48, 49))
    im.save(mj / "assets/carpet.png", compress_level=6)

    root = ET.Element("mujocoinclude")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="hb_carpet_texture", type="2d", file="assets/carpet.png")
    ET.SubElement(asset, "material", name="hb_carpet_material", texture="hb_carpet_texture", texrepeat="1 1", texuniform="false", rgba="1 1 1 1", reflectance="0", specular="0", shininess="0")
    world = ET.SubElement(root, "worldbody")
    field = ET.SubElement(world, "body", name="hb_field", pos="0 0 0")
    # MJCF box texture projection differs by face; a plane gives a stable top-face UV mapping.
    ET.SubElement(field, "geom", name="hb_carpet_visual", type="plane", pos=f"{cx} {cy} 0.0001", size=f"{sx/2} {sy/2} 0.01", material="hb_carpet_material", contype="0", conaffinity="0", group="1")
    thickness = C["carpet_dimensions"][2]
    ET.SubElement(field, "geom", name="hb_carpet_collision", type="box", pos=f"{cx} {cy} {-thickness/2}", size=f"{sx/2} {sy/2} {thickness/2}", rgba="0 0 0 0", friction=" ".join(map(str, C["friction"])), condim="3", contype="1", conaffinity="1", group="3")
    for obstacle in C["obstacles"]:
        x, y = obstacle["position"]
        dx, dy, dz = C["obstacle_dimensions"]
        angle = math.radians(obstacle["yaw_degrees"])/2
        body = ET.SubElement(field, "body", name=f"hb_obstacle_{obstacle['id']}", pos=f"{x} {y} {dz/2}", quat=f"{math.cos(angle)} 0 0 {math.sin(angle)}")
        ET.SubElement(body, "geom", name=f"hb_obstacle_geom_{obstacle['id']}", type="box", size=f"{dx/2} {dy/2} {dz/2}", rgba="0 0 0 0", group="3", friction=" ".join(map(str, C["friction"])), contype="1", conaffinity="1")
        for board in crate_boards(C["obstacle_dimensions"], obstacle["rgb"]):
            rgba = " ".join(f"{v/255:.6f}" for v in board["rgb"])+" 1"
            ET.SubElement(body, "geom", name=f"hb_wood_{obstacle['id']}_{board['name']}", type="box", pos=" ".join(map(str, board["pos"])), size=" ".join(str(v/2) for v in board["size"]), rgba=rgba, group="1", contype="0", conaffinity="0")
    for start in C["starts"]:
        x, y = start["position"]
        ET.SubElement(field, "site", name=f"hb_start_{start['id']}", pos=f"{x} {y} 0.002", type="sphere", size="0.015", rgba="0 0 0 0", group="3")
    ET.indent(root)
    ET.ElementTree(root).write(mj / "field.xml", encoding="unicode", xml_declaration=False)
    scene = ET.Element("mujoco", model=C["name"])
    ET.SubElement(scene, "compiler", angle="radian", autolimits="true")
    ET.SubElement(scene, "option", timestep="0.002", gravity="0 0 -9.81")
    visual = ET.SubElement(scene, "visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1280")
    ET.SubElement(visual, "headlight", diffuse="0.6 0.6 0.6", ambient="0.25 0.25 0.25", specular="0 0 0")
    ET.SubElement(visual, "map", znear="0.01")
    ET.SubElement(scene, "statistic", center="3 0 0", extent="6")
    ET.SubElement(scene, "include", file="field.xml")
    world = ET.SubElement(scene, "worldbody")
    ET.SubElement(world, "light", name="hb_sun", pos="1 -3 7", dir="0.2 0.3 -1", directional="true", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "camera", name="hb_top", pos="3 0 11", xyaxes="1 0 0 0 1 0", fovy="36")
    ET.indent(scene)
    ET.ElementTree(scene).write(mj / "scene.xml", encoding="unicode", xml_declaration=False)
    # Counterclockwise samples; useful as geometric reference, not a collision-avoiding policy.
    ccw = center[::-1]
    tangents = np.roll(ccw, -1, axis=0)-np.roll(ccw, 1, axis=0)
    distances = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(ccw, axis=0), axis=1))])
    with (out / "centerline.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["s_m", "x_m", "y_m", "yaw_rad"])
        for s, p, v in zip(distances, ccw, tangents):
            writer.writerow([f"{s:.7f}", f"{p[0]:.7f}", f"{p[1]:.7f}", f"{math.atan2(v[1], v[0]):.7f}"])
    (out / "spawn_points.json").write_text(json.dumps({"frame": "Z up, metres, origin at small-circle center on carpet top", "direction": "counterclockwise", "z_note": "Positions specify ground contact. Add robot-specific base standing height.", "starts": C["starts"]}, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bend-profile", choices=FIELD_DIRS, help="Generate only this course (default: both).")
    profile = parser.parse_args().bend_profile
    for selected in ([profile] if profile else FIELD_DIRS):
        generate(selected)
