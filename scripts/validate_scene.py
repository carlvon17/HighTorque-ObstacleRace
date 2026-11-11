"""Verify geometry, actual MuJoCo contacts, relocatable includes and rendered assets."""
import json
import math
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/athletics"
MJ = OUT / "mujoco"
C = json.loads((ROOT / "scene_config.json").read_text())
G = json.loads((OUT / "geometry.json").read_text())
model = mujoco.MjModel.from_xml_path(str(MJ / "scene.xml"))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
inner, outer = np.array(G["inner"]), np.array(G["outer"])
assert np.isfinite(inner).all() and np.isfinite(outer).all()
widths = np.linalg.norm(outer-inner, axis=1)
assert np.max(abs(widths-C["track_width"])) < 1e-9
assert model.ngeom == 35 and model.nsite == 3 and model.nq == 0
collision_count=int(np.count_nonzero(model.geom_contype | model.geom_conaffinity))
assert collision_count == 4
assert model.geom_contype[model.geom("hb_carpet_visual").id] == 0
for o in C["obstacles"]:
    g=model.geom(f"hb_obstacle_geom_{o['id']}")
    assert np.allclose(g.size*2, C["obstacle_dimensions"])
    x,y=o["position"]
    assert np.allclose(data.geom_xpos[g.id], [x,y,.2])
    assert 1-1e-9 <= y-.1 and y+.1 <= 1.5+1e-9
    edge_y = y-.1 if o["edge"] == "inner" else y+.1
    assert abs(edge_y-(1 if o["edge"] == "inner" else 1.5)) < 1e-9
    wood=[i for i in range(model.ngeom) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or "").startswith(f"hb_wood_{o['id']}_")]
    assert len(wood) == 10
    for i in wood:
        assert model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0
        assert np.all(np.abs(model.geom_pos[i])+model.geom_size[i] <= np.array(C["obstacle_dimensions"])/2+1e-9)
for s in C["starts"]:
    site=model.site(f"hb_start_{s['id']}")
    assert np.allclose(data.site_xpos[site.id][:2], s["position"])

# Move the include beneath a new project root: validates texture dependency resolution.
with tempfile.TemporaryDirectory(prefix="athletics-qa-") as td:
    tmp=Path(td)
    import shutil
    shutil.copytree(MJ,tmp / "assets/scene/athletics")
    root=ET.Element("mujoco",model="contact_probe")
    ET.SubElement(root,"option",timestep="0.002",gravity="0 0 -9.81")
    ET.SubElement(root,"include",file="assets/scene/athletics/field.xml")
    world=ET.SubElement(root,"worldbody")
    probe=ET.SubElement(world,"body",name="qa_probe",pos="0.55 -1.25 0.5")
    ET.SubElement(probe,"freejoint")
    ET.SubElement(probe,"geom",type="sphere",size="0.025",mass="0.1",friction="0.8 0.005 0.0001")
    ET.ElementTree(root).write(tmp / "robot.xml",encoding="unicode")
    pm=mujoco.MjModel.from_xml_path(str(tmp / "robot.xml"))
    pd=mujoco.MjData(pm)
    results=[]
    for label,xy,height in [("start_"+s["id"],s["position"],0.025) for s in C["starts"]]+[("obstacle_"+o["id"],o["position"],0.425) for o in C["obstacles"]]:
        mujoco.mj_resetData(pm,pd)
        pd.qpos[:3]=[*xy,.8]
        pd.qpos[3:7]=[1,0,0,0]
        for _ in range(1000):
            mujoco.mj_step(pm,pd)
        assert abs(pd.qpos[2]-height) < .001,(label,pd.qpos[2],height)
        assert pd.ncon > 0
        assert np.isfinite(pd.qpos).all()
        results.append({"probe":label,"resting_center_z_m":float(pd.qpos[2]),"expected_z_m":height,"contacts":pd.ncon})

with mujoco.Renderer(model,height=1280,width=1920) as renderer:
    renderer.update_scene(data,camera="hb_top")
    Image.fromarray(renderer.render()).save(OUT / "mujoco_top.png")
    cam=mujoco.MjvCamera()
    cam.lookat[:]=[3,0,0]
    cam.distance=12
    cam.azimuth=100
    cam.elevation=-55
    renderer.update_scene(data,camera=cam)
    Image.fromarray(renderer.render()).save(OUT / "mujoco_overview.png")

report={"status":"passed","mujoco_version":mujoco.__version__,"track_width_min_m":float(widths.min()),"track_width_max_m":float(widths.max()),"static_collision_geoms":collision_count,"noncolliding_visual_geoms":model.ngeom-collision_count,"spawn_sites":3,"obstacle_dimensions_m":C["obstacle_dimensions"],"obstacle_positions_xy_m":[o["position"] for o in C["obstacles"]],"edge_order":[o["edge"] for o in C["obstacles"]],"visual_boards_inside_collision_envelope":True,"layout_accuracy":"photo estimate; not surveyed","relocated_include_and_texture":"passed","sphere_drop_tests":results,"robot_controller_tested":False}
(OUT / "validation.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
