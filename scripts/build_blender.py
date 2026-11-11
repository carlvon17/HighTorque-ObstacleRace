"""Run with Blender --background --python scripts/build_blender.py."""
import bpy
import json
import math
import sys
from pathlib import Path
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from obstacle_parts import crate_boards
OUT = ROOT / "output/athletics"
C = json.loads((ROOT / "scene_config.json").read_text())
G = json.loads((OUT / "geometry.json").read_text())

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
for coll in list(bpy.data.collections):
    if coll.name != "Collection":
        bpy.data.collections.remove(coll)
base_collection = bpy.data.collections.get("Collection")
base_collection.name = "01_Field"


def collection(name):
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c


lines_col = collection("02_White_Tape_8mm")
obs_col = collection("03_Obstacles_200x200x400mm")
start_col = collection("04_Start_Positions")
guide_col = collection("05_Dimensions_Reference_Only")
stage_col = collection("06_Presentation")


def move(obj, coll):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)
    return obj


def linear(v):
    v /= 255
    return v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4


def material(name, rgb, roughness=0.8):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*[linear(v) for v in rgb], 1)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = m.diffuse_color
    bsdf.inputs["Roughness"].default_value = roughness
    return m


green = material("Emerald competition carpet", C["carpet_rgb"], .96)
white = material("Off-white boundary tape", [246,248,239])
red = material("Red start marks", [225,48,49])
guide_mat = material("Reference dimension ink", [220,236,242])
stage_mat = material("Studio background", [31,42,50])

# Fine surface shading only: collider remains perfectly flat.
nodes = green.node_tree.nodes
noise = nodes.new("ShaderNodeTexNoise")
noise.inputs["Scale"].default_value = 700
noise.inputs["Detail"].default_value = 2
texcoord = nodes.new("ShaderNodeTexCoord")
bump = nodes.new("ShaderNodeBump")
bump.inputs["Strength"].default_value = .12
bump.inputs["Distance"].default_value = .0007
green.node_tree.links.new(texcoord.outputs["Object"], noise.inputs["Vector"])
green.node_tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
green.node_tree.links.new(bump.outputs["Normal"], next(n for n in nodes if n.type == "BSDF_PRINCIPLED").inputs["Normal"])


def cube(name, pos, dims, mat, coll, bevel=0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dims
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    move(obj, coll)
    if bevel:
        mod = obj.modifiers.new("Small visual edge radius", "BEVEL")
        mod.width = bevel
        mod.segments = 3
    return obj


sx, sy, sz = C["carpet_dimensions"]
cx, cy = C["carpet_center"]
carpet = cube("Carpet_top_Z0", (cx,cy,-sz/2), (sx,sy,sz), green, base_collection)
carpet["collision"] = "finite box with top Z=0 in MJCF"
carpet["friction_assumption"] = C["friction"][0]


def ribbon(name, points, normals, width, mat, coll, z=.00025):
    verts, faces = [], []
    n = len(points)
    for p, normal in zip(points, normals):
        verts.extend([(p[0]+normal[0]*width/2,p[1]+normal[1]*width/2,z),
                      (p[0]-normal[0]*width/2,p[1]-normal[1]*width/2,z)])
    for i in range(n):
        j=(i+1)%n
        faces.append((2*i,2*i+1,2*j+1,2*j))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts,[],faces)
    mesh.update()
    obj=bpy.data.objects.new(name,mesh)
    coll.objects.link(obj)
    obj.data.materials.append(mat)
    obj["collision"] = False
    obj["width_m"] = width
    return obj


ribbon("Inner_boundary",G["inner"],G["normals"],C["line_width"],white,lines_col)
ribbon("Outer_boundary",G["outer"],G["normals"],C["line_width"],white,lines_col)
for o in C["obstacles"]:
    x,y = o["position"]
    dims=C["obstacle_dimensions"]
    bpy.ops.object.empty_add(type="PLAIN_AXES",location=(x,y,dims[2]/2))
    root=bpy.context.object
    root.name=f"Wood_crate_{o['id']}_assembly"
    root.rotation_euler.z=math.radians(o["yaw_degrees"])
    root["position_basis"]="Estimated from site photograph, not surveyed"
    move(root,obs_col)
    obj=cube(f"Obstacle_{o['id']}",(0,0,0),dims,material(f"Obstacle_{o['id']}_collision",o["rgb"]),obs_col)
    obj.parent=root
    obj["collision_shape"]="BOX"
    obj["static"]=True
    obj.hide_render=True
    obj.hide_set(True)
    for board in crate_boards(dims,o["rgb"]):
        mat=material(f"Wood_{o['id']}_{board['name']}",board["rgb"],.9)
        part=cube(f"Wood_{o['id']}_{board['name']}",board["pos"],board["size"],mat,obs_col,.0007)
        part.parent=root
        part["collision"]=False

for s in C["starts"]:
    x,y=s["position"]
    bpy.ops.mesh.primitive_cylinder_add(vertices=48,radius=.03,depth=.0003,location=(x,y,.0002))
    marker=bpy.context.object
    marker.name=f"Start_{s['id']}_paint"
    marker.data.materials.append(red)
    move(marker,start_col)
    bpy.ops.object.empty_add(type="ARROWS",location=(x,y,0))
    empty=bpy.context.object
    empty.name=f"Spawn_{s['id']}_CCW_positive_X"
    empty.empty_display_size=.16
    empty["ground_position_m"]=[x,y,0.0]
    empty["base_height"]="Set for your robot"
    move(empty,start_col)


def curve(name, coords, radius, mat, coll):
    data=bpy.data.curves.new(name,"CURVE")
    data.dimensions="3D"
    data.bevel_depth=radius
    data.bevel_resolution=2
    spline=data.splines.new("POLY")
    spline.points.add(len(coords)-1)
    for p,co in zip(spline.points,coords):
        p.co=(*co,1)
    obj=bpy.data.objects.new(name,data)
    coll.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def label(text,pos,size=.15):
    data=bpy.data.curves.new(text,"FONT")
    data.body=text
    data.size=size
    data.align_x="CENTER"
    obj=bpy.data.objects.new(text,data)
    guide_col.objects.link(obj)
    obj.location=pos
    obj.data.materials.append(guide_mat)
    return obj


def dimension(a,b,text,textpos):
    z=.007
    curve("dimension "+text,[(a[0],a[1],z),(b[0],b[1],z)],.006,guide_mat,guide_col)
    v=(Vector(b)-Vector(a)).normalized()
    n=Vector((-v.y,v.x))
    for p,sign in [(Vector(a),1),(Vector(b),-1)]:
        tip1=p+sign*v*.10+n*.04
        tip2=p+sign*v*.10-n*.04
        curve("arrow "+text,[(tip1.x,tip1.y,z),(p.x,p.y,z),(tip2.x,tip2.y,z)],.006,guide_mat,guide_col)
    label(text,(*textpos,.01))


dimension((0,0),(5,0),"5.00 m  /  circle centers",(2.5,.12))
dimension((0,0),(-.707,.707),"R 1.00 m",(-.7,.96))
dimension((5,0),(6.414,1.414),"R 2.00 m",(5.9,.38))
dimension((0,-1),(0,-1.5),"0.50 m",(-.52,-1.35))
label("2026  HUABEI  /  HUMANOID ATHLETICS",(3,2.83,.01),.20)
label("METRES   |   COUNTERCLOCKWISE   |   8 mm TAPE",(3,-2.91,.01),.13)
for s in C["starts"]:
    label("START "+s["id"],(s["position"][0],-1.78,.01),.10)
label("3 x 0.20 x 0.20 x 0.40 m",(1.3,2.16,.01),.12)
label("PHOTO ESTIMATE: INNER / OUTER / INNER",(1.3,1.95,.01),.10)
label("0.50 m longitudinal centers (estimated)",(1.3,1.77,.01),.09)
guide_col.hide_render=True
guide_col.hide_viewport=True

cube("Presentation_ground_only",(3,0,-.10),(200,200,.10),stage_mat,stage_col)


def camera(name,location,target,scale):
    data=bpy.data.cameras.new(name)
    data.type="ORTHO"
    data.ortho_scale=scale
    obj=bpy.data.objects.new(name,data)
    stage_col.objects.link(obj)
    obj.location=location
    obj.rotation_euler=(Vector(target)-obj.location).to_track_quat("-Z","Y").to_euler()
    return obj


hero=camera("Overview",(9,-11,13),(3,0,0),12.5)
top=camera("Top_plan",(3,0,14),(3,0,0),11.4)
top.rotation_euler=(0,0,0)
for name,loc,power,size in [("Key",(0,-4,9),1500,8),("Fill",(7,3,7),1100,7)]:
    data=bpy.data.lights.new(name,"AREA")
    data.energy=power
    data.shape="DISK"
    data.size=size
    obj=bpy.data.objects.new(name,data)
    stage_col.objects.link(obj)
    obj.location=loc
    obj.rotation_euler=(Vector((3,0,0))-obj.location).to_track_quat("-Z","Y").to_euler()

scene=bpy.context.scene
scene.unit_settings.system="METRIC"
scene.unit_settings.scale_length=1
scene.unit_settings.length_unit="METERS"
scene.render.engine="CYCLES"
scene.cycles.samples=48
scene.cycles.use_denoising=True
scene.render.resolution_x=2400
scene.render.resolution_y=1600
scene.render.resolution_percentage=100
scene.world.color=(.20,.20,.20)
scene.view_settings.view_transform="AgX"
scene.camera=hero
scene.render.image_settings.file_format="PNG"
scene["source"]="Rulebook page 28 Figure 2.1.1; obstacle layout/appearance revised using supplied site photograph"
scene["obstacle_layout"]="Photo estimate: inner / outer / inner; longitudinal spacing 0.50 m; not surveyed"
scene["perimeter_note"]="Diagram dimensions retained. Modeled inner circumference 19.9927 m; rule text says 19 m."
scene["direction"]="Counterclockwise. Bottom straight travels along +X."
for area in bpy.context.screen.areas:
    if area.type=="VIEW_3D":
        area.spaces.active.region_3d.view_perspective="CAMERA"
        area.spaces.active.region_3d.view_camera_zoom=18
        area.spaces.active.overlay.show_overlays=False
        area.spaces.active.shading.type="MATERIAL"
scene.render.filepath=str(OUT / "overview.png")
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "athletics.blend"))
bpy.ops.render.render(write_still=True)
scene.camera=top
scene.cycles.samples=32
scene.render.filepath=str(OUT / "top_view.png")
bpy.ops.render.render(write_still=True)
guide_col.hide_render=False
guide_col.hide_viewport=False
scene.render.filepath=str(OUT / "dimensioned_plan.png")
bpy.ops.render.render(write_still=True)
guide_col.hide_render=True
guide_col.hide_viewport=True
scene.camera=camera("Obstacle_detail",(2.7,-1.0,1.65),(1.15,1.25,.2),2.2)
scene.render.filepath=str(OUT / "obstacle_detail.png")
bpy.ops.render.render(write_still=True)
print("BLENDER_BUILD_COMPLETE")
