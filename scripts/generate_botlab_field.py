"""Export the generated field as a finite UV-mapped carpet for BotLab."""
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/athletics/mujoco"
TARGET = ROOT / "botlab/microduck_athletics/assets/robot"


def generate_field():
    tree = ET.parse(SOURCE / "field.xml")
    root = tree.getroot()
    asset = root.find("asset")
    visual = root.find(".//geom[@name='hb_carpet_visual']")
    collision = root.find(".//geom[@name='hb_carpet_collision']")
    if asset is None or visual is None or collision is None:
        raise ValueError("Generated field is missing its carpet assets or geoms")
    x, y, half_thickness = map(float, collision.attrib["size"].split())
    thickness = 2 * half_thickness
    # Keep the existing top UVs and top height. Match the physical carpet's 12 mm
    # thickness rather than a 1 mm visual shell prone to self-shadowing artifacts.
    vertices = [
        (-x, -y, 0), (x, -y, 0), (x, y, 0), (-x, y, 0),
        (-x, -y, -thickness), (x, -y, -thickness),
        (x, y, -thickness), (-x, y, -thickness),
    ]
    mesh = ET.Element(
        "mesh", name="hb_carpet_surface",
        vertex=" ".join(f"{v:g}" for point in vertices for v in point),
        face="0 1 2 0 2 3 4 6 5 4 7 6 0 4 5 0 5 1 1 5 6 1 6 2 2 6 7 2 7 3 3 7 4 3 4 0",
        texcoord="0 0 1 0 1 1 0 1 0 0 1 0 1 1 0 1",
    )
    asset.insert(0, mesh)
    visual.set("type", "mesh")
    visual.set("mesh", "hb_carpet_surface")
    del visual.attrib["size"]
    ET.indent(root)
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "assets").mkdir(exist_ok=True)
    tree.write(TARGET / "athletics_field.xml", encoding="unicode")
    shutil.copyfile(SOURCE / "assets/carpet.png", TARGET / "assets/carpet.png")
    return TARGET / "athletics_field.xml"


if __name__ == "__main__":
    print(generate_field())
