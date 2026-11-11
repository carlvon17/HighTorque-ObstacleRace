"""Package the local MicroDuck project using BotLab's editable archive format."""
import json
import mimetypes
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from generate_botlab_field import generate_field

ROOT = Path(__file__).resolve().parents[1]
generate_field()
project = ROOT / "botlab/microduck_athletics"
sources = sorted(
    p.name for p in project.iterdir()
    if p.is_file() and (p.suffix == ".py" or p.name.endswith(".preset.yaml") or p.name == "README.md")
)
assets = sorted(p for p in (project / "assets").rglob("*") if p.is_file())
manifest = {
    "format": "botlab-project", "version": 1, "files": sources,
    "assets": [], "assetBase": "project", "assetPaths": [],
}
for index, asset in enumerate(assets):
    path = asset.relative_to(project).as_posix()
    manifest["assetPaths"].append(path)
    manifest["assets"].append({
        "resourceId": f"workspace-asset-{index}", "source": path, "path": path,
        "name": asset.name,
        "mimeType": mimetypes.guess_type(asset.name)[0] or "application/octet-stream",
        "role": "mesh",
    })
target = ROOT / "output/MicroDuck-Athletics.botlab.zip"
with ZipFile(target, "w", ZIP_DEFLATED, compresslevel=6) as archive:
    archive.writestr("botlab.project.json", json.dumps(manifest, indent=2))
    for source in sources:
        archive.write(project / source, source)
    for asset in assets:
        archive.write(asset, asset.relative_to(project))
with ZipFile(target) as archive:
    assert archive.testzip() is None
print(f"{target}: {target.stat().st_size / 1024**2:.1f} MiB")
