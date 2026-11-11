"""Build a source-hashed, isolated Jetson navigation release; never launch motion."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navigation.runtime import core_hashes
from navigation.command import load_contract


def build(release_id, output, fixture=None):
    if not re.fullmatch(r'[a-z0-9_]+',release_id):raise ValueError('Invalid release identifier')
    destination=output/release_id
    destination.mkdir(parents=True,exist_ok=False)
    names=sorted({str(p.relative_to(ROOT)) for folder in ('navigation','deploy')
                  for p in (ROOT/folder).iterdir() if p.suffix in ('.py','.json','.cpp','.sh','.md','.txt')})
    names+=['sim/visual_navigation.py','scripts/test_shared_navigation.py','scripts/replay_navigation.py','SHARED_NAVIGATION_STATUS.md',
            *['scripts/test_'+name+'.py' for name in ('depth_ground','ground_color_integration','surface_odometry','rgbd_odometry','floor_odometry','floor_projection',
                                                      'map_relocalization','gap_controller','visual_finish','corridor_feedback','motion_recovery')]]
    files={}
    for name in names:
        source=ROOT/name;data=source.read_bytes()
        if name.endswith('.py'):ast.parse(data.decode(),feature_version=(3,8))
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        files[name]=hashlib.sha256(data).hexdigest()
    if fixture:
        target=destination/'validation/sensor_replay.npz';target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(fixture,target);files[str(target.relative_to(destination))]=hashlib.sha256(target.read_bytes()).hexdigest()
    manifest=dict(release_id=release_id,python='>=3.8',navigation_core_sha256=core_hashes(),
                  factory_contract=load_contract(),files=files,
                  default_mode='shadow_without_motor_publishers',
                  real_motion_requires='verified camera/IMU calibration, current sensors, immutable AMP identity, sole guarded publisher')
    (destination/'release_manifest.json').write_text(json.dumps(manifest,indent=2))
    archive=output/(release_id+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(destination.rglob('*')):
            if not p.is_file():continue
            data=p.read_bytes();info=tarfile.TarInfo(str(p.relative_to(destination)))
            info.size=len(data);info.mode=0o644;info.mtime=0;tar.addfile(info,io.BytesIO(data))
    result=dict(release_id=release_id,directory=str(destination.resolve()),archive=str(archive.resolve()),
                archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files))
    (output/(release_id+'.json')).write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',required=True);p.add_argument('--output',type=Path,default=ROOT/'output/deployment/shared_navigation')
    p.add_argument('--fixture',type=Path);args=p.parse_args()
    print(json.dumps(build(args.release,args.output,args.fixture),indent=2))
