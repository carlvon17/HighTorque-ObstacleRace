"""Run recorded sensor inputs through the shared core, without ROS or MuJoCo."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from navigation.runtime import CameraFrame, NavigationRuntime, core_hashes


def replay(path):
    durations=[];actual=[];reasons={}
    with np.load(path,allow_pickle=False) as archive:
        data={name:archive[name] for name in archive.files}
        h,w=data['depth'].shape[1:]
        runtime=NavigationRuntime(w,h,data['intrinsics'],data['camera_position'],data['camera_rotation'],float(data['speed']))
        for i,t in enumerate(data['time']):
            start=time.perf_counter()
            runtime.set_attitude(float(t),data['attitude'][i])
            index=int(data['frame_index'][i])
            if index>=0:runtime.observe(CameraFrame(float(t),data['rgb'][index],data['depth'][index]))
            command=runtime.command(float(t),data['policy_feedback'][i],bool(data['enabled'][i]))
            actual.append(command)
            if index>=0:durations.append((time.perf_counter()-start)*1000)
            reasons[runtime.reason]=reasons.get(runtime.reason,0)+1
        error=float(np.max(abs(np.array(actual)-data['expected_command'])))
    return dict(status='replay_passed' if error<1e-5 else 'replay_output_differs',
                maximum_command_difference=error,cycles=len(actual),camera_frames=len(durations),
                camera_cycle_ms_p50=float(np.median(durations)),camera_cycle_ms_p95=float(np.percentile(durations,95)),
                reasons=reasons,navigation_core_sha256=core_hashes(),last_navigation=runtime.diagnostic(),
                motor_publishers=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fixture',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args();result=replay(args.fixture)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['status']=='replay_passed' else 2)
