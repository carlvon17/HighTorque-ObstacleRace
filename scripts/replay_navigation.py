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


def replay(path, map_observations=None):
    expected_frames = (json.loads(map_observations.read_text()) if map_observations is not None else None)
    map_error = 0.
    map_state_differences = 0
    durations=[];actual=[];reasons={}
    with np.load(path,allow_pickle=False) as archive:
        data={name:archive[name] for name in archive.files}
        rt1 = str(data.get('navigation_profile','shared'))=='rt1'
        if rt1:
            from sim.rt1_control import PROFILE, RT1Control, camera_profile, navigation_sources
            camera=camera_profile()
            w,h=(round(camera[k]*camera['processing_scale']) for k in ('width','height'))
            navigation_sources()
            control=RT1Control(w,h,data['intrinsics'],float(data['speed']))
            runtime=control.runtime
            hashes=PROFILE['navigation_core_sha256']
        else:
            h,w=data['depth'].shape[1:]
            runtime=NavigationRuntime(w,h,data['intrinsics'],data['camera_position'],data['camera_rotation'],float(data['speed']))
            hashes=core_hashes()
        frame_count=0
        for i,t in enumerate(data['time']):
            start=time.perf_counter()
            t=float(t)
            attitude=data['attitude'][i]
            index=int(data['frame_index'][i])
            enabled=bool(data['enabled'][i])
            feedback=data['policy_feedback'][i]
            actual_frame=None
            if rt1:
                if index>=0:
                    control.capture(data['rgb'][index],data['depth'][index],attitude,enabled)
                command=control.tick(t,attitude,feedback,enabled)
                if control.frame_result is not None:
                    frame,observation,state,_=control.frame_result
                    if observation is not None:
                        stamp,actual_frame=frame.time,state
            else:
                runtime.set_attitude(t,attitude)
                if index>=0:
                    runtime.observe(CameraFrame(t,data['rgb'][index],data['depth'][index]))
                    stamp,actual_frame=t,runtime.diagnostic()
                command=runtime.command(t,feedback,enabled)
            if actual_frame is not None:
                if expected_frames is not None:
                    expected=expected_frames[frame_count]
                    if abs(expected['t']-stamp)>1e-6:
                        raise ValueError('Map reference timestamp differs from recorded camera frame')
                    fields=('floor_keyframes','map_revisits','map_tracking_updates','map_recoveries')
                    different=any(actual_frame[key]!=expected[key] for key in fields)
                    a,b=actual_frame['map_xy'],expected['map_xy']
                    if a is not None and b is not None:
                        difference=float(np.linalg.norm(np.subtract(a,b)))
                        if not np.isfinite(difference):
                            raise ValueError('Nonfinite position in mapping replay comparison')
                        map_error=max(map_error,difference)
                    map_state_differences+=int(different or (a is None)!=(b is None))
                frame_count+=1
                durations.append((time.perf_counter()-start)*1000)
            actual.append(command)
            reasons[runtime.reason]=reasons.get(runtime.reason,0)+1
        error=float(np.max(abs(np.array(actual)-data['expected_command'])))
    passed = error < 1e-5 and map_error < 1e-5 and map_state_differences == 0
    return dict(status='replay_passed' if passed else 'replay_output_differs',
                map_frames_compared=len(durations) if expected_frames is not None else 0,
                maximum_map_position_difference_m=map_error if expected_frames is not None else None,
                map_state_difference_frames=map_state_differences if expected_frames is not None else None,
                maximum_command_difference=error,cycles=len(actual),camera_frames=len(durations),
                camera_cycle_ms_p50=float(np.median(durations)) if durations else None,
                camera_cycle_ms_p95=float(np.percentile(durations,95)) if durations else None,
                reasons=reasons,navigation_core_sha256=hashes,last_navigation=runtime.diagnostic(),
                motor_publishers=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fixture',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--map-observations',type=Path,help='Compare every replayed camera frame to the recorded mapping telemetry')
    args=p.parse_args();result=replay(args.fixture, args.map_observations)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['status']=='replay_passed' else 2)
