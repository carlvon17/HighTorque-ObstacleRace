"""Run autonomous RGB-D navigation; truth is used only by world setup/referee."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import shutil
import platform
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from sim.robot import Robot,build_model
from sim.navigation import Track
from sim.scoring import Referee
from sim.sensors import SensorRig
from sim.sensor_faults import SensorFaults
from navigation.runtime import NavigationRuntime,core_hashes
from navigation.command import verify_device_files
from navigation.environment import verify_dependencies
from deploy.geometry import quaternion_matrix_xyzw


def visual_world(output,start,no_obstacles=False,shift=0.,camera_profile=None):
    _,xml=build_model(no_obstacles=no_obstacles)
    root=ET.fromstring(xml)
    camera=root.find("worldbody/body/camera[@name='nav_camera']")
    camera.set('xyaxes','0 -1 0 0.7071067811865476 0 0.7071067811865476')
    camera.set('fovy','80')
    if camera_profile is not None:
        w,h=camera_profile['width'],camera_profile['height']
        camera.attrib.pop('fovy')
        camera.set('resolution',f'{w} {h}')
        # Sensor size is a numerical normalization; only its ratios with focal
        # length matter. It does not claim the real sensor's physical size.
        camera.set('sensorsize',f'{w*1e-5} {h*1e-5}')
        camera.set('focalpixel',f"{camera_profile['fx']} {camera_profile['fy']}")
        # MuJoCo offsets use the image-plane convention; OpenCV uses top-left
        # pixel centers. Verified against rendered off-axis calibration points.
        camera.set('principalpixel',f"{(w-1)/2-camera_profile['cx']} {(h-1)/2-camera_profile['cy']}")
    config=json.loads((ROOT/'scene_config.json').read_text())
    texture=root.find("asset/texture[@name='hb_carpet_texture']")
    im=Image.open(texture.get('file')).convert('RGB');draw=ImageDraw.Draw(im)
    size=np.array(config['carpet_dimensions'][:2]);low=np.array(config['carpet_center'])-size/2
    def px(x,y):return ((x-low[0])/size[0]*(im.width-1),(1-(y-low[1])/size[1])*(im.height-1))
    x,y=start['position']
    # Explicitly modeled visual finish stripe; no contact geometry and no
    # navigation side channel carrying its coordinates.
    draw.rectangle([px(x-.015,y+.25),px(x+.015,y-.25)],fill=(246,248,239))
    path=output/'camera_carpet.png';im.save(path)
    texture.set('file',str(path.resolve()))
    if shift:
        for b in root.findall("worldbody/body/body"):
            if b.get('name','').startswith('hb_obstacle_'):
                p=np.fromstring(b.get('pos'),sep=' ');p[0]+=shift;b.set('pos',' '.join(map(str,p)))
    xml=ET.tostring(root,encoding='unicode')
    return mujoco.MjModel.from_xml_string(xml),xml


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--start',default='01',choices=['01','02','03'])
    ap.add_argument('--duration',type=float,default=240)
    ap.add_argument('--output',type=Path,default=ROOT/'output/sim/visual_01')
    ap.add_argument('--no-obstacles',action='store_true')
    ap.add_argument('--obstacle-shift',type=float,default=0)
    ap.add_argument('--speed',type=float,default=.22)
    ap.add_argument('--seed',type=int,default=1)
    ap.add_argument('--no-noise',action='store_true')
    ap.add_argument('--blind-after',type=float,default=float('inf'))
    ap.add_argument('--spawn',type=float,nargs=3,metavar=('X','Y','YAW'),help='World-only test spawn; never passed to navigation')
    ap.add_argument('--video',action='store_true')
    ap.add_argument('--record-sensors',type=float,default=0.,help='Save the first N seconds of measured inputs for portable replay')
    ap.add_argument('--sensor-faults',type=Path,help='World-side RGB-D fault schedule for regression tests')
    ap.add_argument('--camera-profile',choices=['d435i','legacy'],default='d435i',
                    help='Real RGB intrinsics/640x480 -> 320x240/15 Hz, or original 80-degree camera')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    track=Track();start=next(s for s in track.config['starts'] if s['id']==args.start)
    camera_profile=json.loads((ROOT/'sim/d435i_camera.json').read_text()) if args.camera_profile=='d435i' else None
    model,xml=visual_world(args.output,start,args.no_obstacles,args.obstacle_shift,camera_profile)
    (args.output/'scene.xml').write_text(xml)
    robot=Robot(model,args.spawn or [*start['position'],np.deg2rad(start['yaw_degrees'])])
    rig=SensorRig(model,args.seed,not args.no_noise,camera_profile)
    faults=SensorFaults(args.sensor_faults,rig,track)
    verify_device_files(ROOT/'vendor/pi_plus_pro',pd_path=ROOT/'vendor/pi_plus_pro/pd.yaml')
    odom=NavigationRuntime(rig.width,rig.height,rig.intrinsics,rig.camera_position,rig.camera_rotation,args.speed)
    visual_odometry=odom.odometry
    nav=odom.navigator
    referee=Referee(robot,track)
    rows=[];states=[];frames=[];observations=[];replay=[];replay_rgb=[];replay_depth=[]
    next_camera,next_log=0.,10.;finish_time=None;status='duration_limit';blocked_since=None
    video=imageio.get_writer(str(args.output/'perception.mp4'),fps=rig.fps,macro_block_size=2) if args.video else None
    camera_settings={'profile':args.camera_profile,'capture_resolution':[rig.capture_width,rig.capture_height],
                     'processing_resolution':[rig.width,rig.height],'fps':rig.fps,'intrinsics':rig.intrinsics,
                     'position_body_m':rig.camera_position.tolist(),'rotation_optical_to_body':rig.camera_rotation.tolist(),
                     'timing_note':'Nominal camera deadlines sampled at the next 50 Hz control tick; actual times saved in observations.json.'}
    (args.output/'camera_settings.json').write_text(json.dumps(camera_settings,indent=2))
    manifest={'arguments':{k:str(v) if isinstance(v,Path) else (None if isinstance(v,float) and not np.isfinite(v) else v)
                           for k,v in vars(args).items()},
              'camera':camera_settings,'locomotion':robot.locomotion_metadata(),'navigation_core_sha256':core_hashes(),'source_sha256':{},
              'runtime_environment':dict(python=platform.python_version(),**verify_dependencies())}
    manifest['sensor_faults']=faults.schedule
    for name in ['sim/robot.py','sim/device_amp.py','sim/navigation.py','sim/scoring.py','sim/sensors.py','sim/sensor_faults.py','sim/visual_navigation.py',
                 'vendor/pi_plus_pro/SOURCE.json','vendor/pi_plus_pro/amp.yaml',
                 'sim/d435i_camera.json','scripts/run_visual.py','scene_config.json','upstream.lock.json',
                 *['navigation/'+name for name in core_hashes()]]:
        path=ROOT/name;target=args.output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);manifest['source_sha256'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    wall=time.monotonic();last_s=track.s[track.tree.query(robot.pose[:2])[1]];progress=0.
    try:
        while robot.data.time<args.duration:
            t=float(robot.data.time)
            measured=rig.proprioception(robot.data)
            q=measured.orientation_wxyz
            measured_rotation=quaternion_matrix_xyzw(np.r_[q[1:],q[0]])
            odom.set_attitude(t,measured_rotation)
            pose=odom.pose
            recorded_frame=-1
            if t>=next_camera-1e-7:
                frame=rig.camera(robot.data)
                frame=faults.apply(frame,robot.data,robot.base)
                if t>=args.blind_after:frame.rgb[:]=0;frame.depth[:]=np.nan
                observation=odom.observe(frame)
                if t<args.record_sensors:
                    recorded_frame=len(replay_rgb);replay_rgb.append(frame.rgb.copy());replay_depth.append(frame.depth.copy())
                pose=odom.pose
                observations.append({'t':t,'mode':nav.mode,'lane':observation.lane.tolist() if observation.lane is not None else None,'boxes':observation.obstacles.tolist(),'finish':observation.finish.tolist(),'vo_inliers':visual_odometry.inliers,'sensor_faults':faults.active,**odom.diagnostic()})
                if video:
                    overlay=observation.overlay.copy()
                    cv2.putText(overlay,f'RGB-D NAV | {t:.2f}s | {nav.mode}',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
                    video.append_data(overlay)
                if round(t*10)%100==0:
                    Image.fromarray(observation.overlay).save(args.output/f'camera_{t:06.1f}.png')
                next_camera+=1/rig.fps
            feedback=robot.device.command.copy()
            command=odom.command(t,feedback,enabled=t>=2)
            if t<args.record_sensors:replay.append((t,measured_rotation,feedback,recorded_frame,t>=2,command.copy()))
            robot.step(command,referee.contacts);referee.boundary()
            # Ground truth is below this boundary: evaluation and telemetry only.
            current=track.s[track.tree.query(robot.pose[:2])[1]]
            progress+=(current-last_s+track.length/2)%track.length-track.length/2;last_s=current
            rows.append([robot.data.time,*robot.pose,*pose,*command,progress,odom.turn,odom.distance,*robot.device.command])
            states.append(robot.data.qpos.copy())
            if referee.reason:status=referee.reason;break
            if nav.mode=='blocked':
                if blocked_since is None:blocked_since=t
                elif t-blocked_since>10:status='navigation_blocked';break
            else:blocked_since=None
            if nav.finished and finish_time is None:finish_time=float(robot.data.time)
            if finish_time is not None and robot.data.time-finish_time>=3:status='completed';break
            if robot.data.time>=next_log:
                print(f't={robot.data.time:.1f} progress(eval)={progress:.2f} pose(eval)={robot.pose.round(3)} odom={pose.round(3)} mode={nav.mode} lane={observation.lane} boxes={observation.obstacles.round(3)}',flush=True)
                next_log+=10
    except KeyboardInterrupt:
        status='interrupted'
    finally:
        rig.close()
        if video:video.close()
    if observations:
        Image.fromarray(observation.overlay).save(args.output/'camera_last.png')
        Image.fromarray(frame.rgb).save(args.output/'camera_last_rgb.png')
        np.save(args.output/'camera_last_depth.npy',frame.depth)
    summary={'status':status,'navigation':'shared_RGBD_navigation_with_factory_AMP_adapter','start':args.start,'finish_time_s':finish_time,'sim_time_s':robot.data.time,'wall_time_s':time.monotonic()-wall,'truth_progress_evaluation_only_m':progress,'track_length_evaluation_only_m':track.length,'estimated_turn_rad':odom.turn,'estimated_distance_m':odom.distance,'event_detail':referee.event_detail,'event_time_s':referee.event_time,'sensor_noise':not args.no_noise,'obstacle_shift_world_only_m':args.obstacle_shift,'mode':nav.mode,'controller_inputs':['RGB','metric depth','simulated noisy IMU attitude','policy command feedback'],'excluded_inputs':['world position','world velocity','track geometry','obstacle poses','segmentation','truth contact state'],'finish_basis':'camera transverse stripe after estimated heading turn > 5.5 rad; short-range odometry tracks last observed stripe','limitations':['AHRS attitude measurement is simulated, not an implemented inertial filter.','Camera extrinsics and robot dynamics still require physical Pro calibration.','A 3 cm transverse finish stripe was added to the selected start as an explicit visual world assumption.','No sim-to-real claim; ground truth is used by independent referee only.'],'navigation_core_sha256':manifest['navigation_core_sha256'],'last_navigation':odom.diagnostic(),'odometry':'shared RGBD + AHRS; no simulation-only kinematic fallback'}
    camera_times=np.array([o['t'] for o in observations])
    summary['locomotion']=robot.locomotion_metadata()
    summary['sensor_faults']=faults.schedule
    summary['policy_path']=str(robot.policy_path)
    summary['camera']=camera_settings
    summary['camera_frame_count']=len(camera_times)
    if len(camera_times)>1:
        summary['camera_observed_mean_hz']=float((len(camera_times)-1)/(camera_times[-1]-camera_times[0]))
        summary['camera_frame_interval_min_max_s']=[float(np.diff(camera_times).min()),float(np.diff(camera_times).max())]
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.output/'observations.json').write_text(json.dumps(observations))
    np.savez_compressed(args.output/'states.npz',qpos=np.array(states),time=np.array(rows)[:,0])
    if replay:
        np.savez_compressed(args.output/'sensor_replay.npz',time=np.array([r[0] for r in replay]),
            attitude=np.array([r[1] for r in replay]),policy_feedback=np.array([r[2] for r in replay]),
            frame_index=np.array([r[3] for r in replay]),enabled=np.array([r[4] for r in replay]),
            expected_command=np.array([r[5] for r in replay]),rgb=np.array(replay_rgb),depth=np.array(replay_depth),
            intrinsics=np.array(rig.intrinsics),camera_position=rig.camera_position,camera_rotation=rig.camera_rotation,speed=np.array(args.speed))
    with (args.output/'telemetry.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['t','truth_x','truth_y','truth_yaw','odom_x','odom_y','odom_yaw','vx','vy','wz','eval_progress','estimated_turn','estimated_distance','policy_cmd_vx','policy_cmd_vy','policy_cmd_wz']);w.writerows(rows)
    print(json.dumps(summary,indent=2),flush=True)
    return 0 if status=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
