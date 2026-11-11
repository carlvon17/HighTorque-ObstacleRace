"""Shared RGB-D navigation on the robot; standalone entry is always shadow.

The bounded supervisor may call run(..., publish=True) only after calibration
and immutable factory-contract checks. Published requests are desired policy
velocities; the sole 50 Hz guard applies navigation.command.AMPCommandAdapter.
"""
import argparse
import importlib
import json
from pathlib import Path
import platform
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from navigation.calibration import calibration_reasons
from navigation.command import verify_device_files
from navigation.runtime import CameraFrame, NavigationRuntime, core_hashes
from navigation.release import verify_release
from navigation.environment import verify_dependencies
from deploy.geometry import quaternion_matrix_xyzw


def frame_head_reasons(config, head, timestamp):
    """Only head-pose errors invalidate the fixed camera projection in shadow."""
    fresh = {name: value[1] for name, value in head.items() if abs(value[0]-timestamp) < .2}
    return [reason for reason in calibration_reasons(config, fresh)
            if reason in ('fixed_head_reference_required', 'head_pose_not_at_calibrated_reference')]


def preflight(config):
    verify_dependencies()
    release=verify_release(ROOT)
    identity = verify_device_files('/home/nvidia/action_library/policy/amp',
        '/home/nvidia/sim2real/install/lib/sim2real_master/sim2real_master_node',
        '/home/nvidia/sim2real/src/sim2real/robot_config/pi_plus-S-12L8A0G2H0W/pd.yaml')
    imports = {}
    for name in ('rospy', 'rosgraph', 'rosnode', 'yaml', 'sensor_msgs.msg', 'geometry_msgs.msg', 'std_msgs.msg', 'livelybot_serial.msg', 'sim2real_msg.msg'):
        importlib.import_module(name)
        imports[name] = True
    return dict(status='code_and_factory_contract_verified', python=platform.python_version(),
                release_id=release['release_id'],
                numpy=np.__version__, opencv=cv2.__version__, imports=imports,
                factory_sha256=identity, navigation_core_sha256=core_hashes(),
                calibration_inhibit_reasons=calibration_reasons(config),
                geometry_basis=config.get('calibration_basis', 'calibration_flags'),
                geometry_assumptions=config.get('assumptions', []),
                motor_output_enabled=False)


def run(args, publish=False):
    import rospy
    from geometry_msgs.msg import TwistStamped
    from deploy.shadow import Sensors, capture_worker, read_exact
    config = json.loads(args.config.read_text())
    initial_reasons = calibration_reasons(config)
    if publish and initial_reasons:
        raise ValueError('Motion inhibited: '+', '.join(initial_reasons))
    identity = preflight(config)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out/'preflight.json').write_text(json.dumps(identity, indent=2))
    (out/'config.json').write_text(json.dumps(config, indent=2))
    rospy.init_node('competition_lane_trial' if publish else 'competition_nav_shadow', disable_signals=True)
    sensors = Sensors()
    publisher = rospy.Publisher('/competition/lane_request', TwistStamped, queue_size=1) if publish else None
    proc = None
    rows = []
    commands = []
    runtime = None
    status = 'starting'
    start = time.monotonic()

    def request(vector):
        if publisher is not None:
            msg = TwistStamped()
            msg.header.frame_id = args.token
            msg.header.stamp = rospy.Time.now()
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z = map(float, vector)
            publisher.publish(msg)

    # Perception produces one update per frame (about 4 Hz on this robot), but
    # the 50 Hz guard drops any request older than 0.2 s. Publishing straight
    # from the frame loop therefore left the guard with nothing to forward. The
    # publisher thread repeats the newest planned velocity and applies the
    # observation lease itself, so a slow frame never looks like a dead link.
    LEASE_S = float(config.get('observation_lease_s', .35))
    publish_state = {'vector': [0., 0., 0.], 'stamp': 0., 'allowed': False}
    publish_stop = threading.Event()

    def publish_loop():
        while not publish_stop.is_set():
            vector = publish_state['vector']
            if not publish_state['allowed'] or time.monotonic()-publish_state['stamp'] > LEASE_S:
                vector = [0., 0., 0.]
            request(vector)
            time.sleep(.02)

    publisher_thread = None
    if publish:
        publisher_thread = threading.Thread(target=publish_loop, daemon=True)
        publisher_thread.start()

    try:
        with (out/'camera.log').open('w') as log:
            proc = subprocess.Popen([str(ROOT/'deploy/realsense_stream')], stdout=subprocess.PIPE, stderr=log, bufsize=0)
            header = bytearray()
            while not header.endswith(b'\n'):
                header.extend(read_exact(proc.stdout, 1, 10.))
                if len(header) > 4096:
                    raise ValueError('Invalid camera header')
            camera = json.loads(header)
            (out/'camera_intrinsics.json').write_text(json.dumps(camera, indent=2))
            w, h = camera['width'], camera['height']
            k = np.array([[camera['fx'], 0, camera['cx']], [0, camera['fy'], camera['cy']], [0, 0, 1.]])
            coefficients = np.asarray(camera['coefficients'])
            if np.any(abs(coefficients) > 1e-9) and camera['distortion_model'] != 4:
                raise ValueError('Unsupported nonzero camera distortion')
            # Perception runs on a downscaled image. The pixel-centre convention is
            # preserved so every metric quantity (rays, ground, lanes) is unchanged.
            scale = float(config.get('perception_scale', .5))
            if not .25 <= scale <= .5:
                raise ValueError('perception_scale must be within [0.25, 0.5]')
            sw, sh = int(round(w*scale)), int(round(h*scale))
            intrinsics = [k[0, 0]*scale, k[1, 1]*scale, (k[0, 2]+.5)*scale-.5, (k[1, 2]+.5)*scale-.5]
            # Rectify straight into the perception resolution. Remapping the full
            # 640x480 pair and resizing afterwards spent about 0.25 s per frame on
            # the Jetson for pixels that were then discarded; the decimated image
            # is all that is ever used, so the maps target it directly.
            destination = np.array([[intrinsics[0], 0, intrinsics[2]],
                                    [0, intrinsics[1], intrinsics[3]], [0, 0, 1.]])
            maps = cv2.initUndistortRectifyMap(k, coefficients, None, destination, (sw, sh), cv2.CV_32FC1)
            runtime = NavigationRuntime(sw, sh, intrinsics, config['camera_position_body_m'],
                config['camera_rotation_optical_to_body'], args.speed,
                float(config.get('observation_lease_s', .35)))
            body_from_imu = np.asarray(config['body_from_imu'])
            latest = queue.Queue(maxsize=1)
            # Frame arrival is recorded by the reader, not inferred from how long
            # our own processing took, so a slow frame cannot fake a dead camera.
            camera_status = {'frames': 0, 'last_received': time.monotonic(),
                             'last_frame_time': None, 'error': None}
            threading.Thread(target=capture_worker, args=(proc.stdout, w, h, latest, camera_status), daemon=True).start()
            last_report = -1.
            control_start = time.monotonic()
            while time.monotonic()-control_start < args.duration and not rospy.is_shutdown():
                tick = time.monotonic()
                current_reasons = list(initial_reasons)
                raw = None
                try:
                    raw = latest.get_nowait()
                except queue.Empty:
                    pass
                if isinstance(raw, Exception):
                    raise raw
                try:
                    imu, joints, head = sensors.snapshot(tick)
                    rotation = quaternion_matrix_xyzw(imu[1])@body_from_imu.T
                    runtime.set_attitude(imu[0], rotation)
                    fresh_head = {name: value[1] for name, value in head.items() if tick-value[0] < .2}
                    current_reasons += calibration_reasons(config, fresh_head)
                    if sum(tick-value[0] < .2 for value in joints.values()) != 20:
                        current_reasons.append('joint_feedback_incomplete')
                except ValueError:
                    current_reasons.append('no_fresh_imu')
                processing_ms = None
                if raw is not None:
                    if tick-raw.time > .2:
                        projection_reasons = ['stale_capture']
                    else:
                        imu, _, capture_head = sensors.snapshot(raw.time)
                        projection_reasons = frame_head_reasons(config, capture_head, raw.time)
                    current_reasons += projection_reasons
                    if not projection_reasons:
                        before = time.monotonic()
                        frame_rotation = quaternion_matrix_xyzw(imu[1])@body_from_imu.T
                        rgb = cv2.remap(raw.rgb, *maps, cv2.INTER_LINEAR)
                        depth = cv2.remap(raw.depth, *maps, cv2.INTER_NEAREST)
                        remap_ms = (time.monotonic()-before)*1000
                        depth[(depth < .15) | (depth > 4.)] = np.nan
                        observe_start = time.monotonic()
                        observation = runtime.observe(CameraFrame(raw.time, rgb, depth), frame_rotation, imu[0])
                        processing_ms = (time.monotonic()-observe_start)*1000
                        row = dict(t=tick-start, camera_host_age_s=tick-raw.time, imu_host_delta_s=imu[0]-raw.time,
                            remap_ms=remap_ms, processing_ms=processing_ms, lane=None if observation.lane is None else observation.lane.tolist(),
                            boxes=observation.obstacles.tolist(), **runtime.diagnostic())
                        rows.append(row)
                        with (out/'telemetry.jsonl').open('a') as f:
                            f.write(json.dumps(row)+'\n')
                        if tick-last_report > 1:
                            cv2.imwrite(str(out/'latest_rgb.jpg'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                            cv2.imwrite(str(out/'latest_overlay.jpg'), cv2.cvtColor(observation.overlay, cv2.COLOR_RGB2BGR))
                            print(json.dumps(row), flush=True)
                            last_report = tick
                now = time.monotonic()
                if now-camera_status['last_received'] > 2:
                    raise TimeoutError('No camera frames for 2 seconds')
                # Re-read the control attitude right before planning so the IMU
                # freshness gate measures the sensor link, not how long this loop
                # just spent on perception.
                try:
                    control_imu, _, _ = sensors.snapshot(now)
                    runtime.set_attitude(control_imu[0],
                        quaternion_matrix_xyzw(control_imu[1])@body_from_imu.T)
                except ValueError:
                    pass
                # Shadow calculates candidates with declared transforms, while
                # retaining every calibration inhibit in the diagnostic record.
                enabled = not current_reasons if publish else True
                candidate = runtime.command(now, enabled=enabled)
                desired = runtime.desired.copy()
                computation_ms = (time.monotonic()-tick)*1000
                # Refresh age after planning, which is the expensive part. A
                # late computation may not refresh a stale request's lease.
                if publish and (runtime.last_frame_time is None or time.monotonic()-runtime.last_frame_time>LEASE_S):
                    enabled=False
                    current_reasons.append('stale_after_planning')
                publish_state['vector'] = [float(v) for v in desired]
                publish_state['allowed'] = bool(enabled)
                publish_state['stamp'] = time.monotonic()
                if not publish:
                    request(desired if enabled else [0., 0., 0.])
                commands.append(dict(t=now-start, candidate_raw=candidate.tolist(),
                    requested_policy_velocity=desired.tolist(),
                    published=publish, computation_ms=computation_ms, has_camera_frame=processing_ms is not None,
                    inhibit_reasons=sorted(set(current_reasons)), **runtime.diagnostic()))
                if runtime.navigator.finished:
                    status = 'visual_finish_detected'
                    break
                time.sleep(max(0., .02-(time.monotonic()-tick)))
            else:
                status = 'navigation_time_limit' if publish else 'shadow_complete'
    except (Exception, KeyboardInterrupt) as exc:
        status = 'error: '+str(exc)
    finally:
        publish_stop.set()
        if publisher_thread is not None:
            publisher_thread.join(timeout=1.)
        for _ in range(5):
            request([0., 0., 0.])
            if publish:time.sleep(.02)
        if proc is not None:
            proc.terminate()
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        rospy.signal_shutdown('shared navigation stopped')
        if runtime is not None:
            runtime.floor_map.save(out/'floor_map.npz')
        (out/'commands.json').write_text(json.dumps(commands))
        summary = dict(status=status, frames=len(rows), motor_publishers=0, perception_scale=config.get('perception_scale', .5), velocity_request_publishers=int(publish),
            navigation_core_sha256=core_hashes(), calibration_inhibit_reasons=initial_reasons,
            last_navigation=None if runtime is None else runtime.diagnostic(),
            remap_ms_p50=float(np.median([r['remap_ms'] for r in rows])) if rows else None,
            remap_ms_p95=float(np.percentile([r['remap_ms'] for r in rows], 95)) if rows else None,
            processing_ms_p50=float(np.median([r['processing_ms'] for r in rows])) if rows else None,
            processing_ms_p95=float(np.percentile([r['processing_ms'] for r in rows], 95)) if rows else None,
            camera_cycle_ms_p95=float(np.percentile([r['computation_ms'] for r in commands if r['has_camera_frame']],95)) if any(r['has_camera_frame'] for r in commands) else None,
            control_cycle_ms_p95=float(np.percentile([r['computation_ms'] for r in commands],95)) if commands else None,
            imu_camera_association='host receipt timestamps; exposure latency not calibrated')
        (out/('vision_summary.json' if publish else 'summary.json')).write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    return 2 if status.startswith('error:') or status == 'navigation_time_limit' else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'deploy/pro_config.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--speed', type=float, default=.22)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if not 0 < args.duration <= 300 or not 0 < args.speed <= .4:
        parser.error('Duration/speed outside the bounded navigation contract')
    if args.preflight:
        result = preflight(json.loads(args.config.read_text()))
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'preflight.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        return 0
    return run(args, publish=False)


if __name__ == '__main__':
    raise SystemExit(main())
