"""On-robot terminal commands for saved head pose and one full navigation lap."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONTROL_LOCK = Path('/tmp/competition_navigation_control.lock')
CLI_LOCK = Path('/tmp/competition_navigation_cli.lock')


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path, value):
    temporary = path.with_suffix('.new')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def owned_process(record):
    """Match the unique output directory as well as the executable before signalling."""
    try:
        command = Path('/proc', str(int(record['pid'])), 'cmdline').read_bytes().split(b'\0')
        arguments = [value.decode() for value in command if value]
        return (record['script'] in arguments and '--output' in arguments
                and arguments[arguments.index('--output')+1] == record['output'])
    except (OSError, ValueError, KeyError, IndexError, UnicodeError):
        return False


def task_status(record):
    if not record:
        return dict(status='idle')
    output = Path(record['output'])
    report_path = output if record['kind'] == 'head' else output/'summary.json'
    report = read_json(report_path)
    active = owned_process(record)
    status = 'running' if active else report.get('status', 'exited_without_report')
    return dict(record, status=status, active=active, report=report)


def stop_task(record):
    if not owned_process(record):
        return task_status(record)
    os.kill(record['pid'], signal.SIGINT if record['kind'] == 'head' else signal.SIGTERM)
    deadline = time.monotonic()+15
    while owned_process(record) and time.monotonic() < deadline:
        time.sleep(.1)
    result = task_status(record)
    if result['active']:
        result['status'] = 'stop_requested_waiting_for_cleanup'
    return result


def validate_start(config, reference):
    from navigation.calibration import calibration_reasons
    from deploy.shared_navigation import preflight
    report = preflight(config)
    reasons = calibration_reasons(config, reference)
    if reasons:
        raise ValueError('相机/IMU 配置尚未就绪，请检查 --config 指定的配置。'+', '.join(reasons))
    import rospy
    from sensor_msgs.msg import JointState
    from sim2real_msg.msg import StateInfo
    from std_msgs.msg import Int32
    rospy.init_node('competition_navigation_cli', anonymous=True, disable_signals=True)
    state = rospy.wait_for_message('/sim2real_state_info', StateInfo, timeout=3)
    fsm = rospy.wait_for_message('/fsm_state', Int32, timeout=3)
    if fsm.data != 5 or state.state != 'running' or state.policy_name != 'amp':
        raise ValueError('原厂控制器需要处于 Default / AMP running，当前为 '+state.state)
    joints = rospy.wait_for_message('/sim2real_master_node/rbt_state', JointState, timeout=3)
    reasons = calibration_reasons(config, dict(zip(joints.name, joints.position)))
    if reasons:
        raise ValueError('头部未处于标定位置，请先执行 nav head。'+', '.join(reasons))
    return report


def restore_head(root, folder, state_path, reference):
    output = folder/'head.json'
    script = str(root/'deploy/head_step.py')
    command = [sys.executable, script, '--restore-saved-pose', '--allow-running',
               '--execute', '--output', str(output)]
    with CONTROL_LOCK.open('a+') as control:
        fcntl.flock(control.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (folder/'head.log').open('ab') as log:
            process = subprocess.Popen(command, cwd=str(root), stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=log, start_new_session=True,
                                       pass_fds=(control.fileno(),))
        record = dict(kind='head', pid=process.pid, script=script, output=str(output),
                      log=str(folder/'head.log'))
        try:
            write_json(state_path, record)
            print('正在恢复俯仰和偏航，等待两个关节反馈到位……', flush=True)
            returncode = process.wait()
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=5)
        report = read_json(output)
        if returncode or report.get('status') != 'saved_pose_restored':
            raise RuntimeError('头部恢复失败，请查看 '+str(folder/'head.log')+' '+str(report.get('error', '')))
        for key, joint, tolerance in (('final_pitch', 'head_pitch_joint', .025),
                                      ('final_yaw', 'head_yaw_joint', .04)):
            value = report.get(key, float('nan'))
            if not math.isfinite(value) or abs(value-reference[joint]) > tolerance:
                raise RuntimeError('头部反馈未到位 '+joint)
        return dict(record, status='head_ready', report=report)


def start_lap(root, folder, state_path, config, speed):
    output = folder/'navigation'
    command = [sys.executable, str(root/'deploy/launch_line_trial.py'),
               '--shared-navigation', '--execute', '--duration', '300',
               '--speed', str(speed), '--config', str(config), '--output', str(output)]
    result = subprocess.run(command, cwd=str(root), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    launch = read_json(output/'launch.json')
    if launch.get('status') != 'launched':
        raise RuntimeError('未取得整圈任务启动记录 '+result.stdout)
    record = dict(kind='lap', pid=launch['pid'], script=str(root/'deploy/line_trial.py'),
                  output=str(output), log=str(output/'mission.log'), maximum_duration_s=300)
    try:
        write_json(state_path, record)
    except OSError:
        stop_task(record)
        raise
    return dict(record, status='lap_started', detail='整圈任务已在后台启动，识别终点后结束，最长 300 秒')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('head', 'start', 'status', 'stop', 'check'))
    parser.add_argument('--config', type=Path,
                        default=Path(os.environ.get('NAV_CALIBRATION_FILE', str(ROOT/'deploy/pro_model_config.json'))))
    parser.add_argument('--speed', type=float, default=.22)
    args = parser.parse_args(argv)
    if not math.isfinite(args.speed) or not 0 < args.speed <= .4:
        parser.error('speed must be within (0, 0.4] m/s')
    base = ROOT/'runs/terminal'
    base.mkdir(parents=True, exist_ok=True)
    state_path = base/'latest.json'
    try:
        if args.action in ('status', 'stop'):
            record = read_json(state_path)
            result = stop_task(record) if args.action == 'stop' else task_status(record)
        else:
            from navigation.release import verify_release
            verify_release(ROOT)
            reference = read_json(ROOT/'deploy/task_head_pose.json')['measured_joint_reference_rad']
            if args.action == 'check':
                from deploy.shared_navigation import preflight
                result = preflight(read_json(args.config))
            else:
                with CLI_LOCK.open('a+') as operation:
                    fcntl.flock(operation.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    current = read_json(state_path)
                    if owned_process(current):
                        raise RuntimeError('已有终端任务运行，请执行 nav status 查看或 nav stop 停止')
                    if args.action == 'start':
                        validate_start(read_json(args.config), reference)
                    name = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ_')+uuid.uuid4().hex[:8]
                    folder = base/name
                    folder.mkdir()
                    if args.action == 'head':
                        result = restore_head(ROOT, folder, state_path, reference)
                    else:
                        result = start_lap(ROOT, folder, state_path, args.config.resolve(), args.speed)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BlockingIOError:
        print('已有头部或导航任务占用控制，请等待完成或执行 nav stop。', file=sys.stderr)
        return 2
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
