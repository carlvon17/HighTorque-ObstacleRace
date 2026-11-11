"""Visual navigation over rendered frames with an ideal velocity-tracking body.

The AMP gait rollout is machine-sensitive; on hosts where the frozen policy
does not sustain walking this harness still exercises the complete navigation
stack: the real D435i rendering, RGB-D perception, odometry, the side-step
avoidance state machine and every runtime gate. Only the plant is idealized --
the body tracks the gated navigator command exactly, so the run validates the
strategy, not the locomotion dynamics. It makes no physics or timing claim.
"""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from deploy.geometry import quaternion_matrix_xyzw
import navigation.avoid_controller as ac
from navigation.environment import verify_dependencies
from navigation.runtime import CameraFrame, NavigationRuntime
from navigation.visual import rotation2
from scripts.run_visual import visual_world
from sim.navigation import Track
from sim.robot import Robot
from sim.sensors import SensorRig


def one_sided_boxes(model, track):
    """Snap every box to the majority side of the lane, edge on the same line.

    Both the rendered bodies and the audit geometry move together, so the
    clearance numbers stay faithful to what the camera actually sees.
    """
    obstacles = track.config['obstacles']
    offsets = []
    for obstacle in obstacles:
        center = track.center[track.tree.query(np.asarray(obstacle['position'][:2]))[1]]
        normal = track.normal[track.tree.query(np.asarray(obstacle['position'][:2]))[1]]
        offsets.append(float((np.asarray(obstacle['position'][:2])-center)@normal))
    side = float(np.sign(sum(np.sign(offsets))))
    depth = float(np.mean([abs(offset) for offset in offsets if np.sign(offset) == side]))
    for obstacle, offset in zip(obstacles, offsets):
        if np.sign(offset) == side:
            continue
        index = track.tree.query(np.asarray(obstacle['position'][:2]))[1]
        position = track.center[index]+track.normal[index]*side*depth
        old = np.asarray(obstacle['position'][:2])
        obstacle['position'][:2] = [float(position[0]), float(position[1])]
        for number in range(1, 4):
            body = model.body(f'hb_obstacle_{number:02d}')
            if np.linalg.norm(model.body_pos[body.id, :2]-old) < .01:
                model.body_pos[body.id, :2] = position
    track.obstacles = np.array([o['position'][:2] for o in obstacles])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start', default='01', choices=['01', '02', '03'])
    p.add_argument('--duration', type=float, default=240.)
    p.add_argument('--speed', type=float, default=.22)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--no-noise', action='store_true')
    p.add_argument('--obstacle-shift', type=float, default=0)
    p.add_argument('--one-sided-boxes', action='store_true',
                   help='Move every box onto the majority side of the lane '
                        '(edge on the same line), matching the premise that '
                        'boxes always stand on one side')
    p.add_argument('--spawn', type=float, nargs=3, metavar=('X', 'Y', 'YAW'))
    p.add_argument('--debug-avoid', action='store_true',
                   help='Print avoidance trigger, far-line choice and cruise handoffs')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    track = Track()
    start = next(s for s in track.config['starts'] if s['id'] == args.start)
    camera_profile = json.loads((ROOT/'sim/d435i_camera.json').read_text())
    model, xml = visual_world(args.output, start, False, args.obstacle_shift, camera_profile)
    if args.one_sided_boxes:
        one_sided_boxes(model, track)
    (args.output/'scene.xml').write_text(xml)
    spawn = args.spawn or [*start['position'], np.deg2rad(start['yaw_degrees'])]
    robot = Robot(model, spawn)  # rendering body only; never stepped
    rig = SensorRig(model, args.seed, not args.no_noise, camera_profile)

    odom = NavigationRuntime(rig.width, rig.height, rig.intrinsics,
                             rig.camera_position, rig.camera_rotation, args.speed)
    if args.debug_avoid:
        avoid = odom.navigator.avoid
        begin, select, guide, cruise = (avoid.begin, avoid._select_far,
                                        avoid._guide_blocking, avoid._start_cruise)

        def d_begin(time, pose, corridor, box, creep=False):
            print(f'  BEGIN t={time:6.2f} box={np.round(box, 2).tolist()}'
                  f'{" creep" if creep else ""}', flush=True)
            begin(time, pose, corridor, box, creep=creep)

        def d_select(lines):
            best = select(lines)
            print(f'    far <- {"None" if best is None else np.round(best[0], 2).tolist()} '
                  f'of {len(lines)} lines', flush=True)
            return best

        def d_guide(perception, pose):
            box = guide(perception, pose)
            if box is not None:
                point, tangent = avoid.guide
                normal = np.array([-tangent[1], tangent[0]])
                world = np.array(box[:2])@ac.rotation(pose[2]).T+pose[:2]
                along = (world-point)@tangent
                lateral = (world-point)@normal
                print(f'    GUIDE BLOCK box={np.round(box, 2).tolist()} '
                      f'along={along:.2f} lateral={lateral:.2f} '
                      f'pose={np.round(pose, 2).tolist()}', flush=True)
            return box

        def d_cruise(time, far_line):
            print(f'    CRUISE from t={time:6.2f} line pt={np.round(far_line[0], 2).tolist()}', flush=True)
            cruise(time, far_line)

        avoid.begin, avoid._select_far, avoid._guide_blocking, avoid._start_cruise = \
            d_begin, d_select, d_guide, d_cruise
    from navigation.command import verify_device_files
    verify_device_files(ROOT/'vendor/pi_plus_pro', pd_path=ROOT/'vendor/pi_plus_pro/pd.yaml')

    base = robot.base
    height = float(robot.data.qpos[2])

    def place(pose):
        robot.data.time = float(pose[0])
        robot.data.qpos[:3] = [pose[1], pose[2], height]
        yaw = pose[3]/2
        robot.data.qpos[3:7] = [np.cos(yaw), 0., 0., np.sin(yaw)]
        robot.data.qvel[:] = 0.
        import mujoco
        mujoco.mj_forward(model, robot.data)

    rows = []
    pose = np.array([0., *spawn[:2], spawn[2]])  # t, x, y, yaw
    world = spawn[:2].copy()
    yaw = float(spawn[2])
    next_camera = 0.
    last_s = track.s[track.tree.query(world)[1]]
    progress = 0.
    status = 'duration_limit'
    blocked_since = None
    min_box_clearance = float('inf')
    max_lane_offset = 0.
    finish_time = None
    wall = time.monotonic()
    while pose[0] < args.duration:
        t = pose[0]
        place(pose)
        measured = rig.proprioception(robot.data)
        q = measured.orientation_wxyz
        odom.set_attitude(t, quaternion_matrix_xyzw(np.r_[q[1:], q[0]]))
        if t >= next_camera-1e-7:
            frame = rig.camera(robot.data)
            odom.observe(CameraFrame(t, frame.rgb, frame.depth))
            next_camera += 1/rig.fps
        odom.command(t, enabled=t >= 2)
        state = odom.diagnostic()
        desired = np.asarray(state['desired_velocity'], dtype=float)
        world = world+desired[:2]@rotation2(yaw)*.02
        yaw = float((yaw+desired[2]*.02+np.pi) % (2*np.pi)-np.pi)
        pose = np.array([t+.02, *world, yaw])

        current = track.s[track.tree.query(world)[1]]
        progress += (current-last_s+track.length/2) % track.length-track.length/2
        last_s = current
        clearance = track.obstacle_distance(world[None, :])[0]
        min_box_clearance = min(min_box_clearance, float(clearance))
        lane_offset = float(np.linalg.norm(track.center[track.tree.query(world)[1]]-world))
        max_lane_offset = max(max_lane_offset, lane_offset)
        rows.append([t, *world, yaw, *odom.pose, progress, state['mode'], state['reason'],
                     state['avoid_phase'], *desired])
        if state['reason'] == 'finished':
            status = 'finished'; finish_time = t; break
        if state['mode'] == 'blocked':
            blocked_since = blocked_since or t
            if t-blocked_since > 3.: status = 'navigation_blocked'; break
        else:
            blocked_since = None

    summary = dict(status=status, finish_time_s=finish_time, sim_time_s=rows[-1][0],
                   wall_time_s=time.monotonic()-wall, progress_m=progress,
                   track_length_m=track.length, mode=state['mode'], reason=state['reason'],
                   avoid_phase=state['avoid_phase'], min_bodycenter_to_box_m=min_box_clearance,
                   max_bodycenter_to_lane_centerline_m=max_lane_offset,
                   body_half_extents_m=[.13, .21], lane_half_width_m=.25,
                   one_sided_boxes=bool(args.one_sided_boxes),
                   plant='ideal velocity tracking at 50 Hz; no AMP gait, no contacts',
                   python=platform.python_version(), **verify_dependencies())
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    header = ('t,x,y,yaw,est_x,est_y,est_yaw,progress,mode,reason,avoid_phase,'
              'desired_vx,desired_vy,desired_wz')
    np.savetxt(args.output/'telemetry.csv', np.asarray(rows, dtype=object), delimiter=',',
               fmt='%s', header=header, comments='')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
