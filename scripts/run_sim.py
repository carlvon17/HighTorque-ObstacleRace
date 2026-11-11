"""Run the physical Pi Plus with oracle navigation; headless by default."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from sim.field import DEFAULT_BEND, FIELD_DIRS
from sim.robot import Robot, build_model
from sim.navigation import Track, Navigator
from sim.scoring import Referee


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bend-profile", choices=FIELD_DIRS, default=DEFAULT_BEND)
    parser.add_argument("--start", choices=["01", "02", "03"], default="01")
    parser.add_argument("--duration", type=float, default=240)
    parser.add_argument("--speed", type=float, default=.25)
    parser.add_argument("--no-obstacles", action="store_true")
    parser.add_argument("--planner", choices=["full-body", "feet-only"], default="full-body",
                        help="full-body permits single-foot overhang; feet-only is the collision-prone comparison")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--policy", type=Path, help="Explicit legacy .pt override; default is the audited Pro device policy")
    args = parser.parse_args()
    if not 0 < args.speed <= .5 or not 0 < args.duration <= 300:
        parser.error("Require 0 < speed <= 0.5 m/s and 0 < duration <= 300 s")
    if args.output is None:
        args.output = ROOT / "output/sim" / ("truth" if args.bend_profile == "smooth" else f"truth_{args.bend_profile}")
    args.output.mkdir(parents=True, exist_ok=True)
    track = Track(args.bend_profile)
    track.plan(no_obstacles=args.no_obstacles, full_body=args.planner == "full-body")
    start = next(s for s in track.config["starts"] if s["id"] == args.start)
    pose = [*start["position"], np.deg2rad(start["yaw_degrees"])]
    legacy = args.policy is not None and args.policy.suffix == '.pt'
    model, xml = build_model(no_obstacles=args.no_obstacles, timestep=.002 if legacy else .001, legacy=legacy, bend_profile=args.bend_profile)
    (args.output / "scene.xml").write_text(xml)
    robot = Robot(model, pose, policy=args.policy, from_standby=not legacy)
    nav = Navigator(track, pose, speed=args.speed)
    referee = Referee(robot, track)
    viewer, renderer, camera, video = None, None, None, None
    if args.viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, robot.data)
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -25
    if args.video:
        import imageio.v2 as imageio
        renderer = mujoco.Renderer(model, height=540, width=960)
        camera = mujoco.MjvCamera()
        camera.distance, camera.elevation, camera.azimuth = 2.0, -25, 135
        video = imageio.get_writer(str(args.output / "replay.mp4"), fps=10, macro_block_size=2)
    rows = []
    states = []
    status = "duration_limit"
    next_log, next_frame = 10., 0.
    finish_time = None
    finish_pose = None
    wall_start = time.monotonic()
    try:
        while robot.data.time < min(args.duration, 300):
            tick = time.monotonic()
            command = np.zeros(3) if robot.data.time < 2 else nav.update(robot.pose)
            robot.step(command, referee.contacts)
            referee.boundary()
            rows.append([robot.data.time, *robot.pose, robot.data.qpos[2], *command,
                         *robot.data.sensor("linear-velocity").data[:2], nav.progress,
                         *(robot.device.command if robot.device else command)])
            states.append(robot.data.qpos.copy())
            if viewer:
                if not viewer.is_running():
                    status = "viewer_closed"
                    break
                viewer.cam.lookat[:] = robot.data.xpos[robot.base]
                viewer.sync()
                time.sleep(max(0., robot.dt - (time.monotonic()-tick)))
            if renderer and robot.data.time >= next_frame:
                import cv2
                camera.lookat[:] = robot.data.xpos[robot.base]
                renderer.update_scene(robot.data, camera=camera)
                frame = renderer.render().copy()
                cv2.putText(frame, f"TRUTH NAV + AMP | t={robot.data.time:.1f}s | {nav.progress:.1f}/{track.length:.1f}m",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                video.append_data(frame)
                next_frame += .10
            if referee.reason:
                status = referee.reason
                break
            if nav.finished and finish_time is None:
                finish_time = float(robot.data.time)
                finish_pose = robot.pose.copy()
            if finish_time is not None and robot.data.time-finish_time >= 3:
                status = "completed"
                break
            if robot.data.time >= next_log:
                print(f"t={robot.data.time:.1f}s progress={nav.progress:.2f}m pose={robot.pose.round(3)}", flush=True)
                next_log += 10
    finally:
        if viewer:
            viewer.close()
        if video:
            video.close()
        if renderer:
            renderer.close()
    summary = {
        "bend_profile": args.bend_profile, "status": status, "navigation": "ground_truth_lane_lattice", "locomotion": robot.locomotion_metadata(),
        "start": args.start, "obstacles": not args.no_obstacles,
        "full_body_planning": track.full_body,
        "finish_time_s": finish_time,
        "post_finish_drift_m": float(np.linalg.norm(robot.pose[:2]-finish_pose[:2])) if finish_pose is not None else None,
        "sim_time_s": robot.data.time, "wall_time_s": time.monotonic()-wall_start,
        "progress_m": nav.progress, "lap_length_m": track.length,
        "max_projected_foot_width_m": referee.max_foot_width, "min_base_height_m": referee.min_base_height,
        "foot_overhang_time_s": referee.partial_foot_overhang_frames*robot.dt,
        "event_time_s": referee.event_time, "event_detail": referee.event_detail,
        "planning_envelope": {"half_length_m": .13, "half_width_m": .21, "lateral_margin_m": .015,
                              "longitudinal_margin_m": .005, "max_center_offset_m": .23} if track.full_body else {"radius_m": track.radius, "margin_m": track.margin},
        "policy_path": str(robot.policy_path),
        "limitations": ["Navigation uses simulator pose and known map; no visual completion claim.",
                        "Finish uses one unwrapped centreline lap; no official finish-line geometry is specified.",
                        "Boundary test uses XY foot polygon overlap with the white-line centre boundaries at 50 Hz.",
                        f"Ground contact and obstacle contact checked at {1/model.opt.timestep:g} Hz; fall posture uses engineering thresholds.",
                        "Planning envelope is approximate; MuJoCo original robot collisions determine contact failures.",
                        "Full-body route permits one foot to partially cross a white line; stricter all-feet-inside criterion is NOT claimed."]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.output / "telemetry.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "x_m", "y_m", "yaw_rad", "base_z_m", "cmd_vx", "cmd_vy", "cmd_wz", "actual_vx", "actual_vy", "progress_m", "policy_cmd_vx", "policy_cmd_vy", "policy_cmd_wz"])
        writer.writerows(rows)
    np.savetxt(args.output / "planned_path.csv", track.path, delimiter=",", header="x_m,y_m")
    np.savez(args.output / "final_state.npz", qpos=robot.data.qpos, qvel=robot.data.qvel)
    np.savez_compressed(args.output / "states.npz", qpos=np.asarray(states), time=np.asarray(rows)[:, 0])
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
