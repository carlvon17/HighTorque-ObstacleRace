"""Render recorded physical states, plus the actual body-mounted camera view."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from sim.navigation import Track


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()
    if args.speed <= 0 or args.fps <= 0:
        parser.error("Speed and fps must be positive")
    if args.speed / args.fps > .02001:
        print('WARNING: playback samples below recorded 50 Hz; gait can alias and appear to slide.', flush=True)
    run = args.run
    state = np.load(run / "states.npz")
    model = mujoco.MjModel.from_xml_path(str((run / "scene.xml").resolve()))
    data = mujoco.MjData(model)
    summary = json.loads((run / "summary.json").read_text())
    track = Track()
    trace = np.loadtxt(run / "telemetry.csv", delimiter=",", skiprows=1)
    path = np.loadtxt(run / "planned_path.csv", delimiter=",")
    # Plan/actual trajectory plot uses the same metric carpet projection.
    canvas = Image.open(ROOT / "output/athletics/mujoco/assets/carpet.png").resize((1040, 640))
    draw = ImageDraw.Draw(canvas)
    c = track.config
    low = np.array(c["carpet_center"])-np.array(c["carpet_dimensions"][:2])/2
    scale = np.array([1040, 640])/np.array(c["carpet_dimensions"][:2])
    def pixel(p):
        q = (np.asarray(p)-low)*scale
        return (float(q[0]), float(640-q[1]))
    for o in c['obstacles']:
        x, y = o['position']
        draw.rectangle([pixel([x-.1,y+.1]),pixel([x+.1,y-.1])], fill=(219, 196, 143))
    draw.line([pixel(p) for p in path], fill=(20, 210, 255), width=3)
    draw.line([pixel(p) for p in trace[:, 1:3]], fill=(255, 220, 40), width=2)
    draw.text((16, 16), "CYAN: planned | YELLOW: physical base trajectory | TRUTH NAV", fill="white")
    canvas.save(run / "trajectory.png")
    renderer = mujoco.Renderer(model, height=540, width=960)
    onboard = mujoco.Renderer(model, height=240, width=320)
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 2.0, -30, 110
    sample_times = np.arange(state['time'][0], state['time'][-1], args.speed/args.fps)
    indices = np.searchsorted(state['time'], sample_times)
    writer = imageio.get_writer(str(run / "replay.mp4"), fps=args.fps, macro_block_size=2)
    snapshots = set()
    try:
        for count, i in enumerate(indices):
            data.qpos[:] = state['qpos'][i]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.body('base_link').xpos
            renderer.update_scene(data, camera=camera)
            frame = renderer.render().copy()
            onboard.update_scene(data, camera='nav_camera')
            rgb = onboard.render().copy()
            frame[285:525, 625:945] = rgb
            cv2.putText(frame, 'SIM CAMERA (not used by this baseline)', (620, 275), cv2.FONT_HERSHEY_SIMPLEX, .38, (255,255,255), 1)
            cv2.putText(frame, f'TRUTH NAV + AMP | t={state["time"][i]:.1f}s | playback {args.speed:g}x', (20, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 1)
            cv2.putText(frame, 'Original joint dynamics / single-foot overhang allowed', (20, 52), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
            writer.append_data(frame)
            key = 'start' if state['time'][i]<3 else ('obstacle' if 1.3 < data.qpos[0] < 1.7 and data.qpos[1]>1.1 else None)
            if key and key not in snapshots:
                Image.fromarray(frame).save(run / f'{key}.png')
                Image.fromarray(rgb).save(run / f'{key}_camera.png')
                onboard.enable_depth_rendering()
                onboard.update_scene(data, camera='nav_camera')
                np.save(run / f'{key}_depth.npy', onboard.render())
                onboard.disable_depth_rendering()
                snapshots.add(key)
            if count % 100 == 0:
                print(f'Rendered {count}/{len(indices)} frames', flush=True)
        Image.fromarray(frame).save(run / 'finish.png')
    finally:
        writer.close()
        renderer.close()
        onboard.close()
    print(f'{summary["status"]}: {run / "replay.mp4"}')


if __name__ == '__main__':
    main()
