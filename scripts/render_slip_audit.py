"""Render the last obstacle from recorded states at their native 50 Hz."""
from pathlib import Path
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = ROOT / 'output/sim/start_01'
    out = ROOT / 'output/sim/slip_audit'
    out.mkdir(parents=True, exist_ok=True)
    state = np.load(run / 'states.npz')
    model = mujoco.MjModel.from_xml_path(str(run / 'scene.xml'))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=540, width=960)
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 1.6, -20, -80
    camera.lookat[:] = [.65, 1.35, .28]
    indices = np.flatnonzero((state['time'] >= 88) & (state['time'] < 98))
    assert np.allclose(np.diff(state['time'][indices]), .02)
    writer = imageio.get_writer(str(out / 'last_box_1x_50fps.mp4'), fps=50,
                               macro_block_size=2)
    tiles = []
    try:
        for k, i in enumerate(indices):
            data.qpos[:] = state['qpos'][i]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = renderer.render().copy()
            cv2.putText(frame, f'LAST BOX | t={state["time"][i]:.2f}s | 1x / 50 fps',
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 1)
            cv2.putText(frame, 'Original physics | low foot lift / brief touchdown slip',
                        (20, 55), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
            writer.append_data(frame)
            if any(abs(state['time'][i] - t) < .005 for t in (93,93.2,93.4,93.6,93.8,94)):
                tiles.append(Image.fromarray(frame).resize((480,270)))
            if k % 100 == 0:
                print(f'Rendered {k}/{len(indices)}', flush=True)
    finally:
        writer.close()
        renderer.close()
    sheet = Image.new('RGB', (1440,540))
    for k, tile in enumerate(tiles):
        sheet.paste(tile, ((k%3)*480, (k//3)*270))
    sheet.save(out / 'last_box_frames.jpg')
    print(out / 'last_box_1x_50fps.mp4')


if __name__ == '__main__':
    main()
