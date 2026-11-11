"""Render the vision-controlled rollout at native 50 Hz, with perception inset."""
import argparse,json
from pathlib import Path
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    args=p.parse_args();run=args.run
    state=np.load(run/'states.npz');summary=json.loads((run/'summary.json').read_text())
    model=mujoco.MjModel.from_xml_path(str((run/'scene.xml').resolve()));data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height=540,width=960)
    camera=mujoco.MjvCamera();camera.distance=1.8;camera.elevation=-25
    reader=imageio.get_reader(str(run/'perception.mp4'))
    observations=json.loads((run/'observations.json').read_text())
    sensor_times=np.array([observation['t'] for observation in observations])
    writer=imageio.get_writer(str(run/'replay_1x_50fps.mp4'),fps=50,macro_block_size=2)
    sensor_index=-1;sensor_frame=None
    try:
        for i,(q,t) in enumerate(zip(state['qpos'],state['time'])):
            data.qpos[:]=q;mujoco.mj_forward(model,data)
            base=model.body('base_link').id;mat=data.xmat[base].reshape(3,3)
            camera.lookat[:]=data.xpos[base]
            camera.azimuth=100+np.rad2deg(np.arctan2(mat[1,0],mat[0,0]))
            renderer.update_scene(data,camera=camera);frame=renderer.render().copy()
            target=min(len(sensor_times)-1,int(np.searchsorted(sensor_times,t+1e-6,side='right'))-1)
            while sensor_index<target:
                try:sensor_frame=reader.get_next_data();sensor_index+=1
                except (IndexError,StopIteration):break
            if sensor_frame is not None:
                frame[285:525,625:945]=cv2.resize(sensor_frame,(320,240))
            cv2.putText(frame,f'RGB-D AUTONOMOUS NAV + AMP | {t:.2f}s | 1x / 50 fps',(18,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            cv2.putText(frame,'RGB-D + AHRS navigation | fixed factory AMP | truth is referee-only',(18,50),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            writer.append_data(frame)
            if i%500==0:
                print(f'Rendered {i}/{len(state["time"])}',flush=True)
            if i in (100,len(state['time'])//2,len(state['time'])-1):
                Image.fromarray(frame).save(run/f'replay_{i:05d}.png')
    finally:
        writer.close();reader.close();renderer.close()
    print(summary['status'],run/'replay_1x_50fps.mp4')


if __name__=='__main__':main()
