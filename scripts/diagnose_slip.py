"""Replay recorded commands in physics and audit loaded foot contact slip.

Reads the original run without changing its model, commands, or result files.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from sim.robot import Robot


def contact_phases(rows):
    """Distinguish touchdown transients from sustained support on each foot."""
    age = np.zeros(len(rows))
    episodes = []
    for foot in (0, 1):
        ids = np.flatnonzero(rows[:, 1] == foot)
        times = np.unique(rows[ids, 0])
        episode, last, since = [], -10., -10.
        ages = {}
        for t in times:
            if t-last > .003:
                if episode:
                    episodes.append((foot, np.asarray(episode)))
                episode, since = [], t
            at_time = rows[ids[rows[ids, 0] == t]]
            episode.append([t, np.average(at_time[:, 3], weights=at_time[:, 2])])
            ages[t] = t-since
            last = t
        if episode:
            episodes.append((foot, np.asarray(episode)))
        age[ids] = [ages[t] for t in rows[ids, 0]]
    events = []
    for foot, episode in episodes:
        if 93 <= episode[0, 0] < 95:
            events.append({'foot':foot, 'touchdown_s':float(episode[0, 0]),
                           'contact_duration_s':float(episode[-1, 0]-episode[0, 0]+.002),
                           'integrated_contact_slip_m':float(episode[:, 1].sum()*.002)})
    return age, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'output/sim/start_01')
    parser.add_argument('--output', type=Path, default=ROOT/'output/sim/slip_audit')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trace = np.loadtxt(args.run/'telemetry.csv', delimiter=',', skiprows=1)
    recorded = np.load(args.run/'states.npz')
    summary = json.loads((args.run/'summary.json').read_text())
    config = json.loads((ROOT/'scene_config.json').read_text())
    start = next(s for s in config['starts'] if s['id']==summary['start'])
    model = mujoco.MjModel.from_xml_path(str((args.run/'scene.xml').resolve()))
    robot = Robot(model, [*start['position'], np.deg2rad(start['yaw_degrees'])], policy=Path(summary['policy_path']),
                  from_standby=summary.get('locomotion', {}).get('from_measured_standby_attitude', False))
    ground = model.geom('hb_carpet_collision').id
    feet = [model.body(f'{side}_ankle_roll_link').id for side in ('l','r')]
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    force = np.zeros(6)
    audit = mujoco.MjData(model)
    contacts, poses = [], []
    windows = [(10, 15), (70, 100)]
    def capture():
        if not any(a <= robot.data.time <= b for a,b in windows):
            return
        # Recompute forces and Jacobians at the same instantaneous state on
        # a copy. Never disturb the original rollout or its solver warm start.
        mujoco.mj_copyData(audit, model, robot.data)
        mujoco.mj_forward(model, audit)
        data = audit
        for ci, contact in enumerate(data.contact):
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if ground not in (g1,g2) or contact.efc_address < 0:
                continue
            other = g2 if g1==ground else g1
            body = int(model.geom_bodyid[other])
            if body not in feet:
                continue
            mujoco.mj_contactForce(model, data, ci, force)
            normal_force = max(float(force[0]), 0.)
            if normal_force < 1:
                continue
            mujoco.mj_jac(model, data, jp, jr, contact.pos, body)
            velocity = jp @ data.qvel
            normal = contact.frame[:3]
            tangential = velocity-normal*np.dot(velocity,normal)
            contacts.append([data.time, feet.index(body), normal_force,
                             np.linalg.norm(tangential), *tangential, contact.dist,
                             np.linalg.norm(force[1:3])/normal_force, contact.friction[0]])
    error = 0.
    for i, row in enumerate(trace):
        if row[0] > 100.1:
            break
        robot.step(row[5:8], capture)
        error = max(error, float(np.max(np.abs(robot.data.qpos-recorded['qpos'][i]))))
        if error > 1e-7:
            raise RuntimeError(f'Cannot reproduce original run: qpos error={error}')
        if any(a <= robot.data.time <= b for a,b in windows):
            poses.append([robot.data.time, *robot.pose, *robot.data.xpos[feet[0]], *robot.data.xpos[feet[1]], *row[5:8]])
    a = np.asarray(contacts)
    age, events = contact_phases(a)
    results = []
    for lo,hi in [(10,15),(73,78),(79,82),(83,87),(88,92),(92,93),(93,95),(95,97),(97,100)]:
        rows = a[(a[:,0]>=lo)&(a[:,0]<hi)]
        subset = trace[(trace[:,0]>=lo)&(trace[:,0]<hi)]
        load = rows[:,2]
        sustained = a[(a[:,0]>=lo)&(a[:,0]<hi)&(age>=.05)]
        results.append({'start_s':lo,'end_s':hi,
                        'base_displacement_m':float(np.linalg.norm(subset[-1,1:3]-subset[0,1:3])),
                        'load_weighted_slip_mps':float(np.average(rows[:,3],weights=load)),
                        'support_after_50ms_slip_mps':float(np.average(sustained[:,3],weights=sustained[:,2])),
                        'loaded_contact_slip_p95_mps':float(np.percentile(rows[load>=10,3],95)),
                        'load_fraction_above_0_03_mps':float(load[rows[:,3]>.03].sum()/load.sum())})
    report = {'max_reproduction_qpos_error':error,
              'method':'At foot-ground contacts with >1 N normal force, Jacobian contact-point tangential speed, averaged with normal-force weights.',
              'limitation':'Instantaneous force and velocity are evaluated with mj_forward on a copied state. All-contact averages include touchdown impacts; use the separate sustained-support metric. Thresholds are diagnostics, not competition rules. Integrated contact slip is a path-length estimate, not net foot displacement.',
              'last_box_touchdowns':events,
              'windows':results}
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(args.output/'contacts.npz', rows=a, columns=np.array(['t','foot','normal_force','slip_speed','slip_vx','slip_vy','slip_vz','contact_distance','friction_ratio','mu']))
    np.savetxt(args.output/'foot_poses.csv',poses,delimiter=',',header='t,x,y,yaw,left_x,left_y,left_z,right_x,right_y,right_z,cmd_vx,cmd_vy,cmd_wz')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
