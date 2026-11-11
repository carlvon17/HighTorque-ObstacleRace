"""Compare an optimized BotLab package with the saved pre-fix package."""
import argparse
import json
import tempfile
from pathlib import Path
from zipfile import ZipFile

import mujoco
import numpy as np
from botlab import open_headless

ROOT = Path(__file__).resolve().parents[1]


def rollout(project):
    with open_headless(project) as run:
        rows = []
        for _ in range(400):
            run.step()
            rows.append(run.physics.data.qpos.copy())
        assert np.isfinite(rows).all()
        assert run.physics.data.qpos[2] > 0.10
        return np.array(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "tmp/botlab/before-shadow-fix/MicroDuck-Athletics.botlab.zip")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="botlab-field-qa-") as directory:
        base = Path(directory) / "before"
        updated = Path(directory) / "after"
        with ZipFile(args.baseline) as archive:
            archive.extractall(base)
        with ZipFile(ROOT / "output/MicroDuck-Athletics.botlab.zip") as archive:
            assert archive.testzip() is None
            archive.extractall(updated)
        models = [mujoco.MjModel.from_xml_path(str(p / "assets/robot/scene.xml")) for p in (base, updated)]
        before, after = models
        mask = (before.geom_contype | before.geom_conaffinity) != 0
        assert np.array_equal(mask, (after.geom_contype | after.geom_conaffinity) != 0)
        for field in ("geom_type", "geom_pos", "geom_quat", "geom_size", "geom_friction", "geom_condim", "geom_solref", "geom_solimp", "geom_contype", "geom_conaffinity"):
            np.testing.assert_array_equal(getattr(before, field)[mask], getattr(after, field)[mask], err_msg=field)
        for field in ("qpos0", "body_pos", "body_quat", "jnt_pos", "dof_damping", "actuator_gainprm", "actuator_biasprm"):
            np.testing.assert_array_equal(getattr(before, field), getattr(after, field), err_msg=field)
        # Visuals are excluded from contacts. Verify MuJoCo's compiled mesh is flat
        # and the mesh transform still places the top 0.1 mm above contact Z=0.
        data = mujoco.MjData(after)
        mujoco.mj_forward(after, data)
        geom = after.geom("hb_carpet_visual").id
        mesh = int(after.geom_dataid[geom])
        start, count = int(after.mesh_vertadr[mesh]), int(after.mesh_vertnum[mesh])
        vertices = after.mesh_vert[start:start + count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        np.testing.assert_allclose([vertices[:, 2].min(), vertices[:, 2].max()], [-0.0119, 0.0001], atol=1e-8)
        np.testing.assert_array_equal(before.mesh_texcoord, after.mesh_texcoord)
        old_rollout, new_rollout = rollout(base), rollout(updated)
        np.testing.assert_allclose(new_rollout, old_rollout, rtol=0, atol=1e-12)
        report = {
            "status": "passed", "mujoco_version": mujoco.__version__,
            "physical_collision_geoms_unchanged": int(mask.sum()),
            "visual_bounds_z_m": [float(vertices[:, 2].min()), float(vertices[:, 2].max())],
            "uv_unchanged": True, "standing_duration_s": 8,
            "max_qpos_difference": float(np.abs(old_rollout - new_rollout).max()),
            "final_base_pose": new_rollout[-1, :7].tolist(),
            "scope": "Same native runtime and policy, 400 control steps per package; no autonomous navigation claim.",
        }
        destination = ROOT / "tmp/botlab/shadow-fix-physics.json"
        destination.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
