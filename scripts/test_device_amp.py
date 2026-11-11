"""Device inference parity and controller interface regressions."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import unittest
import numpy as np
import torch
import yaml
from sim.device_amp import DEVICE, DeviceAMP
from sim.robot import DEFAULT, ISAAC_TO_MJ, MJ_TO_ISAAC, Robot, build_model
from scripts.export_device_amp import extract


class DeviceAMPTests(unittest.TestCase):
    def test_exact_device_weights_match_independent_jetson_inference(self):
        controller = DeviceAMP()
        with np.load(DEVICE/'engine_reference.npz', allow_pickle=False) as reference:
            with torch.inference_mode():
                actual = controller.actor(torch.from_numpy(reference['inputs'])).numpy()
            np.testing.assert_allclose(actual, reference['outputs'], atol=2e-5, rtol=2e-5)

    def test_export_rejects_another_engine_revision(self):
        raw = bytearray((DEVICE/'amp.trt').read_bytes())
        raw[225] ^= 1
        with self.assertRaisesRegex(ValueError, 'Unrecognized engine'):
            extract(raw)

    def test_running_controller_trace_matches_history_action_and_targets(self):
        c = DeviceAMP()
        pd = yaml.safe_load((DEVICE/'pd.yaml').read_text())
        to_pd = [c.config['joint_names'].index(n) for n in pd['joint_names'][:20]]
        with np.load(DEVICE/'runtime_reference.npz', allow_pickle=False) as trace:
            # This fixture came from the already-running controller's read-only
            # debug topics, joined by sequence number, not from our adapter.
            np.testing.assert_array_equal(np.diff(trace['seq']), 1)
            np.testing.assert_array_equal(trace['input'][:,-69:], trace['obs'])
            np.testing.assert_array_equal(trace['input'][1:,:-69], trace['input'][:-1,69:])
            np.testing.assert_array_equal(trace['obs'][1:,49:], trace['action'][:-1])
            for observations, action, output in zip(trace['input'], trace['action'], trace['output_q']):
                actual = c.infer(observations.astype(np.float32)[None])
                np.testing.assert_allclose(actual, action, atol=1e-6, rtol=1e-6)
                # Factory debug output_q is in PD order, before motor zero/sign
                # conversion. Offset-free target displacement must match it.
                np.testing.assert_allclose((c.target()-c.offset)[to_pd], output, atol=1e-6, rtol=1e-6)

    def test_factory_filter_deadband_and_stop_decay(self):
        c = DeviceAMP()
        # This factory behavior is intentional: sub-threshold commands do not
        # accumulate through the filter. It must not be silently compensated.
        for _ in range(50):
            np.testing.assert_array_equal(c.filter_command([.25,.25,.55]), [0,0,0])
        np.testing.assert_allclose(c.filter_command([.4,-.4,.8]), [.12,-.12,.24])
        np.testing.assert_allclose(c.filter_command([0,0,0]), [.084,-.084,0])
        np.testing.assert_array_equal(c.filter_command([0,0,0]), [0,0,0])
        for command in ([float('nan'),0,0], [1,2]):
            with self.assertRaises(ValueError): c.filter_command(command)

    def test_observation_order_limits_history_and_last_action(self):
        c = DeviceAMP()
        c.action[:] = np.linspace(-2,2,20)
        positions = c.offset + np.linspace(-.2,.2,20)
        obs = c.observe([25,-25,1],[0,0,-1],[.4,0,.8],positions,np.arange(20)).reshape(5,69)
        np.testing.assert_array_equal(obs[:-1], 0)
        np.testing.assert_allclose(obs[-1,:9], [18,-18,1,0,0,-1,.12,0,.24])
        np.testing.assert_allclose(obs[-1,9:29], positions-c.offset, atol=1e-8)
        np.testing.assert_array_equal(obs[-1,29:49], np.minimum(np.arange(20),18))
        np.testing.assert_allclose(obs[-1,49:], c.action)
        next_obs = c.observe([0,0,0],[0,0,-1],[0,0,0],c.offset,np.zeros(20)).reshape(5,69)
        np.testing.assert_array_equal(next_obs[-2], obs[-1])

    def test_mujoco_adapter_uses_device_gains_and_joint_displacements(self):
        model,_ = build_model(no_obstacles=True)
        robot = Robot(model,[.55,-1.25,0])
        self.assertEqual(robot.substeps, 20)
        np.testing.assert_array_equal(robot.kd[MJ_TO_ISAAC], robot.device.config['kd'])
        np.testing.assert_array_equal(robot.kp[MJ_TO_ISAAC], robot.device.config['kp'])
        # Raw encoder shoulder offsets must not become MJCF shoulder angles.
        robot.device.action[:] = np.linspace(-1,1,20)
        target = DEFAULT+(robot.device.target()-robot.device.offset)[ISAAC_TO_MJ]
        np.testing.assert_allclose(target, DEFAULT+.25*robot.device.action[ISAAC_TO_MJ])
        robot.step([0,0,0])
        np.testing.assert_allclose(robot.history[-60:-40], 0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
