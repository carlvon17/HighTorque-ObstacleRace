"""Start options reach both consumers before the detached lap may move."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy import navigation_cli as cli
from deploy.box_panel import BoxPanel, VisionMailbox
from navigation.box_servo import BoxServo
from navigation.box_tuning import BoxTuning


class StartTuningTest(unittest.TestCase):
    def test_start_reaches_panel_and_controller_then_allows_live_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'deploy').mkdir()
            pose = root/'deploy/task_head_pose.json'
            pose.write_text(json.dumps({'measured_joint_reference_rad': {}}))
            config = root/'deploy/pro_model_config.json'
            config.write_text('{}')
            expected = dict(asdict(BoxTuning()), side_step=3., forward_speed=21., first_gap=40.5)

            def detached(command, **kwargs):
                output = Path(command[command.index('--output')+1])
                panel = BoxPanel(output, '127.0.0.1', 0, root/'no-key').start()
                try:
                    state = panel.state()
                    self.assertEqual(state['defaults'], asdict(BoxTuning()))
                    self.assertEqual((state['revision'], state['desired']), (1, expected))
                    servo = BoxServo()
                    mailbox = VisionMailbox(output/'vision')
                    mailbox.update(servo, 0.)
                    self.assertEqual(panel.state()['control']['applied'], expected)
                    changed = dict(expected, side_step=3.5)
                    self.assertEqual(panel.submit(dict(base=1, values=changed))[0], 200)
                    mailbox.update(servo, 1.)
                    self.assertEqual(servo.panel_status(1.)['applied'], changed)
                    with self.assertRaises(FileExistsError):
                        cli.start_lap(root, output.parent, output.parent/'unused', config, .22, BoxTuning())
                    self.assertEqual(panel.state()['control']['applied'], changed)
                finally:
                    panel.close()
                (output/'launch.json').write_text(json.dumps(dict(status='launched', pid=123)))
                return subprocess.CompletedProcess(command, 0, stdout='', stderr='')

            with patch.object(cli, 'ROOT', root), patch.object(cli, 'CLI_LOCK', root/'cli.lock'), \
                    patch('navigation.release.verify_release'), patch.object(cli, 'validate_start'), \
                    patch.object(cli.subprocess, 'run', side_effect=detached), \
                    patch.object(cli, 'follow_lap', return_value=0), redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(['start', '--config', str(config), '--speed', '.22',
                                           '--box-side-step', '3', '--box-forward-speed', '21',
                                           '--box-first-gap', '40.5']), 0)
            record = json.loads((root/'runs/terminal/latest.json').read_text())
            self.assertEqual(record['box_start_values'], expected)

    def test_invalid_options_stop_before_launch(self):
        cases = [('start','--box-side-step','9.1'), ('start','--box-forward-step','nan'),
                 ('start','--box-side-speed','20.5'), ('start','--box-forward-speed','19'),
                 ('start','--box-first-gap','39.1'), ('stop','--box-side-step','3')]
        for args in cases:
            with self.subTest(args=args), patch.object(cli, 'start_lap') as start, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as refused:
                cli.main(args)
            self.assertEqual(refused.exception.code, 2)
            start.assert_not_called()
