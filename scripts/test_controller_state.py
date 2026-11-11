from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.controller_state import factory_state_evidence


def mode(before, after, node='/sim2real_master_node'):
    return '1.0 INFO '+node+' [fsm.h:86(ModeFsmDef::print)] [topics: /rosout] from '+before+' to '+after+'\n'


def child(before, after, node='/sim2real_master_node'):
    return '2.0 INFO '+node+' [sim2real.cpp:1875(Sim2Real::execPdLoop)] [topics: /rosout] State changed: '+before+' -> '+after+'\n'


class ControllerStateTests(unittest.TestCase):
    def test_factory_standby_is_recovered(self):
        self.assertEqual(factory_state_evidence(mode('CandidateDefault', 'ExecDefault')+child('init', 'standing')+child('standing', 'standby'))['child_state'], 'standby')

    def test_protection_and_reentry_invalidate_previous_standby(self):
        log=mode('CandidateDefault', 'ExecDefault')+child('standing', 'standby')+mode('ExecDefault', 'ProtectionShutdown')+mode('CandidateDefault', 'ExecDefault')
        self.assertIsNone(factory_state_evidence(log)['child_state'])

    def test_other_nodes_cannot_supply_factory_state(self):
        self.assertIsNone(factory_state_evidence(mode('CandidateDefault', 'ExecDefault')+child('standing', 'standby', '/another_node'))['child_state'])

    def test_missing_mode_trace_fails_closed(self):
        self.assertIsNone(factory_state_evidence(child('standing', 'standby'))['child_state'])


if __name__ == '__main__':
    unittest.main()
