"""Launcher checks with mocked processes and network; no hardware is started."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from launcher_core import LauncherCore, save

class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        save(self.root/"config.json", dict(ipc_dir="ipc",dark_file="dark.txt",results_dir="results",
            intersect=dict(brokers=[dict(password="secret-value")])) )
        self.messages = []
        self.core = LauncherCore(self.root, self.messages.append)

    def make_files(self):
        for name in (".venv/Scripts/python.exe", "launcher_rpc.py", "build_qt/bin/raman_intersect.exe", "dark.txt"):
            p = self.root/name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()

    def test_startup_reuses_existing_qt_and_service(self):
        self.make_files()
        self.core.find_docker = Mock(return_value=Path("docker.exe"))
        self.core.run = Mock(return_value="ok")
        self.core.local_status = Mock(return_value=dict(scan_active=False))
        self.core.service_processes = Mock(return_value=[object()])
        self.core.rpc = Mock(return_value=dict(qt=dict(scan_state="idle")))
        self.core.spawn = Mock()
        self.assertEqual(self.core.startup()["qt"]["scan_state"], "idle")
        self.core.spawn.assert_not_called()
        self.assertEqual(self.core.rpc.call_args.args[0], "get_status")

    def test_startup_never_sends_motion_or_laser_commands(self):
        self.make_files()
        self.core.find_docker = Mock(return_value=Path("docker.exe"))
        self.core.run = Mock(return_value="ok")
        self.core.local_status = Mock(side_effect=[None, dict(scan_active=False)])
        self.core.service_processes = Mock(return_value=[])
        self.core.rpc = Mock(return_value=dict(qt=dict(scan_state="idle")))
        self.core.spawn = Mock()
        self.core.startup()
        self.assertEqual(self.core.spawn.call_count, 2)
        self.assertEqual([c.args[0] for c in self.core.rpc.call_args_list], ["get_status"])

    def test_uncertain_start_keeps_id_across_launcher_restart(self):
        ready = dict(qt=dict(laser_on=True, remote_scan_enabled=True,scan_active=False))
        self.core.rpc = Mock(side_effect=[ready, TimeoutError("uncertain")])
        with self.assertRaises(TimeoutError): self.core.scan("1d-y")
        pending = dict(self.core.pending_start)
        reopened = LauncherCore(self.root)
        self.assertEqual(reopened.pending_start, pending)
        reopened.rpc = Mock(side_effect=[ready, dict(ok=True,session_id="session-one")])
        reopened.scan("1d-y")
        self.assertEqual(reopened.rpc.call_args.args[1], pending)
        self.assertIsNone(reopened.pending_start)

    def test_scan_requires_ready_state(self):
        self.core.rpc = Mock(return_value=dict(qt=dict(laser_on=False, remote_scan_enabled=True)))
        with self.assertRaises(RuntimeError): self.core.scan("1d-y")
        self.assertEqual(self.core.rpc.call_count, 1)

    def test_cancel_uses_current_session(self):
        self.core.rpc = Mock(side_effect=[dict(qt=dict(scan_active=True,session_id="current")),dict(ok=True)])
        self.core.cancel()
        self.assertEqual(self.core.rpc.call_args.args[1]["session_id"], "current")

    def test_results_pagination(self):
        self.core.rpc = Mock(side_effect=[dict(points=[dict(point_id=1)],analyzed_points=2,next_after_point=1),
            dict(points=[dict(point_id=2)],analyzed_points=2,next_after_point=2)])
        self.assertEqual(len(self.core.results("session")["points"]),2)

    def test_secret_not_logged(self):
        self.core.log("failed secret-value")
        self.assertNotIn("secret-value", self.messages[0])

    def test_restart_refused_during_acquisition(self):
        self.core.local_status = Mock(return_value=dict(scan_active=True))
        self.core.service_processes = Mock()
        with self.assertRaises(RuntimeError): self.core.restart_service()
        self.core.service_processes.assert_not_called()

if __name__ == "__main__":
    unittest.main(verbosity=2)
