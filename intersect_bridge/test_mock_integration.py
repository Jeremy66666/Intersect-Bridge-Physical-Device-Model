"""Runs ONLY generated_mock Qt and mock SDK-free backend/motor executables."""
import os
import subprocess
import time
import uuid
import psutil
from common import ROOT, configuration, atomic_json
from qt_client import QtClient
from controller import Controller

def wait(predicate, seconds=40):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            result = predicate()
            if result: return result
        except (RuntimeError, FileNotFoundError):
            pass
        time.sleep(.1)
    raise AssertionError("Timed out waiting for mock state")

def main():
    cfg = configuration()
    base = ROOT / "build_mock/bin"
    cfg.update(ipc_dir=str(base/"ipc"), scan_root=str(base/"scan_sessions"),
               results_dir=str(ROOT/"runtime/mock_results"))
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_OPENGL="software",
               QT_QPA_PLATFORM_PLUGIN_PATH="C:/Qt/6.6.3/msvc2019_64/plugins/platforms")
    env["PATH"] = "C:/Qt/6.6.3/msvc2019_64/bin;" + env["PATH"]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    log = ROOT/"runtime/mock_qt.log"
    log.parent.mkdir(exist_ok=True)
    qt = QtClient(cfg["ipc_dir"])
    old_boot = None
    try: old_boot = qt.status()["boot_id"]
    except RuntimeError: pass
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen([str(base/"raman_intersect.exe")], env=env, stdout=output, stderr=output, creationflags=flags)
        try:
            wait(lambda: qt.status()["boot_id"] != old_boot)
            controller = Controller(cfg)
            rid = uuid.uuid4().hex
            reply = controller.start_scan("1d-y", rid)
            assert reply["ok"], reply
            sid = reply["session_id"]
            wait(lambda: qt.status()["scan_state"] == "completed")
            assert qt.status()["last_completed_point"] == 13
            repeated = controller.start_scan("1d-y", rid)
            assert repeated["session_id"] == sid
            assert not qt.status()["scan_active"], "Duplicate command restarted motor"
            controller.process_once()
            assert controller.results(sid)["analyzed_points"] == 13
            # An expired command must never start a second scan.
            expired = uuid.uuid4().hex
            atomic_json(base/"ipc/requests"/(expired+".json"), dict(request_id=expired,
                operation="start_scan", mode="1d-y", boot_id=qt.status()["boot_id"], expires_ms=0))
            wait(lambda: (base/"ipc/responses"/(expired+".json")).exists())
            assert not qt.request("start_scan", expired, mode="1d-y")["ok"]
            second = controller.start_scan("1d-x", uuid.uuid4().hex)
            assert second["ok"], second
            cancelled = qt.request("cancel_scan", session_id=second["session_id"])
            assert cancelled["ok"], cancelled
            wait(lambda: not qt.status()["scan_active"])
            print("PASS: mock Qt -> 13 points -> SVM; duplicate/expiry protection; cancellation. No real hardware used.")
        finally:
            children = psutil.Process(process.pid).children(recursive=True) if process.poll() is None else []
            process.terminate()
            process.wait(timeout=10)
            for child in children:
                try: child.terminate()
                except psutil.NoSuchProcess: pass
            psutil.wait_procs(children, timeout=5)

if __name__ == "__main__":
    main()
