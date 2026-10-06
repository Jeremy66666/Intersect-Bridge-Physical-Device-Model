"""Real MQTT/INTERSECT transport against a uniquely named fake service; no device access."""
import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from launcher_core import LauncherCore, load, save

ROOT = Path(__file__).resolve().parent

def serve(config):
    from intersect_sdk import IntersectService, IntersectServiceConfig, default_intersect_lifecycle_loop
    from service import RamanBridge
    class FakeQt:
        def status(self): return dict(scan_state="idle", scan_active=False, laser_on=False, test_only=True)
    class FakeController:
        qt = FakeQt()
        error = ""
        def results(self, session_id, after_point=0):
            return dict(session_id=session_id, analyzed_points=1, points=[] if after_point else [dict(point_id=1,label="TEST_ONLY")],next_after_point=1,analysis_error="")
    logging.basicConfig(level=logging.INFO)
    service = IntersectService([RamanBridge(FakeController())], IntersectServiceConfig(**load(config)["intersect"]))
    default_intersect_lifecycle_loop(service)

def test():
    with tempfile.TemporaryDirectory(dir=ROOT/"runtime/launcher") as temp:
        folder = Path(temp)
        config = load(ROOT/"config.json")
        config["intersect"]["hierarchy"]["service"] = "launcher-check-" + uuid.uuid4().hex[:8]
        save(folder/"config.json", config)
        for name in ("common.py", "launcher_rpc.py"):
            shutil.copyfile(ROOT/name, folder/name)
        core = LauncherCore(folder)
        core.python = Path(sys.executable)
        with (folder/"server.log").open("w") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--serve", str(folder/"config.json")],
                stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                deadline = time.monotonic()+40
                while True:
                    try:
                        status = core.rpc("get_status", timeout=3)
                        break
                    except Exception:
                        if time.monotonic() > deadline:
                            raise RuntimeError("Test service did not reply; broker may be unavailable")
                        time.sleep(1)
                assert status["qt"]["test_only"] is True
                result = core.results("test-session")
                assert result["points"][0]["label"] == "TEST_ONLY"
                print("PASS: real broker -> INTERSECT test service -> launcher RPC status/results. No hardware accessed.")
            finally:
                process.terminate()
                process.wait(timeout=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve")
    args = parser.parse_args()
    serve(args.serve) if args.serve else test()
