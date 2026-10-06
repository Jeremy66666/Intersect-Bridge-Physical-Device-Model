"""Launcher orchestration. No laser-enable or low-level motor commands are exposed."""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
import psutil

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)

def child_env():
    env = os.environ.copy()
    # PyInstaller/PySide's environment must not leak into the existing Qt 6.6 app.
    for key in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "PYTHONHOME", "PYTHONPATH"):
        env.pop(key, None)
    if getattr(sys, "frozen", False):
        base = str(getattr(sys, "_MEIPASS", "")).lower()
        env["PATH"] = ";".join(p for p in env.get("PATH", "").split(";") if not (base and p.lower().startswith(base)))
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env

class LauncherCore:
    def __init__(self, root, emit=lambda text: None):
        self.root = Path(root).resolve()
        self.emit = emit
        self.python = self.root / ".venv/Scripts/python.exe"
        self.runtime = self.root / "runtime/launcher"
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.session_id = ""
        self.pending_start = None
        self.state_path = self.runtime / "state.json"
        if self.state_path.exists():
            state = load(self.state_path)
            self.session_id = state.get("session_id", "")
            self.pending_start = state.get("pending_start")

    def cfg(self):
        return load(self.root / "config.json")

    def path(self, name):
        return (self.root / self.cfg()[name]).resolve()

    def remember(self):
        save(self.state_path, dict(session_id=self.session_id, pending_start=self.pending_start))

    def log(self, text):
        secret = self.cfg()["intersect"]["brokers"][0].get("password", "")
        safe = str(text).replace(secret, "[password hidden]") if secret else str(text)
        self.emit(safe)

    def spawn(self, args, *, env=None, log_name=None, visible=False):
        output = subprocess.PIPE
        handle = None
        if log_name:
            handle = (self.runtime / log_name).open("a", encoding="utf-8")
            output = handle
        options = dict(cwd=self.root, env=env or child_env(), stdout=output, stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL)
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
            if not visible:
                info = subprocess.STARTUPINFO()
                info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                info.wShowWindow = 0
                options["startupinfo"] = info
        try:
            return subprocess.Popen([str(a) for a in args], **options)
        finally:
            if handle:
                handle.close()

    def run(self, args, timeout=30, env=None):
        process = self.spawn(args, env=env)
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise TimeoutError("Operation timed out. Device commands will not be resent automatically.")
        text = (output or b"").decode("utf-8", errors="replace")
        if process.returncode:
            raise RuntimeError(text[-3500:] or f"Process exit code {process.returncode}")
        return text

    def find_docker(self):
        candidates = [shutil.which("docker"),
            str(Path(os.environ.get("LOCALAPPDATA", ""))/"Programs/DockerDesktop/resources/bin/docker.exe"),
            "C:/Program Files/Docker/Docker/resources/bin/docker.exe"]
        for name in candidates:
            if name and Path(name).is_file():
                return Path(name)
        raise FileNotFoundError("Docker Desktop was not found. Install it or add docker.exe to PATH.")

    def rpc(self, operation, payload=None, timeout=12):
        uid = uuid.uuid4().hex
        request = self.runtime / (uid + ".request.json")
        response = self.runtime / (uid + ".response.json")
        save(request, dict(operation=operation, payload=payload, timeout=timeout))
        try:
            self.run([self.python, self.root/"launcher_rpc.py", "--config", self.root/"config.json",
                      "--request", request, "--response", response], timeout=timeout+15)
            answer = load(response)
            if answer["error"]:
                raise RuntimeError(str(answer["result"]))
            return answer["result"]
        finally:
            request.unlink(missing_ok=True)
            response.unlink(missing_ok=True)

    def local_status(self):
        path = self.path("ipc_dir") / "status.json"
        if path.exists():
            status = load(path)
            if time.time()*1000 - status.get("utc_ms", 0) < 10000:
                return status
        return None

    def service_processes(self):
        found = []
        for process in psutil.process_iter(["name", "cmdline"]):
            try:
                args = process.info["cmdline"] or []
                if not any(Path(arg).name.lower() == "service.py" for arg in args):
                    continue
                for arg in args:
                    if Path(arg).name.lower() == "service.py" and (Path(process.cwd()) / arg).resolve() == self.root/"service.py":
                        found.append(process)
                        break
            except (psutil.Error, OSError):
                continue
        return found

    def startup(self):
        for path in [self.python, self.root/"config.json", self.root/"launcher_rpc.py", self.root/"build_qt/bin/raman_intersect.exe"]:
            if not path.is_file():
                raise FileNotFoundError(f"Missing file: {path}. Keep the EXE in the intersect_bridge folder.")
        if not self.path("dark_file").is_file():
            raise FileNotFoundError("The configured dark spectrum file does not exist. Click Open Config and update dark_file.")
        docker = self.find_docker()
        self.log("Checking Docker engine...")
        try:
            self.run([docker, "info", "--format", "{{.ServerVersion}}"], timeout=12)
        except (RuntimeError, TimeoutError):
            desktop = docker.parents[2]/"Docker Desktop.exe"
            if not desktop.is_file():
                raise RuntimeError("Open Docker Desktop, wait for Engine running, then retry.")
            self.log("Starting Docker Desktop; waiting for the engine (up to 120 seconds)...")
            self.spawn([desktop], log_name="docker-desktop.log")
            deadline = time.monotonic()+120
            while True:
                try:
                    self.run([docker, "info", "--format", "{{.ServerVersion}}"], timeout=8)
                    break
                except (RuntimeError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise RuntimeError("Docker engine is not ready. Check Docker Desktop for errors, then click Start / Check System.")
                    time.sleep(2)
        self.log("Docker is ready. Starting the RabbitMQ broker...")
        env = child_env()
        env["RAMAN_BROKER_PASSWORD"] = self.cfg()["intersect"]["brokers"][0]["password"]
        self.run([docker, "compose", "--ansi", "never", "up", "-d"], timeout=180, env=env)
        deadline = time.monotonic()+90
        while True:
            try:
                self.run([docker, "compose", "exec", "-T", "broker", "rabbitmq-diagnostics", "-q", "check_running"], timeout=15, env=env)
                break
            except (RuntimeError, TimeoutError):
                if time.monotonic() > deadline:
                    raise RuntimeError("The broker is not ready. Check the broker logs in Docker.")
                time.sleep(2)
        self.log("The broker is ready.")
        if self.local_status():
            self.log("Reusing the running bridge Qt application.")
        else:
            self.log("Opening bridge Qt (no automatic laser activation or scanning)...")
            env = child_env()
            env["PATH"] = "C:/Qt/6.6.3/msvc2019_64/bin;" + env.get("PATH", "")
            env["QT_QPA_PLATFORM_PLUGIN_PATH"] = "C:/Qt/6.6.3/msvc2019_64/plugins/platforms"
            self.spawn([self.root/"build_qt/bin/raman_intersect.exe"], env=env, log_name="qt.log", visible=True)
            deadline = time.monotonic()+25
            while not self.local_status():
                if time.monotonic() > deadline:
                    raise RuntimeError("Qt is not ready. Check its window; close the original Qt application if it is using the devices.")
                time.sleep(.3)
        if self.service_processes():
            self.log("Reusing the existing service.py process (restart it after configuration changes).")
        else:
            self.log("Loading models and starting the INTERSECT service...")
            self.spawn([self.python, self.root/"service.py"], log_name="service.log")
        self.log("Querying Qt status through INTERSECT...")
        deadline = time.monotonic()+50
        while True:
            try:
                result = self.rpc("get_status", timeout=6)
                if result.get("offline"):
                    raise RuntimeError(str(result))
                self.log("Startup complete. Prepare the optical setup, enable the laser, and allow remote scans in Qt.")
                return result
            except (RuntimeError, TimeoutError, FileNotFoundError):
                if time.monotonic() > deadline:
                    raise RuntimeError("No INTERSECT response. Check the configured password and runtime/launcher/service.log. No duplicate service will be started.")
                time.sleep(1)

    def scan(self, mode):
        state = self.rpc("get_status")
        qt = state.get("qt", {})
        if not self.pending_start:
            if not qt.get("laser_on") or not qt.get("remote_scan_enabled") or qt.get("scan_active") or qt.get("backend_fault") or qt.get("laser_state_uncertain"):
                raise RuntimeError("Cannot scan yet. In Qt, check that the laser is on, remote scans are enabled, Monitor is off, and no scan or device fault is active.")
            self.pending_start = dict(mode=mode, request_id="gui-"+uuid.uuid4().hex)
            self.remember()
        elif self.pending_start["mode"] != mode:
            raise RuntimeError("The previous start result is uncertain. Retry the same request using the previous scan mode.")
        self.log("Sending start request: " + self.pending_start["request_id"])
        result = self.rpc("start_scan", self.pending_start, timeout=25)
        if not result.get("ok"):
            self.pending_start = None
            self.remember()
            raise RuntimeError(result.get("error", "Qt rejected the start request"))
        self.session_id = result["session_id"]
        self.pending_start = None
        self.remember()
        self.log("Scan started. Session: " + self.session_id)
        return result

    def restart_service(self):
        status = self.local_status()
        if not status:
            raise RuntimeError("Open bridge Qt and confirm its local status before restarting the analysis service.")
        if status.get("scan_active") or status.get("collecting"):
            raise RuntimeError("Scanning or acquisition is active. Wait for it to finish before restarting the analysis service.")
        if not self.path("dark_file").is_file():
            raise FileNotFoundError("The configured dark spectrum file does not exist. Keeping the existing service.")
        self.log("Checking the new configuration and models...")
        self.run([self.python, self.root/"service.py", "--check"], timeout=60)
        for process in self.service_processes():
            try:
                process.terminate()
                process.wait(timeout=10)
            except psutil.NoSuchProcess:
                pass
        self.log("Restarting the analysis service to load the current configuration. Qt and laser states are unchanged.")
        self.spawn([self.python, self.root/"service.py"], log_name="service.log")
        deadline = time.monotonic()+45
        while True:
            try: return self.rpc("get_status", timeout=6)
            except (RuntimeError, TimeoutError):
                if time.monotonic() > deadline:
                    raise RuntimeError("No service response after restart. Check service.log.")
                time.sleep(1)

    def cancel(self):
        status = self.rpc("get_status")
        qt = status.get("qt", {})
        if not qt.get("scan_active"):
            raise RuntimeError("No scan is currently running.")
        result = self.rpc("cancel_scan", dict(session_id=qt["session_id"], request_id="cancel-"+uuid.uuid4().hex), timeout=25)
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error", result)))
        self.log("Cancellation completed. Laser state is unchanged; turn off the laser in Qt.")
        return result

    def results(self, session_id):
        if not session_id:
            raise ValueError("Start a scan from this launcher or enter an existing INTERSECT session ID.")
        points = []
        after = 0
        for page in range(30):
            result = self.rpc("get_results", dict(session_id=session_id, after_point=after))
            points.extend(result["points"])
            following = result["next_after_point"]
            if not result["points"] or len(points) >= result["analyzed_points"] or following <= after:
                break
            after = following
        result["points"] = points
        return result
