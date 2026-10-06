"""Persistent scan bindings and background inference; independent of INTERSECT transport."""
import threading
from pathlib import Path
from common import atomic_json, read_json, identifier
from qt_client import QtClient
from svm_predictor import Predictor

class Controller:
    def __init__(self, config):
        self.config = config
        self.qt = QtClient(config["ipc_dir"])
        self.predictor = Predictor(config["model_dir"], config["dark_file"])
        self.root = Path(config["results_dir"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.scan_root = Path(config["scan_root"]).resolve()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.error = ""
        self.thread = threading.Thread(target=self._worker, daemon=True)

    def start_scan(self, mode, request_id):
        if mode not in ("1d-x", "1d-y", "2d"):
            raise ValueError("Unsupported scan mode")
        with self.lock:
            reply = self.qt.request("start_scan", request_id, mode=mode)
            if reply.get("ok"):
                self.bind(reply["session_id"], reply["session_dir"])
            return reply

    def bind(self, session_id, directory):
        sid = identifier(session_id)
        directory = Path(directory).resolve()
        if directory.parent != self.scan_root:
            raise ValueError("Qt returned a scan outside configured scan_root")
        path = self.root / sid / "binding.json"
        binding = dict(session_id=sid, directory=str(directory), model_sha256=self.predictor.version,
                       dark_sha256=self.predictor.dark_hash)
        if path.exists() and read_json(path) != binding:
            raise ValueError("Session already bound to different model/dark data")
        atomic_json(path, binding)

    def process_once(self):
        for binding_path in self.root.glob("*/binding.json"):
            binding = read_json(binding_path)
            if (binding["model_sha256"], binding["dark_sha256"]) != (self.predictor.version, self.predictor.dark_hash):
                continue  # Never mix model/dark versions within a session.
            directory = Path(binding["directory"]).resolve()
            if directory.parent != self.scan_root:
                raise ValueError("Invalid scan binding")
            for point in sorted(directory.glob("point_*.json")):
                output = binding_path.parent / point.name
                if output.exists():
                    continue
                record = read_json(point)
                if record["session_id"] != binding["session_id"] or point.name != f"point_{int(record['point_id']):04d}.json":
                    raise ValueError("Point/session mismatch")
                atomic_json(output, self.predictor.predict(record))

    def results(self, session_id, after_point=0):
        path = self.root / identifier(session_id)
        if not (path / "binding.json").exists():
            raise ValueError("Unknown analysis session")
        rows = [read_json(p) for p in sorted(path.glob("point_*.json"))]
        selected = sorted((r for r in rows if r["point_id"] > after_point), key=lambda r:r["point_id"])[:50]
        return dict(session_id=session_id, analyzed_points=len(rows), points=selected,
                    next_after_point=selected[-1]["point_id"] if selected else after_point,
                    analysis_error=self.error)

    def _worker(self):
        while not self.stop.is_set():
            try:
                self.process_once()
                self.error = ""
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            self.stop.wait(.5)
