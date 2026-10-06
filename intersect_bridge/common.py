"""Local configuration and atomic files; importing this module never starts hardware."""
import json
import os
import re
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")

def identifier(value):
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise ValueError("Invalid identifier")
    return value

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    try:
        for attempt in range(20):
            try:
                os.replace(temp, path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(.05)
    finally:
        temp.unlink(missing_ok=True)

def configuration(path=None):
    path = Path(path or ROOT / "config.json")
    value = read_json(path)
    for key in ("ipc_dir", "scan_root", "model_dir", "dark_file", "results_dir"):
        if value.get(key):
            candidate = Path(value[key])
            value[key] = str((path.parent / candidate).resolve() if not candidate.is_absolute() else candidate)
    return value
