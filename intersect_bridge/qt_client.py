"""File mailbox to the separately built Qt controller. No device SDK access."""
import time
import uuid
from pathlib import Path
from common import atomic_json, read_json, identifier

class QtClient:
    def __init__(self, directory):
        self.root = Path(directory)

    def status(self):
        path = self.root / "status.json"
        if not path.exists() or time.time() - path.stat().st_mtime > 10:
            raise RuntimeError("Bridge Qt is offline or its heartbeat is stale")
        state = read_json(path)
        if time.time() * 1000 - state["utc_ms"] > 10000:
            raise RuntimeError("Stale Qt heartbeat")
        return state

    def request(self, operation, request_id=None, **arguments):
        rid = identifier(request_id or uuid.uuid4().hex)
        response = self.root / "responses" / (rid + ".json")
        request_path = self.root / "requests" / (rid + ".json")
        args = dict(operation=operation, **arguments)
        if request_path.exists():
            old = read_json(request_path)
            if any(old.get(k) != v for k, v in args.items()) or set(old) - {"request_id", "boot_id", "expires_ms"} != set(args):
                raise ValueError("Request ID already belongs to a different command")
        else:
            state = self.status()
            atomic_json(request_path, dict(args, request_id=rid, boot_id=state["boot_id"],
                                           expires_ms=int(time.time() * 1000) + 10000))
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if response.exists():
                return read_json(response)
            time.sleep(.05)
        raise TimeoutError(f"Qt reply uncertain; query status or retry SAME request_id={rid}, do not submit a new start")
