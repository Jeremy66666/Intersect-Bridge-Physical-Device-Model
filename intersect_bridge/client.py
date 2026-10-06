"""Remote client; uses only the intersect configuration block, not local device paths."""
import argparse
import json
import threading
import uuid
from intersect_sdk import (IntersectClient, IntersectClientConfig, IntersectClientCallback, IntersectDirectMessageParams)
from common import configuration

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--timeout", type=float, default=30)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    start = sub.add_parser("start-scan")
    start.add_argument("--mode", required=True, choices=["1d-x", "1d-y", "2d"])
    start.add_argument("--request-id")
    cancel = sub.add_parser("cancel-scan")
    cancel.add_argument("session_id")
    cancel.add_argument("--request-id")
    results = sub.add_parser("results")
    results.add_argument("session_id")
    results.add_argument("--after-point", type=int, default=0)
    args = p.parse_args()
    cfg = configuration(args.config)["intersect"]
    hierarchy = cfg["hierarchy"]
    destination = ".".join(hierarchy.get(k) or "-" for k in ["organization", "facility", "system", "subsystem", "service"])
    payload = None
    operation = {"status":"get_status", "start-scan":"start_scan", "cancel-scan":"cancel_scan", "results":"get_results"}[args.command]
    if args.command in ("start-scan", "cancel-scan"):
        rid = args.request_id or uuid.uuid4().hex
        print("request_id=" + rid, flush=True)
        payload = dict(request_id=rid)
        payload.update(mode=args.mode) if args.command == "start-scan" else payload.update(session_id=args.session_id)
    elif args.command == "results":
        payload = dict(session_id=args.session_id, after_point=args.after_point)
    done = threading.Event()
    answer = []
    def callback(source, operation, has_error, value):
        answer.append((has_error, value))
        done.set()
        return None
    message = IntersectDirectMessageParams(destination=destination, operation="RamanBridge." + operation, payload=payload)
    config = IntersectClientConfig(brokers=cfg["brokers"],
        initial_message_event_config=IntersectClientCallback(messages_to_send=[message]))
    client = IntersectClient(config=config, user_callback=callback)
    try:
        client.startup()
        if not done.wait(args.timeout):
            raise TimeoutError("No response; state is uncertain. Retry mutations only with the SAME request ID.")
        error, value = answer[0]
        if isinstance(value, str):
            try: value = json.loads(value)
            except ValueError: pass
        print(json.dumps(dict(error=error, result=value), ensure_ascii=False, indent=2))
        if error:
            raise SystemExit(1)
    finally:
        client.shutdown()

if __name__ == "__main__":
    main()
