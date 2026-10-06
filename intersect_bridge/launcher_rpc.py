"""One INTERSECT request for the GUI; structured output stays separate from SDK logs."""
import argparse
import threading
import json
from intersect_sdk import IntersectClient, IntersectClientConfig, IntersectClientCallback, IntersectDirectMessageParams
from common import configuration, atomic_json, read_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    client = None
    try:
        req = read_json(args.request)
        cfg = configuration(args.config)["intersect"]
        hierarchy = cfg["hierarchy"]
        target = ".".join(hierarchy.get(k) or "-" for k in ("organization", "facility", "system", "subsystem", "service"))
        ready = threading.Event()
        received = []
        def callback(source, operation, has_error, payload):
            if source == target and operation == "RamanBridge." + req["operation"]:
                received.append((has_error, payload))
                ready.set()
            return None
        message = IntersectDirectMessageParams(destination=target,
            operation="RamanBridge." + req["operation"], payload=req.get("payload"))
        client = IntersectClient(config=IntersectClientConfig(brokers=cfg["brokers"],
            initial_message_event_config=IntersectClientCallback(messages_to_send=[message])), user_callback=callback)
        client.startup()
        if not ready.wait(req.get("timeout", 12)):
            raise TimeoutError("No INTERSECT response. The start result may be uncertain. Check status or retry with the same request ID.")
        error, payload = received[0]
        if isinstance(payload, str):
            try: payload = json.loads(payload)
            except ValueError: pass
        atomic_json(args.response, dict(error=error, result=payload))
    except Exception as exc:
        atomic_json(args.response, dict(error=True, result=str(exc)))
    finally:
        if client is not None:
            client.shutdown()

if __name__ == "__main__":
    main()
