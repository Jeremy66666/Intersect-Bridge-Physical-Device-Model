"""INTERSECT entry point. Requires the bridge Qt, broker and configured dark/model files."""
import argparse
import json
import logging
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict
from intersect_sdk import (IntersectBaseCapabilityImplementation, IntersectService,
    IntersectServiceConfig, intersect_message, intersect_status, default_intersect_lifecycle_loop)
from common import configuration
from controller import Controller

class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["1d-x", "1d-y", "2d"]
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")

class CancelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")

class ResultsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    after_point: int = Field(default=0, ge=0)

def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)

class RamanBridge(IntersectBaseCapabilityImplementation):
    intersect_sdk_capability_name = "RamanBridge"

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    @intersect_status()
    def status(self) -> str:
        try:
            return encode(dict(qt=self.controller.qt.status(), analysis_error=self.controller.error))
        except Exception as exc:
            return encode(dict(offline=True, error=str(exc)))

    @intersect_message()
    def get_status(self) -> str:
        return self.status()

    @intersect_message()
    def start_scan(self, request: StartRequest) -> str:
        return encode(self.controller.start_scan(request.mode, request.request_id))

    @intersect_message()
    def cancel_scan(self, request: CancelRequest) -> str:
        return encode(self.controller.qt.request("cancel_scan", request.request_id, session_id=request.session_id))

    @intersect_message()
    def get_results(self, request: ResultsRequest) -> str:
        return encode(self.controller.results(request.session_id, request.after_point))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--check", action="store_true", help="Validate models and service schema without connecting to devices/broker")
    args = parser.parse_args()
    config = configuration(args.config)
    controller = Controller(config)
    service = IntersectService([RamanBridge(controller)], IntersectServiceConfig(**config["intersect"]))
    if args.check:
        print("Model loading and INTERSECT service/schema construction passed. Hardware not started.")
        return
    controller.thread.start()
    try:
        default_intersect_lifecycle_loop(service)
    finally:
        controller.stop.set()
        controller.thread.join(timeout=5)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
