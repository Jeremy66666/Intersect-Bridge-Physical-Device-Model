"""Offline inference on an existing session; does not open Qt or control devices."""
import argparse
from pathlib import Path
from common import configuration, read_json, atomic_json, ROOT
from svm_predictor import Predictor

def main():
    p = argparse.ArgumentParser()
    p.add_argument("session_directory")
    p.add_argument("--config")
    p.add_argument("--output", default=str(ROOT / "runtime/replay.json"))
    args = p.parse_args()
    cfg = configuration(args.config)
    predictor = Predictor(cfg["model_dir"], cfg["dark_file"])
    rows = [predictor.predict(read_json(path)) for path in sorted(Path(args.session_directory).glob("point_*.json"))]
    if not rows:
        raise ValueError("No point JSON files found")
    atomic_json(args.output, dict(points=rows, note="Pure-class scores are not accuracy or evidence of chemical purity."))
    print(f"Analyzed {len(rows)} points -> {args.output}")

if __name__ == "__main__":
    main()
