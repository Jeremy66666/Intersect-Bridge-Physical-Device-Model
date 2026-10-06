"""Export existing point TXT files without changing their contents or order."""
import json
import os
from pathlib import Path
import sys

def export(folder):
    folder = Path(folder)
    target = folder / "laser_log_combined.txt"
    if target.exists():
        raise FileExistsError(target)
    files = sorted(folder.glob("point_*.json"))
    if not files:
        raise ValueError("No point records found")
    blocks = []
    for expected, file in enumerate(files, 1):
        record = json.loads(file.read_text(encoding="utf-8"))
        text = file.with_suffix(".txt").read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if line.startswith("Image Data")]
        if record["point_id"] != expected or record["status"] != "OK" or len(lines) != 1:
            raise ValueError(f"Invalid point: {file}")
        values = [int(value, 16) for value in lines[0][len("Image Data"):].split()]
        if len(values) != record["pixel_count"] or values != record["intensity"]:
            raise ValueError(f"TXT/JSON mismatch: {file}")
        blocks.append(text.rstrip() + "\nStop Capture\n\n")
    # Exclusive creation prevents overwriting a live or existing scan export.
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("".join(blocks))
        stream.flush()
        os.fsync(stream.fileno())
    print(f"Exported {len(blocks)} points: {target.resolve()}")

if __name__ == "__main__":
    export(sys.argv[1])
