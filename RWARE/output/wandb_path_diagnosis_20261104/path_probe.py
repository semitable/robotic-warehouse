"""Isolated stdlib-only long-path reproduction. Never writes into training runs."""
import json
import os
from pathlib import Path

out = Path(__file__).resolve().parent
long = out / ("x" * 85) / ("y" * 50) / "video_51200_01234567890123456789.mp4"
long.parent.mkdir(parents=True, exist_ok=True)
result = {"path_chars": len(str(long)), "parent_exists": long.parent.is_dir()}
(out / "short_marker.txt").write_bytes(b"diagnostic")
result["short_write"] = "success"
try:
    long.write_bytes(b"diagnostic")
    result["long_write"] = "success"
except OSError as exc:
    result["long_write"] = type(exc).__name__
    result["error"] = str(exc)
extended = "\\\\?\\" + str(long)
with open(extended, "xb") as stream:
    stream.write(b"diagnostic marker; not a video")
result["extended_write"] = "success"
(out / "path_probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
