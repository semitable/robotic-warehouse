"""Append this turn verbatim with CRLF, preserving every existing conversation-log byte."""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
target = root / "CONVERSATION_LOG.md"
fragment = Path(__file__).with_name("T166_transcript.md").read_text(encoding="utf-8")
marker = fragment.splitlines()[0]
if marker in target.read_text(encoding="utf-8"):
    raise SystemExit("T166 already present; nothing appended")
payload = ("\r\n" + fragment.replace("\r\n", "\n").replace("\n", "\r\n")).encode("utf-8")
with target.open("ab") as stream:
    stream.write(payload)
print(f"Appended T166 ({len(payload)} bytes, CRLF)")
