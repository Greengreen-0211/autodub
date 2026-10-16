"""Optional command-backed secondary ASR for unresolved dialogue only.

Configure ``ASR_SECONDARY_COMMAND_JSON`` as a JSON argv list containing an
``{audio}`` placeholder.  The command may print either plain transcript text
or a JSON object with ``text``, optional ``language``, and optional
``timestamps``.  No shell is used, so API keys and paths are not interpolated
through a command line shell.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, List, Optional


class SecondaryASR:
    def __init__(self) -> None:
        raw = os.getenv("ASR_SECONDARY_COMMAND_JSON", "").strip()
        self.argv: List[str] = []
        self.error: Optional[str] = None
        if not raw:
            return
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list) or not parsed or not all(
                isinstance(item, str) for item in parsed
            ):
                raise ValueError("must be a non-empty JSON string array")
            if not any("{audio}" in item for item in parsed):
                raise ValueError("missing {audio} placeholder")
            self.argv = parsed
        except Exception as exc:
            self.error = f"invalid_secondary_command:{exc}"

    @property
    def available(self) -> bool:
        return bool(self.argv) and self.error is None

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        if not self.available:
            return {"text": "", "error": self.error or "not_configured"}
        argv = [item.replace("{audio}", audio_path) for item in self.argv]
        timeout = max(10.0, float(os.getenv("ASR_SECONDARY_TIMEOUT", "180")))
        try:
            completed = subprocess.run(
                argv,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except Exception as exc:
            return {"text": "", "error": f"{type(exc).__name__}:{exc}"}
        output = completed.stdout.strip()
        if not output:
            return {"text": "", "error": "empty_output"}
        try:
            payload = json.loads(output.splitlines()[-1])
        except json.JSONDecodeError:
            payload = {"text": output}
        if not isinstance(payload, dict):
            payload = {"text": str(payload)}
        payload["text"] = str(payload.get("text") or "").strip()
        return payload
