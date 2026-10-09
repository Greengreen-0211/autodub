"""Save an immutable source/UI snapshot before editing; never include local runtime data."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def backup(label):
    if not label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in label):
        raise ValueError("Use a lowercase label containing letters, numbers, - or _.")
    files = sorted({p for pattern in ("*.py", "*.css", "requirements*.txt", "*.md", ".gitignore",
                                      "tests/*.py", "examples/*.png", "examples/*.json")
                    for p in ROOT.glob(pattern) if p.is_file()})
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = ROOT / "backups" / f"frontend_{stamp}_{label}.zip"
    destination.parent.mkdir(exist_ok=True)
    manifest = {"label": label, "created_at": datetime.now().astimezone().isoformat(),
                "scope": "frontend source, checks, requirements and preview artifacts; no runtime DB or environment",
                "files": []}
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            data = path.read_bytes()
            relative = path.relative_to(ROOT).as_posix()
            archive.writestr(relative, data)
            manifest["files"].append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        archive.writestr("BACKUP_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Backup verification failed.")
        for entry in manifest["files"]:
            if hashlib.sha256(archive.read(entry["path"])).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Backup hash mismatch: {entry['path']}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    print(backup(parser.parse_args().label))
