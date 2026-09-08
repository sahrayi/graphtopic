"""Validate and describe the locked paper-experiment environment."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
LOCK = ROOT / "environment-lock.json"


def _requirements(path: Path) -> dict[str, str]:
    pinned = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith(("#", "-")):
            name, version = line.split("==", 1)
            pinned[name] = version
    return pinned


def validate_reference_environment() -> dict:
    """Fail before downloading data when the numerical environment is not locked."""
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    required_python = lock["python"]
    actual_python = platform.python_version()
    failures = []
    if actual_python != required_python:
        failures.append(f"Python: expected {required_python}, found {actual_python}")
    requirements = _requirements(REPOSITORY / lock["requirements"])
    installed = {}
    for distribution, expected in requirements.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        installed[distribution] = actual
        if actual != expected:
            failures.append(f"{distribution}: expected {expected}, found {actual or 'missing'}")
    if failures:
        details = "\n  - ".join(failures)
        raise RuntimeError(
            "The paper runner requires its locked reference environment.\n"
            f"  - {details}\n"
            "Create it from requirements/experiments-reference.txt and rerun."
        )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
        "packages": installed,
        "lock": lock,
    }
