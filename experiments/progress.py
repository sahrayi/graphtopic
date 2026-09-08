"""Small flushed progress reporter shared by long-running paper commands."""

from __future__ import annotations

from datetime import datetime


def status(message: str) -> None:
    print(f"[{datetime.now().astimezone():%H:%M:%S}] {message}", flush=True)


class Progress:
    """Print start, integer-percentage changes, and completion without dependencies."""

    def __init__(self, label: str, total: int):
        self.label = label
        self.total = max(int(total), 1)
        self._last_percent = -1
        self.update(0)

    def update(self, completed: int, detail: str | None = None) -> None:
        completed = min(max(int(completed), 0), self.total)
        percent = completed * 100 // self.total
        if percent == self._last_percent and completed != self.total:
            return
        self._last_percent = percent
        suffix = f" - {detail}" if detail else ""
        status(f"{self.label}: {completed:,}/{self.total:,} ({percent}%){suffix}")
