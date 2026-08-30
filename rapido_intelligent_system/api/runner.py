"""Programmatic uvicorn runner used by the marimo server notebook.

Two entry points:

* ``serve_blocking`` — used when the notebook is run **as a script**: blocks on
  the server (Ctrl+C stops it cleanly).
* ``ServerHandle`` — used in **interactive** marimo mode: starts uvicorn in a
  daemon thread from a button and stops it on demand.
"""

from __future__ import annotations

import threading
from typing import Any

import uvicorn


def serve_blocking(app: Any, host: str, port: int) -> None:
    """Blocking server start (script mode)."""
    uvicorn.run(app, host=host, port=port, log_level="info")


class ServerHandle:
    """Non-blocking server on a daemon thread (interactive mode)."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def start(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        if self.is_running():
            return
        config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def is_running(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())