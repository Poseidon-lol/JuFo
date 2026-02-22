"""Small local HTTP server helpers for live dashboard display."""

from __future__ import annotations

import contextlib
import logging
import threading
import urllib.parse
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Tuple


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # pragma: no cover - cosmetic
        return


def _normalise_host(host: str) -> str:
    host = (host or "127.0.0.1").strip()
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host


def start_dashboard_server(
    html_path: Path | str,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = False,
    start_thread: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[ThreadingHTTPServer, str]:
    """Start a background local file server for a dashboard HTML file.

    Returns (server, url). The caller should keep ``server`` referenced.
    """

    logger = logger or logging.getLogger(__name__)
    html_path = Path(html_path).resolve()
    html_path.parent.mkdir(parents=True, exist_ok=True)
    if not html_path.exists():
        html_path.write_text(
            "<!doctype html><html><body><h3>Warte auf Dashboard-Updates...</h3></body></html>",
            encoding="utf-8",
        )

    handler = partial(_QuietHandler, directory=str(html_path.parent))
    server = ThreadingHTTPServer((host, int(port)), handler)
    server.daemon_threads = True

    if start_thread:
        thread = threading.Thread(target=server.serve_forever, daemon=True, name="jtvae-live-dashboard-server")
        thread.start()

    bind_host, bind_port = server.server_address[:2]
    display_host = _normalise_host(str(bind_host))
    rel_name = urllib.parse.quote(html_path.name)
    url = f"http://{display_host}:{bind_port}/{rel_name}"
    logger.info("Live dashboard server running at %s", url)
    if open_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(url, new=1, autoraise=True)
    return server, url


def serve_dashboard_blocking(
    html_path: Path | str,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Serve a dashboard file until interrupted with Ctrl+C."""

    logger = logger or logging.getLogger(__name__)
    server, url = start_dashboard_server(
        html_path,
        host=host,
        port=port,
        open_browser=open_browser,
        start_thread=False,
        logger=logger,
    )
    logger.info("Serving dashboard. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping dashboard server.")
    finally:
        with contextlib.suppress(Exception):
            server.shutdown()
        with contextlib.suppress(Exception):
            server.server_close()
    return url
